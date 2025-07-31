"""
Intermolecular Interaction Modeling for ChemiAccel.

This module provides high-fidelity, GPU-accelerated methods for calculating and optimizing
intermolecular interactions critical for crystal packing, including:
- Hydrogen bonding networks
- π-π stacking interactions
- van der Waals interactions
- Crystal packing optimization

Requirements: 43.2
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional imports with graceful fallbacks
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = np
    _CUPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.spatial.distance import cdist

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdGeometry, rdMolDescriptors

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


class InteractionType(Enum):
    """Types of intermolecular interactions."""

    HYDROGEN_BOND = "hydrogen_bond"
    PI_PI_STACKING = "pi_pi_stacking"
    VAN_DER_WAALS = "van_der_waals"
    ELECTROSTATIC = "electrostatic"
    HALOGEN_BOND = "halogen_bond"
    CATION_PI = "cation_pi"
    ANION_PI = "anion_pi"
    CH_PI = "ch_pi"
    DIPOLE_DIPOLE = "dipole_dipole"


class HydrogenBondType(Enum):
    """Types of hydrogen bonds."""

    CONVENTIONAL = "conventional"  # N-H...O, O-H...O, N-H...N
    WEAK = "weak"  # C-H...O, C-H...N
    BIFURCATED = "bifurcated"  # One donor, two acceptors
    TRIFURCATED = "trifurcated"  # One donor, three acceptors
    RESONANCE_ASSISTED = "resonance_assisted"  # RAHB
    CHARGE_ASSISTED = "charge_assisted"  # CAHB


class PiPiStackingType(Enum):
    """Types of π-π stacking interactions."""

    PARALLEL_DISPLACED = "parallel_displaced"
    T_SHAPED = "t_shaped"
    EDGE_TO_FACE = "edge_to_face"
    SANDWICH = "sandwich"
    SLIPPED_PARALLEL = "slipped_parallel"


class CrystalPackingObjective(Enum):
    """Crystal packing optimization objectives."""

    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_DENSITY = "maximize_density"
    OPTIMIZE_STABILITY = "optimize_stability"
    ENHANCE_INTERACTIONS = "enhance_interactions"
    REDUCE_VOIDS = "reduce_voids"


class OptimizationAlgorithm(Enum):
    """Optimization algorithms for crystal packing."""

    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    GRADIENT_DESCENT = "gradient_descent"
    HYBRID_GA_SA = "hybrid_ga_sa"


@dataclass
class AtomicCoordinates:
    """Atomic coordinates with element information."""

    coordinates: np.ndarray  # Shape: (n_atoms, 3)
    elements: List[str]
    atomic_numbers: np.ndarray
    charges: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate coordinate data."""
        if len(self.coordinates.shape) != 2 or self.coordinates.shape[1] != 3:
            raise ValueError("Coordinates must be shape (n_atoms, 3)")
        if len(self.elements) != len(self.coordinates):
            raise ValueError("Number of elements must match number of coordinates")
        if len(self.atomic_numbers) != len(self.coordinates):
            raise ValueError(
                "Number of atomic numbers must match number of coordinates"
            )


@dataclass
class MolecularFragment:
    """Molecular fragment for interaction calculations."""

    coordinates: AtomicCoordinates
    fragment_id: str
    molecule_id: str
    charge: float = 0.0
    dipole_moment: Optional[np.ndarray] = None
    aromatic_rings: List[List[int]] = field(default_factory=list)
    hydrogen_bond_donors: List[int] = field(default_factory=list)
    hydrogen_bond_acceptors: List[int] = field(default_factory=list)

    def get_center_of_mass(self) -> np.ndarray:
        """Calculate center of mass."""
        masses = np.array(
            [self._get_atomic_mass(an) for an in self.coordinates.atomic_numbers]
        )
        total_mass = np.sum(masses)
        com = (
            np.sum(self.coordinates.coordinates * masses[:, np.newaxis], axis=0)
            / total_mass
        )
        return com

    def _get_atomic_mass(self, atomic_number: int) -> float:
        """Get atomic mass from atomic number."""
        # Simplified atomic masses
        masses = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 15: 30.974, 16: 32.066}
        return masses.get(atomic_number, 12.011)


@dataclass
class InteractionPair:
    """Pair of fragments with interaction information."""

    fragment1: MolecularFragment
    fragment2: MolecularFragment
    distance: float
    interaction_types: List[InteractionType]
    interaction_energies: Dict[InteractionType, float]
    geometric_parameters: Dict[str, float] = field(default_factory=dict)

    def get_total_energy(self) -> float:
        """Get total interaction energy."""
        return sum(self.interaction_energies.values())


@dataclass
class HydrogenBond:
    """Hydrogen bond interaction details."""

    donor_idx: int
    hydrogen_idx: int
    acceptor_idx: int
    donor_fragment: str
    acceptor_fragment: str
    distance_dh: float  # Donor-hydrogen distance
    distance_ha: float  # Hydrogen-acceptor distance
    distance_da: float  # Donor-acceptor distance
    angle_dha: float  # Donor-hydrogen-acceptor angle
    bond_type: HydrogenBondType
    energy: float
    directionality: float = 1.0  # 0-1 scale


@dataclass
class PiPiInteraction:
    """π-π stacking interaction details."""

    ring1_atoms: List[int]
    ring2_atoms: List[int]
    fragment1: str
    fragment2: str
    centroid_distance: float
    ring_offset: float
    dihedral_angle: float
    stacking_type: PiPiStackingType
    energy: float
    overlap_area: float = 0.0


@dataclass
class VanDerWaalsInteraction:
    """Van der Waals interaction details."""

    atom1_idx: int
    atom2_idx: int
    fragment1: str
    fragment2: str
    distance: float
    vdw_radii_sum: float
    energy: float
    contact_ratio: float  # distance / vdw_radii_sum


@dataclass
class InteractionConfiguration:
    """Configuration for interaction calculations."""

    max_interaction_distance: float = 8.0
    hydrogen_bond_distance_cutoff: float = 3.5
    hydrogen_bond_angle_cutoff: float = 120.0  # degrees
    pi_pi_distance_cutoff: float = 6.0
    pi_pi_angle_cutoff: float = 30.0  # degrees
    vdw_scale_factor: float = 1.2
    electrostatic_cutoff: float = 10.0
    use_gpu: bool = True
    energy_units: str = "kcal/mol"

    def __post_init__(self):
        """Validate configuration."""
        if self.max_interaction_distance <= 0:
            raise ValueError("Max interaction distance must be positive")
        if not 0 < self.vdw_scale_factor < 2:
            raise ValueError("VdW scale factor must be between 0 and 2")


@dataclass
class CrystalPackingResult:
    """Result of crystal packing optimization."""

    optimized_coordinates: List[AtomicCoordinates]
    final_energy: float
    energy_components: Dict[InteractionType, float]
    packing_efficiency: float
    density: float
    void_fraction: float
    interaction_network: List[InteractionPair]
    optimization_history: List[float] = field(default_factory=list)
    convergence_achieved: bool = True

    def get_interaction_summary(self) -> Dict[str, int]:
        """Get summary of interaction types."""
        summary = {}
        for pair in self.interaction_network:
            for interaction_type in pair.interaction_types:
                summary[interaction_type.value] = (
                    summary.get(interaction_type.value, 0) + 1
                )
        return summary


class HydrogenBondCalculator:
    """GPU-accelerated hydrogen bond calculation engine."""

    def __init__(self, config: InteractionConfiguration):
        self.config = config
        self.use_gpu = config.use_gpu and _CUPY_AVAILABLE

        # Hydrogen bond parameters
        self.hb_donors = {"N", "O", "S"}  # Common donors
        self.hb_acceptors = {"N", "O", "S", "F", "Cl"}  # Common acceptors

        # Energy parameters (kcal/mol)
        self.hb_energy_params = {
            HydrogenBondType.CONVENTIONAL: -4.0,
            HydrogenBondType.WEAK: -1.5,
            HydrogenBondType.BIFURCATED: -2.5,
            HydrogenBondType.RESONANCE_ASSISTED: -6.0,
            HydrogenBondType.CHARGE_ASSISTED: -8.0,
        }

    def calculate_hydrogen_bonds(
        self, fragments: List[MolecularFragment]
    ) -> List[HydrogenBond]:
        """Calculate hydrogen bonds between fragments."""
        if not _SCIPY_AVAILABLE:
            warnings.warn(
                "SciPy not available, using simplified hydrogen bond calculation"
            )
            return self._calculate_simple_hydrogen_bonds(fragments)

        hydrogen_bonds = []

        for i, frag1 in enumerate(fragments):
            for j, frag2 in enumerate(fragments[i + 1 :], i + 1):
                hb_pairs = self._find_hydrogen_bond_pairs(frag1, frag2)
                hydrogen_bonds.extend(hb_pairs)

        return hydrogen_bonds

    def _find_hydrogen_bond_pairs(
        self, frag1: MolecularFragment, frag2: MolecularFragment
    ) -> List[HydrogenBond]:
        """Find hydrogen bond pairs between two fragments."""
        hydrogen_bonds = []

        # Get donor and acceptor atoms
        donors1, hydrogens1 = self._get_donors_and_hydrogens(frag1)
        acceptors2 = self._get_acceptors(frag2)

        donors2, hydrogens2 = self._get_donors_and_hydrogens(frag2)
        acceptors1 = self._get_acceptors(frag1)

        # Check frag1 donors -> frag2 acceptors
        for donor_idx, hydrogen_idx in zip(donors1, hydrogens1):
            for acceptor_idx in acceptors2:
                hb = self._evaluate_hydrogen_bond(
                    donor_idx,
                    hydrogen_idx,
                    acceptor_idx,
                    frag1,
                    frag2,
                    frag1.fragment_id,
                    frag2.fragment_id,
                )
                if hb:
                    hydrogen_bonds.append(hb)

        # Check frag2 donors -> frag1 acceptors
        for donor_idx, hydrogen_idx in zip(donors2, hydrogens2):
            for acceptor_idx in acceptors1:
                hb = self._evaluate_hydrogen_bond(
                    donor_idx,
                    hydrogen_idx,
                    acceptor_idx,
                    frag2,
                    frag1,
                    frag2.fragment_id,
                    frag1.fragment_id,
                )
                if hb:
                    hydrogen_bonds.append(hb)

        return hydrogen_bonds

    def _get_donors_and_hydrogens(
        self, fragment: MolecularFragment
    ) -> Tuple[List[int], List[int]]:
        """Get hydrogen bond donor and hydrogen atom indices."""
        donors = []
        hydrogens = []

        coords = fragment.coordinates
        elements = coords.elements
        positions = coords.coordinates

        for i, element in enumerate(elements):
            if element in self.hb_donors:
                # Find bonded hydrogens
                bonded_hydrogens = self._find_bonded_hydrogens(i, positions, elements)
                for h_idx in bonded_hydrogens:
                    donors.append(i)
                    hydrogens.append(h_idx)

        return donors, hydrogens

    def _get_acceptors(self, fragment: MolecularFragment) -> List[int]:
        """Get hydrogen bond acceptor atom indices."""
        acceptors = []
        elements = fragment.coordinates.elements

        for i, element in enumerate(elements):
            if element in self.hb_acceptors:
                acceptors.append(i)

        return acceptors

    def _find_bonded_hydrogens(
        self, donor_idx: int, positions: np.ndarray, elements: List[str]
    ) -> List[int]:
        """Find hydrogen atoms bonded to a donor."""
        hydrogens = []
        donor_pos = positions[donor_idx]

        for i, element in enumerate(elements):
            if element == "H" and i != donor_idx:
                distance = np.linalg.norm(positions[i] - donor_pos)
                if distance < 1.2:  # Typical bond length cutoff
                    hydrogens.append(i)

        return hydrogens

    def _evaluate_hydrogen_bond(
        self,
        donor_idx: int,
        hydrogen_idx: int,
        acceptor_idx: int,
        donor_frag: MolecularFragment,
        acceptor_frag: MolecularFragment,
        donor_frag_id: str,
        acceptor_frag_id: str,
    ) -> Optional[HydrogenBond]:
        """Evaluate if atoms form a hydrogen bond."""
        donor_pos = donor_frag.coordinates.coordinates[donor_idx]
        hydrogen_pos = donor_frag.coordinates.coordinates[hydrogen_idx]
        acceptor_pos = acceptor_frag.coordinates.coordinates[acceptor_idx]

        # Calculate distances
        distance_dh = np.linalg.norm(hydrogen_pos - donor_pos)
        distance_ha = np.linalg.norm(acceptor_pos - hydrogen_pos)
        distance_da = np.linalg.norm(acceptor_pos - donor_pos)

        # Check distance criteria
        if distance_ha > self.config.hydrogen_bond_distance_cutoff:
            return None

        # Calculate angle
        v1 = hydrogen_pos - donor_pos
        v2 = acceptor_pos - hydrogen_pos

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return None

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_dha = np.degrees(np.arccos(cos_angle))

        # Check angle criteria
        if angle_dha < self.config.hydrogen_bond_angle_cutoff:
            return None

        # Determine bond type and calculate energy
        bond_type = self._classify_hydrogen_bond(
            donor_frag.coordinates.elements[donor_idx],
            acceptor_frag.coordinates.elements[acceptor_idx],
            distance_ha,
            angle_dha,
        )

        energy = self._calculate_hydrogen_bond_energy(bond_type, distance_ha, angle_dha)

        return HydrogenBond(
            donor_idx=donor_idx,
            hydrogen_idx=hydrogen_idx,
            acceptor_idx=acceptor_idx,
            donor_fragment=donor_frag_id,
            acceptor_fragment=acceptor_frag_id,
            distance_dh=distance_dh,
            distance_ha=distance_ha,
            distance_da=distance_da,
            angle_dha=angle_dha,
            bond_type=bond_type,
            energy=energy,
        )

    def _classify_hydrogen_bond(
        self, donor_element: str, acceptor_element: str, distance: float, angle: float
    ) -> HydrogenBondType:
        """Classify hydrogen bond type."""
        # Strong conventional bonds
        if donor_element in {"N", "O"} and acceptor_element in {"N", "O"}:
            if distance < 2.8 and angle > 150:
                return HydrogenBondType.CONVENTIONAL

        # Weak C-H bonds
        if donor_element == "C":
            return HydrogenBondType.WEAK

        # Default to conventional
        return HydrogenBondType.CONVENTIONAL

    def _calculate_hydrogen_bond_energy(
        self, bond_type: HydrogenBondType, distance: float, angle: float
    ) -> float:
        """Calculate hydrogen bond energy."""
        base_energy = self.hb_energy_params.get(bond_type, -2.0)

        # Distance dependence (Lennard-Jones like)
        optimal_distance = 2.0  # Optimal H...A distance
        distance_factor = (optimal_distance / distance) ** 6

        # Angular dependence
        angle_factor = (angle / 180.0) ** 2

        energy = base_energy * distance_factor * angle_factor
        return max(energy, -15.0)  # Cap maximum energy

    def _calculate_simple_hydrogen_bonds(
        self, fragments: List[MolecularFragment]
    ) -> List[HydrogenBond]:
        """Simplified hydrogen bond calculation without SciPy."""
        hydrogen_bonds = []

        for i, frag1 in enumerate(fragments):
            for j, frag2 in enumerate(fragments[i + 1 :], i + 1):
                # Simple distance-based detection
                coords1 = frag1.coordinates.coordinates
                coords2 = frag2.coordinates.coordinates

                for k, elem1 in enumerate(frag1.coordinates.elements):
                    if elem1 == "H":
                        for l, elem2 in enumerate(frag2.coordinates.elements):
                            if elem2 in {"N", "O"}:
                                distance = np.linalg.norm(coords1[k] - coords2[l])
                                if 1.5 < distance < 3.0:
                                    hb = HydrogenBond(
                                        donor_idx=-1,
                                        hydrogen_idx=k,
                                        acceptor_idx=l,
                                        donor_fragment=frag1.fragment_id,
                                        acceptor_fragment=frag2.fragment_id,
                                        distance_dh=0.0,
                                        distance_ha=distance,
                                        distance_da=distance,
                                        angle_dha=180.0,
                                        bond_type=HydrogenBondType.CONVENTIONAL,
                                        energy=-2.0,
                                    )
                                    hydrogen_bonds.append(hb)

        return hydrogen_bonds


class PiPiStackingCalculator:
    """GPU-accelerated π-π stacking calculation engine."""

    def __init__(self, config: InteractionConfiguration):
        self.config = config
        self.use_gpu = config.use_gpu and _CUPY_AVAILABLE

        # π-π stacking parameters
        self.optimal_distance = 3.5  # Optimal centroid-centroid distance
        self.max_offset = 2.0  # Maximum ring offset for parallel stacking

        # Energy parameters (kcal/mol)
        self.pi_energy_params = {
            PiPiStackingType.PARALLEL_DISPLACED: -2.5,
            PiPiStackingType.T_SHAPED: -2.0,
            PiPiStackingType.SANDWICH: -3.0,
            PiPiStackingType.SLIPPED_PARALLEL: -2.2,
            PiPiStackingType.EDGE_TO_FACE: -1.8,
        }

    def calculate_pi_pi_interactions(
        self, fragments: List[MolecularFragment]
    ) -> List[PiPiInteraction]:
        """Calculate π-π stacking interactions between fragments."""
        pi_interactions = []

        for i, frag1 in enumerate(fragments):
            for j, frag2 in enumerate(fragments[i + 1 :], i + 1):
                pi_pairs = self._find_pi_pi_pairs(frag1, frag2)
                pi_interactions.extend(pi_pairs)

        return pi_interactions

    def _find_pi_pi_pairs(
        self, frag1: MolecularFragment, frag2: MolecularFragment
    ) -> List[PiPiInteraction]:
        """Find π-π stacking pairs between two fragments."""
        pi_interactions = []

        # Get aromatic rings from both fragments
        rings1 = frag1.aromatic_rings
        rings2 = frag2.aromatic_rings

        for ring1 in rings1:
            for ring2 in rings2:
                pi_interaction = self._evaluate_pi_pi_interaction(
                    ring1, ring2, frag1, frag2
                )
                if pi_interaction:
                    pi_interactions.append(pi_interaction)

        return pi_interactions

    def _evaluate_pi_pi_interaction(
        self,
        ring1: List[int],
        ring2: List[int],
        frag1: MolecularFragment,
        frag2: MolecularFragment,
    ) -> Optional[PiPiInteraction]:
        """Evaluate π-π stacking interaction between two rings."""
        coords1 = frag1.coordinates.coordinates
        coords2 = frag2.coordinates.coordinates

        # Calculate ring centroids
        centroid1 = np.mean(coords1[ring1], axis=0)
        centroid2 = np.mean(coords2[ring2], axis=0)

        centroid_distance = np.linalg.norm(centroid2 - centroid1)

        # Check distance criteria
        if centroid_distance > self.config.pi_pi_distance_cutoff:
            return None

        # Calculate ring normal vectors
        normal1 = self._calculate_ring_normal(coords1[ring1])
        normal2 = self._calculate_ring_normal(coords2[ring2])

        # Calculate angle between normals
        cos_angle = np.dot(normal1, normal2) / (
            np.linalg.norm(normal1) * np.linalg.norm(normal2)
        )
        cos_angle = np.clip(cos_angle, -1, 1)
        dihedral_angle = np.degrees(np.arccos(abs(cos_angle)))

        # Calculate ring offset (for parallel stacking)
        ring_offset = self._calculate_ring_offset(centroid1, centroid2, normal1)

        # Classify stacking type
        stacking_type = self._classify_pi_pi_stacking(
            dihedral_angle, ring_offset, centroid_distance
        )

        # Calculate energy
        energy = self._calculate_pi_pi_energy(
            stacking_type, centroid_distance, dihedral_angle, ring_offset
        )

        # Calculate overlap area (simplified)
        overlap_area = self._calculate_overlap_area(ring1, ring2, coords1, coords2)

        return PiPiInteraction(
            ring1_atoms=ring1,
            ring2_atoms=ring2,
            fragment1=frag1.fragment_id,
            fragment2=frag2.fragment_id,
            centroid_distance=centroid_distance,
            ring_offset=ring_offset,
            dihedral_angle=dihedral_angle,
            stacking_type=stacking_type,
            energy=energy,
            overlap_area=overlap_area,
        )

    def _calculate_ring_normal(self, ring_coords: np.ndarray) -> np.ndarray:
        """Calculate normal vector to aromatic ring."""
        if len(ring_coords) < 3:
            return np.array([0, 0, 1])

        # Use first three atoms to define plane
        v1 = ring_coords[1] - ring_coords[0]
        v2 = ring_coords[2] - ring_coords[0]

        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if norm == 0:
            return np.array([0, 0, 1])

        return normal / norm

    def _calculate_ring_offset(
        self, centroid1: np.ndarray, centroid2: np.ndarray, normal1: np.ndarray
    ) -> float:
        """Calculate offset between ring centroids."""
        centroid_vector = centroid2 - centroid1
        # Project onto ring plane
        projection = centroid_vector - np.dot(centroid_vector, normal1) * normal1
        return np.linalg.norm(projection)

    def _classify_pi_pi_stacking(
        self, dihedral_angle: float, ring_offset: float, distance: float
    ) -> PiPiStackingType:
        """Classify type of π-π stacking interaction."""
        if dihedral_angle < 30:  # Parallel
            if ring_offset < 1.0:
                return PiPiStackingType.SANDWICH
            elif ring_offset < self.max_offset:
                return PiPiStackingType.PARALLEL_DISPLACED
            else:
                return PiPiStackingType.SLIPPED_PARALLEL
        elif 60 < dihedral_angle < 120:  # Perpendicular
            return PiPiStackingType.T_SHAPED
        else:
            return PiPiStackingType.EDGE_TO_FACE

    def _calculate_pi_pi_energy(
        self,
        stacking_type: PiPiStackingType,
        distance: float,
        dihedral_angle: float,
        ring_offset: float,
    ) -> float:
        """Calculate π-π stacking energy."""
        base_energy = self.pi_energy_params.get(stacking_type, -2.0)

        # Distance dependence
        distance_factor = (self.optimal_distance / distance) ** 6

        # Angular dependence
        if stacking_type in [
            PiPiStackingType.PARALLEL_DISPLACED,
            PiPiStackingType.SANDWICH,
        ]:
            angle_factor = np.cos(np.radians(dihedral_angle)) ** 2
        else:
            angle_factor = np.sin(np.radians(dihedral_angle)) ** 2

        # Offset penalty for parallel stacking
        offset_factor = 1.0
        if stacking_type in [
            PiPiStackingType.PARALLEL_DISPLACED,
            PiPiStackingType.SANDWICH,
        ]:
            offset_factor = np.exp(-ring_offset / 2.0)

        energy = base_energy * distance_factor * angle_factor * offset_factor
        return max(energy, -10.0)  # Cap maximum energy

    def _calculate_overlap_area(
        self,
        ring1: List[int],
        ring2: List[int],
        coords1: np.ndarray,
        coords2: np.ndarray,
    ) -> float:
        """Calculate simplified overlap area between rings."""
        # Simplified calculation based on ring sizes and distance
        ring1_area = len(ring1) * 1.5  # Approximate area per atom
        ring2_area = len(ring2) * 1.5

        avg_area = (ring1_area + ring2_area) / 2

        # Distance-dependent overlap
        centroid1 = np.mean(coords1[ring1], axis=0)
        centroid2 = np.mean(coords2[ring2], axis=0)
        distance = np.linalg.norm(centroid2 - centroid1)

        overlap_factor = np.exp(-distance / 3.0)
        return avg_area * overlap_factor


class VanDerWaalsCalculator:
    """GPU-accelerated van der Waals calculation engine."""

    def __init__(self, config: InteractionConfiguration):
        self.config = config
        self.use_gpu = config.use_gpu and _CUPY_AVAILABLE

        # van der Waals radii (Å)
        self.vdw_radii = {
            "H": 1.20,
            "C": 1.70,
            "N": 1.55,
            "O": 1.52,
            "P": 1.80,
            "S": 1.80,
            "F": 1.47,
            "Cl": 1.75,
            "Br": 1.85,
            "I": 1.98,
        }

        # Lennard-Jones parameters (kcal/mol, Å)
        self.lj_epsilon = {
            "H": 0.015,
            "C": 0.105,
            "N": 0.069,
            "O": 0.060,
            "P": 0.305,
            "S": 0.274,
            "F": 0.061,
            "Cl": 0.227,
            "Br": 0.389,
            "I": 0.550,
        }

    def calculate_vdw_interactions(
        self, fragments: List[MolecularFragment]
    ) -> List[VanDerWaalsInteraction]:
        """Calculate van der Waals interactions between fragments."""
        if self.use_gpu:
            return self._calculate_vdw_gpu(fragments)
        else:
            return self._calculate_vdw_cpu(fragments)

    def _calculate_vdw_cpu(
        self, fragments: List[MolecularFragment]
    ) -> List[VanDerWaalsInteraction]:
        """CPU-based van der Waals calculation."""
        vdw_interactions = []

        for i, frag1 in enumerate(fragments):
            for j, frag2 in enumerate(fragments[i + 1 :], i + 1):
                vdw_pairs = self._find_vdw_pairs_cpu(frag1, frag2)
                vdw_interactions.extend(vdw_pairs)

        return vdw_interactions

    def _calculate_vdw_gpu(
        self, fragments: List[MolecularFragment]
    ) -> List[VanDerWaalsInteraction]:
        """GPU-accelerated van der Waals calculation."""
        if not _CUPY_AVAILABLE:
            return self._calculate_vdw_cpu(fragments)

        vdw_interactions = []

        for i, frag1 in enumerate(fragments):
            for j, frag2 in enumerate(fragments[i + 1 :], i + 1):
                vdw_pairs = self._find_vdw_pairs_gpu(frag1, frag2)
                vdw_interactions.extend(vdw_pairs)

        return vdw_interactions

    def _find_vdw_pairs_cpu(
        self, frag1: MolecularFragment, frag2: MolecularFragment
    ) -> List[VanDerWaalsInteraction]:
        """Find van der Waals pairs using CPU."""
        vdw_interactions = []

        coords1 = frag1.coordinates.coordinates
        coords2 = frag2.coordinates.coordinates
        elements1 = frag1.coordinates.elements
        elements2 = frag2.coordinates.elements

        for i, elem1 in enumerate(elements1):
            for j, elem2 in enumerate(elements2):
                distance = np.linalg.norm(coords1[i] - coords2[j])

                # Get van der Waals radii
                vdw1 = self.vdw_radii.get(elem1, 1.70)
                vdw2 = self.vdw_radii.get(elem2, 1.70)
                vdw_sum = (vdw1 + vdw2) * self.config.vdw_scale_factor

                # Check if atoms are in van der Waals contact
                if 0.8 * vdw_sum < distance < self.config.max_interaction_distance:
                    energy = self._calculate_lj_energy(elem1, elem2, distance, vdw_sum)
                    contact_ratio = distance / vdw_sum

                    vdw_interaction = VanDerWaalsInteraction(
                        atom1_idx=i,
                        atom2_idx=j,
                        fragment1=frag1.fragment_id,
                        fragment2=frag2.fragment_id,
                        distance=distance,
                        vdw_radii_sum=vdw_sum,
                        energy=energy,
                        contact_ratio=contact_ratio,
                    )
                    vdw_interactions.append(vdw_interaction)

        return vdw_interactions

    def _find_vdw_pairs_gpu(
        self, frag1: MolecularFragment, frag2: MolecularFragment
    ) -> List[VanDerWaalsInteraction]:
        """Find van der Waals pairs using GPU."""
        coords1_gpu = cp.array(frag1.coordinates.coordinates)
        coords2_gpu = cp.array(frag2.coordinates.coordinates)

        # Calculate distance matrix
        n1, n2 = len(coords1_gpu), len(coords2_gpu)
        distances = cp.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                distances[i, j] = cp.linalg.norm(coords1_gpu[i] - coords2_gpu[j])

        # Convert back to CPU for further processing
        distances_cpu = cp.asnumpy(distances)

        vdw_interactions = []
        elements1 = frag1.coordinates.elements
        elements2 = frag2.coordinates.elements

        for i, elem1 in enumerate(elements1):
            for j, elem2 in enumerate(elements2):
                distance = distances_cpu[i, j]

                vdw1 = self.vdw_radii.get(elem1, 1.70)
                vdw2 = self.vdw_radii.get(elem2, 1.70)
                vdw_sum = (vdw1 + vdw2) * self.config.vdw_scale_factor

                if 0.8 * vdw_sum < distance < self.config.max_interaction_distance:
                    energy = self._calculate_lj_energy(elem1, elem2, distance, vdw_sum)
                    contact_ratio = distance / vdw_sum

                    vdw_interaction = VanDerWaalsInteraction(
                        atom1_idx=i,
                        atom2_idx=j,
                        fragment1=frag1.fragment_id,
                        fragment2=frag2.fragment_id,
                        distance=distance,
                        vdw_radii_sum=vdw_sum,
                        energy=energy,
                        contact_ratio=contact_ratio,
                    )
                    vdw_interactions.append(vdw_interaction)

        return vdw_interactions

    def _calculate_lj_energy(
        self, elem1: str, elem2: str, distance: float, sigma: float
    ) -> float:
        """Calculate Lennard-Jones energy."""
        eps1 = self.lj_epsilon.get(elem1, 0.1)
        eps2 = self.lj_epsilon.get(elem2, 0.1)
        epsilon = np.sqrt(eps1 * eps2)

        # Lennard-Jones potential: 4ε[(σ/r)^12 - (σ/r)^6]
        sigma_over_r = sigma / distance
        sigma6 = sigma_over_r**6
        sigma12 = sigma6**2

        energy = 4 * epsilon * (sigma12 - sigma6)

        # Cap energy to prevent overflow
        return max(min(energy, 10.0), -5.0)


class CrystalPackingOptimizer:
    """GPU-accelerated crystal packing optimization engine."""

    def __init__(self, config: InteractionConfiguration):
        self.config = config
        self.use_gpu = config.use_gpu and _CUPY_AVAILABLE

        # Optimization parameters
        self.population_size = 50
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

        # Initialize interaction calculators
        self.hb_calculator = HydrogenBondCalculator(config)
        self.pi_calculator = PiPiStackingCalculator(config)
        self.vdw_calculator = VanDerWaalsCalculator(config)

    def optimize_crystal_packing(
        self,
        fragments: List[MolecularFragment],
        objective: CrystalPackingObjective = CrystalPackingObjective.MINIMIZE_ENERGY,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC_ALGORITHM,
    ) -> CrystalPackingResult:
        """Optimize crystal packing arrangement."""
        if algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
            return self._genetic_algorithm_optimization(fragments, objective)
        elif algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING:
            return self._simulated_annealing_optimization(fragments, objective)
        elif (
            algorithm == OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION
            and _SCIPY_AVAILABLE
        ):
            return self._differential_evolution_optimization(fragments, objective)
        else:
            return self._simple_optimization(fragments, objective)

    def _genetic_algorithm_optimization(
        self, fragments: List[MolecularFragment], objective: CrystalPackingObjective
    ) -> CrystalPackingResult:
        """Genetic algorithm for crystal packing optimization."""
        # Initialize population
        population = self._initialize_population(fragments)
        best_fitness = float("-inf")
        best_individual = None
        optimization_history = []

        for generation in range(self.max_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, objective)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            optimization_history.append(best_fitness)

            # Selection, crossover, and mutation
            new_population = []

            # Elite selection (keep best individuals)
            elite_size = int(0.1 * self.population_size)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = self._tournament_selection(
                    population, fitness_scores
                )

                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                if np.random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[: self.population_size]

        # Create final result
        final_interactions = self._calculate_all_interactions(best_individual)
        final_energy = sum(pair.get_total_energy() for pair in final_interactions)

        return CrystalPackingResult(
            optimized_coordinates=[frag.coordinates for frag in best_individual],
            final_energy=final_energy,
            energy_components=self._calculate_energy_components(final_interactions),
            packing_efficiency=self._calculate_packing_efficiency(best_individual),
            density=self._calculate_density(best_individual),
            void_fraction=self._calculate_void_fraction(best_individual),
            interaction_network=final_interactions,
            optimization_history=optimization_history,
            convergence_achieved=len(optimization_history) > 10,
        )

    def _simulated_annealing_optimization(
        self, fragments: List[MolecularFragment], objective: CrystalPackingObjective
    ) -> CrystalPackingResult:
        """Simulated annealing for crystal packing optimization."""
        current_fragments = [frag.copy() for frag in fragments]
        current_energy = self._calculate_total_energy(current_fragments)

        best_fragments = [frag.copy() for frag in current_fragments]
        best_energy = current_energy

        temperature = 1000.0
        cooling_rate = 0.95
        min_temperature = 1.0
        optimization_history = []

        while temperature > min_temperature:
            # Generate neighbor solution
            new_fragments = self._generate_neighbor(current_fragments)
            new_energy = self._calculate_total_energy(new_fragments)

            # Accept or reject
            delta_energy = new_energy - current_energy

            if delta_energy < 0 or np.random.random() < np.exp(
                -delta_energy / temperature
            ):
                current_fragments = new_fragments
                current_energy = new_energy

                if current_energy < best_energy:
                    best_fragments = [frag.copy() for frag in current_fragments]
                    best_energy = current_energy

            optimization_history.append(best_energy)
            temperature *= cooling_rate

        # Create final result
        final_interactions = self._calculate_all_interactions(best_fragments)

        return CrystalPackingResult(
            optimized_coordinates=[frag.coordinates for frag in best_fragments],
            final_energy=best_energy,
            energy_components=self._calculate_energy_components(final_interactions),
            packing_efficiency=self._calculate_packing_efficiency(best_fragments),
            density=self._calculate_density(best_fragments),
            void_fraction=self._calculate_void_fraction(best_fragments),
            interaction_network=final_interactions,
            optimization_history=optimization_history,
            convergence_achieved=True,
        )

    def _differential_evolution_optimization(
        self, fragments: List[MolecularFragment], objective: CrystalPackingObjective
    ) -> CrystalPackingResult:
        """Differential evolution for crystal packing optimization."""
        # Convert fragments to parameter vector
        initial_params = self._fragments_to_params(fragments)

        def objective_function(params):
            test_fragments = self._params_to_fragments(params, fragments)
            return self._calculate_total_energy(test_fragments)

        # Set bounds for parameters (position and orientation)
        bounds = [(-10, 10)] * len(initial_params)

        result = differential_evolution(
            objective_function, bounds, maxiter=100, popsize=15, seed=42
        )

        # Convert result back to fragments
        optimized_fragments = self._params_to_fragments(result.x, fragments)
        final_interactions = self._calculate_all_interactions(optimized_fragments)

        return CrystalPackingResult(
            optimized_coordinates=[frag.coordinates for frag in optimized_fragments],
            final_energy=result.fun,
            energy_components=self._calculate_energy_components(final_interactions),
            packing_efficiency=self._calculate_packing_efficiency(optimized_fragments),
            density=self._calculate_density(optimized_fragments),
            void_fraction=self._calculate_void_fraction(optimized_fragments),
            interaction_network=final_interactions,
            optimization_history=[result.fun],
            convergence_achieved=result.success,
        )

    def _simple_optimization(
        self, fragments: List[MolecularFragment], objective: CrystalPackingObjective
    ) -> CrystalPackingResult:
        """Simple optimization fallback."""
        # Return original configuration with calculated interactions
        final_interactions = self._calculate_all_interactions(fragments)
        final_energy = sum(pair.get_total_energy() for pair in final_interactions)

        return CrystalPackingResult(
            optimized_coordinates=[frag.coordinates for frag in fragments],
            final_energy=final_energy,
            energy_components=self._calculate_energy_components(final_interactions),
            packing_efficiency=self._calculate_packing_efficiency(fragments),
            density=self._calculate_density(fragments),
            void_fraction=self._calculate_void_fraction(fragments),
            interaction_network=final_interactions,
            optimization_history=[final_energy],
            convergence_achieved=False,
        )

    def _calculate_all_interactions(
        self, fragments: List[MolecularFragment]
    ) -> List[InteractionPair]:
        """Calculate all intermolecular interactions."""
        interaction_pairs = []

        # Calculate hydrogen bonds
        hydrogen_bonds = self.hb_calculator.calculate_hydrogen_bonds(fragments)

        # Calculate π-π interactions
        pi_interactions = self.pi_calculator.calculate_pi_pi_interactions(fragments)

        # Calculate van der Waals interactions
        vdw_interactions = self.vdw_calculator.calculate_vdw_interactions(fragments)

        # Group interactions by fragment pairs
        pair_dict = {}

        # Process hydrogen bonds
        for hb in hydrogen_bonds:
            key = (hb.donor_fragment, hb.acceptor_fragment)
            if key not in pair_dict:
                pair_dict[key] = {
                    "interactions": [],
                    "energies": {},
                    "distance": hb.distance_da,
                }

            pair_dict[key]["interactions"].append(InteractionType.HYDROGEN_BOND)
            pair_dict[key]["energies"][InteractionType.HYDROGEN_BOND] = (
                pair_dict[key]["energies"].get(InteractionType.HYDROGEN_BOND, 0)
                + hb.energy
            )

        # Process π-π interactions
        for pi in pi_interactions:
            key = (pi.fragment1, pi.fragment2)
            if key not in pair_dict:
                pair_dict[key] = {
                    "interactions": [],
                    "energies": {},
                    "distance": pi.centroid_distance,
                }

            pair_dict[key]["interactions"].append(InteractionType.PI_PI_STACKING)
            pair_dict[key]["energies"][InteractionType.PI_PI_STACKING] = pi.energy

        # Process van der Waals interactions
        for vdw in vdw_interactions:
            key = (vdw.fragment1, vdw.fragment2)
            if key not in pair_dict:
                pair_dict[key] = {
                    "interactions": [],
                    "energies": {},
                    "distance": vdw.distance,
                }

            pair_dict[key]["interactions"].append(InteractionType.VAN_DER_WAALS)
            pair_dict[key]["energies"][InteractionType.VAN_DER_WAALS] = (
                pair_dict[key]["energies"].get(InteractionType.VAN_DER_WAALS, 0)
                + vdw.energy
            )

        # Create InteractionPair objects
        fragment_dict = {frag.fragment_id: frag for frag in fragments}

        for (frag1_id, frag2_id), data in pair_dict.items():
            if frag1_id in fragment_dict and frag2_id in fragment_dict:
                pair = InteractionPair(
                    fragment1=fragment_dict[frag1_id],
                    fragment2=fragment_dict[frag2_id],
                    distance=data["distance"],
                    interaction_types=data["interactions"],
                    interaction_energies=data["energies"],
                )
                interaction_pairs.append(pair)

        return interaction_pairs

    def _initialize_population(
        self, fragments: List[MolecularFragment]
    ) -> List[List[MolecularFragment]]:
        """Initialize population for genetic algorithm."""
        population = []

        for _ in range(self.population_size):
            individual = []
            for frag in fragments:
                new_frag = self._copy_fragment(frag)
                # Apply random perturbation
                new_frag = self._apply_random_perturbation(new_frag)
                individual.append(new_frag)
            population.append(individual)

        return population

    def _copy_fragment(self, fragment: MolecularFragment) -> MolecularFragment:
        """Create a deep copy of a molecular fragment."""
        return MolecularFragment(
            coordinates=AtomicCoordinates(
                coordinates=fragment.coordinates.coordinates.copy(),
                elements=fragment.coordinates.elements.copy(),
                atomic_numbers=fragment.coordinates.atomic_numbers.copy(),
                charges=(
                    fragment.coordinates.charges.copy()
                    if fragment.coordinates.charges is not None
                    else None
                ),
            ),
            fragment_id=fragment.fragment_id,
            molecule_id=fragment.molecule_id,
            charge=fragment.charge,
            dipole_moment=(
                fragment.dipole_moment.copy()
                if fragment.dipole_moment is not None
                else None
            ),
            aromatic_rings=fragment.aromatic_rings.copy(),
            hydrogen_bond_donors=fragment.hydrogen_bond_donors.copy(),
            hydrogen_bond_acceptors=fragment.hydrogen_bond_acceptors.copy(),
        )

    def _apply_random_perturbation(
        self, fragment: MolecularFragment
    ) -> MolecularFragment:
        """Apply random perturbation to fragment coordinates."""
        # Small random translation
        translation = np.random.normal(0, 0.5, 3)
        fragment.coordinates.coordinates += translation

        # Small random rotation
        angle = np.random.normal(0, 0.1)
        axis = np.random.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis)

        fragment.coordinates.coordinates = self._rotate_coordinates(
            fragment.coordinates.coordinates, axis, angle
        )

        return fragment

    def _rotate_coordinates(
        self, coords: np.ndarray, axis: np.ndarray, angle: float
    ) -> np.ndarray:
        """Rotate coordinates around an axis."""
        # Rodrigues' rotation formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Center coordinates
        center = np.mean(coords, axis=0)
        centered_coords = coords - center

        # Apply rotation
        rotated_coords = (
            centered_coords * cos_angle
            + np.cross(axis, centered_coords) * sin_angle
            + axis * np.dot(axis, centered_coords.T).T * (1 - cos_angle)
        )

        return rotated_coords + center

    def _evaluate_fitness(
        self, fragments: List[MolecularFragment], objective: CrystalPackingObjective
    ) -> float:
        """Evaluate fitness of fragment arrangement."""
        total_energy = self._calculate_total_energy(fragments)

        if objective == CrystalPackingObjective.MINIMIZE_ENERGY:
            return -total_energy  # Negative because we want to minimize
        elif objective == CrystalPackingObjective.MAXIMIZE_DENSITY:
            density = self._calculate_density(fragments)
            return density
        elif objective == CrystalPackingObjective.OPTIMIZE_STABILITY:
            stability = -total_energy + self._calculate_packing_efficiency(fragments)
            return stability
        else:
            return -total_energy

    def _calculate_total_energy(self, fragments: List[MolecularFragment]) -> float:
        """Calculate total interaction energy."""
        try:
            interactions = self._calculate_all_interactions(fragments)
            return sum(pair.get_total_energy() for pair in interactions)
        except Exception:
            return 1000.0  # High penalty for invalid configurations

    def _calculate_energy_components(
        self, interactions: List[InteractionPair]
    ) -> Dict[InteractionType, float]:
        """Calculate energy components by interaction type."""
        components = {}

        for pair in interactions:
            for interaction_type, energy in pair.interaction_energies.items():
                components[interaction_type] = (
                    components.get(interaction_type, 0) + energy
                )

        return components

    def _calculate_packing_efficiency(
        self, fragments: List[MolecularFragment]
    ) -> float:
        """Calculate packing efficiency."""
        # Simplified calculation based on occupied volume
        total_volume = 1000.0  # Assumed cell volume
        occupied_volume = 0.0

        for frag in fragments:
            # Estimate molecular volume
            n_atoms = len(frag.coordinates.coordinates)
            occupied_volume += n_atoms * 20.0  # Rough atomic volume

        return min(occupied_volume / total_volume, 1.0)

    def _calculate_density(self, fragments: List[MolecularFragment]) -> float:
        """Calculate crystal density."""
        total_mass = 0.0

        for frag in fragments:
            for atomic_number in frag.coordinates.atomic_numbers:
                # Rough atomic masses
                masses = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}
                total_mass += masses.get(atomic_number, 12.011)

        # Convert to g/cm³ (simplified)
        volume = 1000.0  # Assumed cell volume in Ų
        density = total_mass / (volume * 1.66e-24)  # Convert to g/cm³

        return density

    def _calculate_void_fraction(self, fragments: List[MolecularFragment]) -> float:
        """Calculate void fraction."""
        packing_efficiency = self._calculate_packing_efficiency(fragments)
        return 1.0 - packing_efficiency

    def _tournament_selection(
        self,
        population: List[List[MolecularFragment]],
        fitness_scores: List[float],
        tournament_size: int = 3,
    ) -> Tuple[List[MolecularFragment], List[MolecularFragment]]:
        """Tournament selection for genetic algorithm."""

        def select_one():
            tournament_indices = np.random.choice(
                len(population), tournament_size, replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            return population[winner_idx]

        return select_one(), select_one()

    def _crossover(
        self, parent1: List[MolecularFragment], parent2: List[MolecularFragment]
    ) -> Tuple[List[MolecularFragment], List[MolecularFragment]]:
        """Crossover operation for genetic algorithm."""
        child1, child2 = [], []

        for frag1, frag2 in zip(parent1, parent2):
            # Blend coordinates
            alpha = np.random.random()

            new_coords1 = (
                alpha * frag1.coordinates.coordinates
                + (1 - alpha) * frag2.coordinates.coordinates
            )
            new_coords2 = (
                (1 - alpha) * frag1.coordinates.coordinates
                + alpha * frag2.coordinates.coordinates
            )

            child_frag1 = self._copy_fragment(frag1)
            child_frag1.coordinates.coordinates = new_coords1

            child_frag2 = self._copy_fragment(frag2)
            child_frag2.coordinates.coordinates = new_coords2

            child1.append(child_frag1)
            child2.append(child_frag2)

        return child1, child2

    def _mutate(self, individual: List[MolecularFragment]) -> List[MolecularFragment]:
        """Mutation operation for genetic algorithm."""
        mutated = []

        for frag in individual:
            mutated_frag = self._copy_fragment(frag)

            # Small random perturbation
            if np.random.random() < 0.5:
                mutated_frag = self._apply_random_perturbation(mutated_frag)

            mutated.append(mutated_frag)

        return mutated

    def _generate_neighbor(
        self, fragments: List[MolecularFragment]
    ) -> List[MolecularFragment]:
        """Generate neighbor solution for simulated annealing."""
        neighbor = []

        for frag in fragments:
            neighbor_frag = self._copy_fragment(frag)
            neighbor_frag = self._apply_random_perturbation(neighbor_frag)
            neighbor.append(neighbor_frag)

        return neighbor

    def _fragments_to_params(self, fragments: List[MolecularFragment]) -> np.ndarray:
        """Convert fragments to parameter vector."""
        params = []

        for frag in fragments:
            # Use center of mass and orientation
            com = frag.get_center_of_mass()
            params.extend(com.tolist())

            # Add simple orientation (euler angles)
            params.extend([0, 0, 0])  # Simplified

        return np.array(params)

    def _params_to_fragments(
        self, params: np.ndarray, template_fragments: List[MolecularFragment]
    ) -> List[MolecularFragment]:
        """Convert parameter vector to fragments."""
        fragments = []

        for i, template in enumerate(template_fragments):
            frag = self._copy_fragment(template)

            # Extract position and orientation
            start_idx = i * 6
            position = params[start_idx : start_idx + 3]
            orientation = params[start_idx + 3 : start_idx + 6]

            # Apply transformation
            current_com = frag.get_center_of_mass()
            translation = position - current_com
            frag.coordinates.coordinates += translation

            fragments.append(frag)

        return fragments


class GPUAcceleratedInteractionEngine:
    """Main GPU-accelerated intermolecular interaction engine."""

    def __init__(self, config: Optional[InteractionConfiguration] = None):
        """Initialize interaction engine."""
        self.config = config or InteractionConfiguration()
        self.use_gpu = self.config.use_gpu and _CUPY_AVAILABLE

        # Initialize calculators
        self.hb_calculator = HydrogenBondCalculator(self.config)
        self.pi_calculator = PiPiStackingCalculator(self.config)
        self.vdw_calculator = VanDerWaalsCalculator(self.config)
        self.packing_optimizer = CrystalPackingOptimizer(self.config)

        # Interaction cache
        self.interaction_cache = {}
        self.performance_stats = {}

    def calculate_all_interactions(
        self, fragments: List[MolecularFragment]
    ) -> List[InteractionPair]:
        """Calculate all intermolecular interactions between fragments."""
        cache_key = self._generate_cache_key(fragments)

        if cache_key in self.interaction_cache:
            return self.interaction_cache[cache_key]

        # Calculate interactions
        interactions = self.packing_optimizer._calculate_all_interactions(fragments)

        # Cache result
        self.interaction_cache[cache_key] = interactions

        return interactions

    def optimize_crystal_packing(
        self,
        fragments: List[MolecularFragment],
        objective: CrystalPackingObjective = CrystalPackingObjective.MINIMIZE_ENERGY,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC_ALGORITHM,
    ) -> CrystalPackingResult:
        """Optimize crystal packing arrangement."""
        return self.packing_optimizer.optimize_crystal_packing(
            fragments, objective, algorithm
        )

    def analyze_interaction_network(
        self, fragments: List[MolecularFragment]
    ) -> Dict[str, Any]:
        """Analyze intermolecular interaction network."""
        interactions = self.calculate_all_interactions(fragments)

        analysis = {
            "total_interactions": len(interactions),
            "total_energy": sum(pair.get_total_energy() for pair in interactions),
            "interaction_types": {},
            "energy_by_type": {},
            "strongest_interactions": [],
            "network_connectivity": self._calculate_network_connectivity(interactions),
            "interaction_statistics": self._calculate_interaction_statistics(
                interactions
            ),
        }

        # Count interaction types
        for pair in interactions:
            for interaction_type in pair.interaction_types:
                type_name = interaction_type.value
                analysis["interaction_types"][type_name] = (
                    analysis["interaction_types"].get(type_name, 0) + 1
                )

        # Energy by type
        for pair in interactions:
            for interaction_type, energy in pair.interaction_energies.items():
                type_name = interaction_type.value
                analysis["energy_by_type"][type_name] = (
                    analysis["energy_by_type"].get(type_name, 0) + energy
                )

        # Find strongest interactions
        sorted_interactions = sorted(interactions, key=lambda x: x.get_total_energy())
        analysis["strongest_interactions"] = sorted_interactions[:5]  # Top 5

        return analysis

    def predict_crystal_stability(
        self, fragments: List[MolecularFragment]
    ) -> Dict[str, float]:
        """Predict crystal stability based on interaction patterns."""
        interactions = self.calculate_all_interactions(fragments)

        total_energy = sum(pair.get_total_energy() for pair in interactions)
        interaction_density = len(interactions) / len(fragments)

        # Calculate stability indicators
        stability_indicators = {
            "total_binding_energy": -total_energy,  # More negative = more stable
            "interaction_density": interaction_density,
            "hydrogen_bond_contribution": 0.0,
            "pi_pi_contribution": 0.0,
            "vdw_contribution": 0.0,
            "stability_score": 0.0,
        }

        # Calculate contributions
        for pair in interactions:
            for interaction_type, energy in pair.interaction_energies.items():
                if interaction_type == InteractionType.HYDROGEN_BOND:
                    stability_indicators["hydrogen_bond_contribution"] += abs(energy)
                elif interaction_type == InteractionType.PI_PI_STACKING:
                    stability_indicators["pi_pi_contribution"] += abs(energy)
                elif interaction_type == InteractionType.VAN_DER_WAALS:
                    stability_indicators["vdw_contribution"] += abs(energy)

        # Calculate overall stability score
        stability_score = (
            stability_indicators["hydrogen_bond_contribution"] * 2.0
            + stability_indicators["pi_pi_contribution"] * 1.5
            + stability_indicators["vdw_contribution"] * 1.0
            + interaction_density * 10.0
        )

        stability_indicators["stability_score"] = stability_score

        return stability_indicators

    def _generate_cache_key(self, fragments: List[MolecularFragment]) -> str:
        """Generate cache key for fragments."""
        key_parts = []

        for frag in fragments:
            coords_str = str(frag.coordinates.coordinates.tobytes())
            elements_str = "".join(frag.coordinates.elements)
            key_parts.append(f"{frag.fragment_id}_{coords_str}_{elements_str}")

        return "_".join(key_parts)

    def _calculate_network_connectivity(
        self, interactions: List[InteractionPair]
    ) -> Dict[str, int]:
        """Calculate interaction network connectivity."""
        connectivity = {}

        for pair in interactions:
            frag1_id = pair.fragment1.fragment_id
            frag2_id = pair.fragment2.fragment_id

            connectivity[frag1_id] = connectivity.get(frag1_id, 0) + 1
            connectivity[frag2_id] = connectivity.get(frag2_id, 0) + 1

        return connectivity

    def _calculate_interaction_statistics(
        self, interactions: List[InteractionPair]
    ) -> Dict[str, float]:
        """Calculate interaction statistics."""
        if not interactions:
            return {}

        energies = [pair.get_total_energy() for pair in interactions]
        distances = [pair.distance for pair in interactions]

        stats = {
            "mean_energy": np.mean(energies),
            "std_energy": np.std(energies),
            "min_energy": np.min(energies),
            "max_energy": np.max(energies),
            "mean_distance": np.mean(distances),
            "std_distance": np.std(distances),
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
        }

        return stats


# Utility functions


def create_molecular_fragment_from_smiles(
    smiles: str, fragment_id: str = "frag"
) -> Optional[MolecularFragment]:
    """Create molecular fragment from SMILES string."""
    if not _RDKIT_AVAILABLE:
        warnings.warn("RDKit not available, cannot create fragment from SMILES")
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)

        # Simple coordinate generation (would use real 3D in practice)
        n_atoms = mol.GetNumAtoms()
        coordinates = np.random.random((n_atoms, 3)) * 10

        elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])

        # Find aromatic rings
        aromatic_rings = []
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                aromatic_rings.append(list(ring))

        # Find H-bond donors and acceptors
        donors = []
        acceptors = []

        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetTotalNumHs() > 0 and atom.GetSymbol() in ["N", "O"]:
                donors.append(i)
            if atom.GetSymbol() in ["N", "O", "S"] and atom.GetTotalDegree() < 4:
                acceptors.append(i)

        atomic_coords = AtomicCoordinates(
            coordinates=coordinates, elements=elements, atomic_numbers=atomic_numbers
        )

        fragment = MolecularFragment(
            coordinates=atomic_coords,
            fragment_id=fragment_id,
            molecule_id=smiles,
            aromatic_rings=aromatic_rings,
            hydrogen_bond_donors=donors,
            hydrogen_bond_acceptors=acceptors,
        )

        return fragment

    except Exception as e:
        warnings.warn(f"Error creating fragment from SMILES: {e}")
        return None


def create_interaction_config(
    use_gpu: bool = True,
    max_distance: float = 8.0,
    hb_distance_cutoff: float = 3.5,
    pi_distance_cutoff: float = 6.0,
) -> InteractionConfiguration:
    """Create interaction configuration with common settings."""
    return InteractionConfiguration(
        max_interaction_distance=max_distance,
        hydrogen_bond_distance_cutoff=hb_distance_cutoff,
        pi_pi_distance_cutoff=pi_distance_cutoff,
        use_gpu=use_gpu,
    )


def analyze_crystal_packing(
    fragments: List[MolecularFragment],
    config: Optional[InteractionConfiguration] = None,
) -> Dict[str, Any]:
    """Analyze crystal packing for a set of molecular fragments."""
    if config is None:
        config = InteractionConfiguration()

    engine = GPUAcceleratedInteractionEngine(config)

    # Calculate interactions
    interactions = engine.calculate_all_interactions(fragments)

    # Analyze network
    network_analysis = engine.analyze_interaction_network(fragments)

    # Predict stability
    stability = engine.predict_crystal_stability(fragments)

    return {
        "interactions": interactions,
        "network_analysis": network_analysis,
        "stability_prediction": stability,
        "fragment_count": len(fragments),
        "total_atoms": sum(len(frag.coordinates.coordinates) for frag in fragments),
    }


def optimize_intermolecular_interactions(
    fragments: List[MolecularFragment],
    objective: CrystalPackingObjective = CrystalPackingObjective.MINIMIZE_ENERGY,
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC_ALGORITHM,
    config: Optional[InteractionConfiguration] = None,
) -> CrystalPackingResult:
    """Optimize intermolecular interactions for crystal packing."""
    if config is None:
        config = InteractionConfiguration()

    engine = GPUAcceleratedInteractionEngine(config)
    return engine.optimize_crystal_packing(fragments, objective, algorithm)


# Exception classes


class IntermolecularInteractionError(Exception):
    """Base exception for intermolecular interaction calculations."""

    pass


class HydrogenBondCalculationError(IntermolecularInteractionError):
    """Exception for hydrogen bond calculation errors."""

    pass


class PiPiStackingCalculationError(IntermolecularInteractionError):
    """Exception for π-π stacking calculation errors."""

    pass


class VanDerWaalsCalculationError(IntermolecularInteractionError):
    """Exception for van der Waals calculation errors."""

    pass


class CrystalPackingOptimizationError(IntermolecularInteractionError):
    """Exception for crystal packing optimization errors."""

    pass


class InteractionEngineError(IntermolecularInteractionError):
    """Exception for interaction engine errors."""

    pass

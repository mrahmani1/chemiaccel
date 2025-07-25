"""
Thermoset Network and Curing Simulator.

This module provides Monte Carlo simulation for thermoset curing processes,
GPU-accelerated network topology analysis, gel point detection, and property
prediction from network topology descriptors.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning(
        "RDKit not available. Some thermoset analysis features will be limited."
    )

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available. GPU acceleration will be disabled.")

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning(
        "NetworkX not available. Some network analysis features will be limited."
    )

from ..core.exceptions import ChemiAccelError, InvalidMoleculeError
from ..utils.rdkit_integration import parse_smiles_safe


class CuringMechanism(Enum):
    """Types of curing mechanisms for thermosets."""

    EPOXY_AMINE = "epoxy_amine"
    EPOXY_ANHYDRIDE = "epoxy_anhydride"
    POLYURETHANE = "polyurethane"
    PHENOLIC = "phenolic"
    UNSATURATED_POLYESTER = "unsaturated_polyester"
    VINYL_ESTER = "vinyl_ester"
    BISMALEIMIDE = "bismaleimide"
    CYANATE_ESTER = "cyanate_ester"


class NetworkState(Enum):
    """States of the thermoset network during curing."""

    LIQUID = "liquid"
    GEL_POINT = "gel_point"
    RUBBERY = "rubbery"
    GLASSY = "glassy"


@dataclass
class ReactiveGroup:
    """Represents a reactive group in the thermoset system."""

    group_id: int
    group_type: str
    reactivity: float
    steric_factor: float
    position: Tuple[float, float, float]
    is_reacted: bool = False
    connected_to: List[int] = field(default_factory=list)


@dataclass
class CrosslinkBond:
    """Represents a crosslink bond between reactive groups."""

    bond_id: int
    group1_id: int
    group2_id: int
    bond_strength: float
    formation_time: float
    bond_type: str


@dataclass
class NetworkTopology:
    """Represents the topology of the thermoset network."""

    nodes: List[ReactiveGroup]
    bonds: List[CrosslinkBond]
    adjacency_matrix: np.ndarray
    cluster_sizes: List[int]
    largest_cluster_size: int
    gel_fraction: float
    crosslink_density: float
    average_functionality: float


@dataclass
class CuringKinetics:
    """Kinetic parameters for curing simulation."""

    rate_constants: Dict[str, float]
    activation_energies: Dict[str, float]
    temperature: float
    catalyst_concentration: float
    diffusion_coefficients: Dict[str, float]


@dataclass
class CuringResult:
    """Results from thermoset curing simulation."""

    time_points: np.ndarray
    conversion_profile: np.ndarray
    gel_point_time: Optional[float]
    gel_point_conversion: Optional[float]
    final_crosslink_density: float
    final_gel_fraction: float
    network_topology: NetworkTopology
    mechanical_properties: Dict[str, float]
    thermal_properties: Dict[str, float]
    simulation_metadata: Dict[str, Any]


class ThermosetCuringSimulator:
    """Monte Carlo simulator for thermoset curing processes."""

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the thermoset curing simulator.

        Args:
            use_gpu: Whether to use GPU acceleration when available
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        if self.use_gpu:
            logging.info("GPU acceleration enabled for thermoset simulation")
        else:
            logging.info("Using CPU for thermoset simulation")

        # Default kinetic parameters for common systems
        self.default_kinetics = {
            CuringMechanism.EPOXY_AMINE: CuringKinetics(
                rate_constants={"k1": 1e-3, "k2": 5e-4},
                activation_energies={"E1": 50000, "E2": 60000},
                temperature=393.15,  # 120°C
                catalyst_concentration=0.01,
                diffusion_coefficients={"epoxy": 1e-12, "amine": 8e-13},
            ),
            CuringMechanism.EPOXY_ANHYDRIDE: CuringKinetics(
                rate_constants={"k1": 2e-3, "k2": 1e-3},
                activation_energies={"E1": 55000, "E2": 65000},
                temperature=423.15,  # 150°C
                catalyst_concentration=0.005,
                diffusion_coefficients={"epoxy": 1e-12, "anhydride": 6e-13},
            ),
            CuringMechanism.POLYURETHANE: CuringKinetics(
                rate_constants={"k1": 5e-3, "k2": 2e-3},
                activation_energies={"E1": 45000, "E2": 50000},
                temperature=353.15,  # 80°C
                catalyst_concentration=0.02,
                diffusion_coefficients={"isocyanate": 1.2e-12, "polyol": 5e-13},
            ),
        }

    def setup_initial_network(
        self,
        resin_smiles: str,
        hardener_smiles: str,
        stoichiometry: float,
        system_size: int = 1000,
    ) -> List[ReactiveGroup]:
        """
        Set up initial network configuration for simulation.

        Args:
            resin_smiles: SMILES string of the resin component
            hardener_smiles: SMILES string of the hardener component
            stoichiometry: Stoichiometric ratio (hardener/resin)
            system_size: Number of molecules in the simulation box

        Returns:
            List of reactive groups in the initial system
        """
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for network setup")

        resin_mol = parse_smiles_safe(resin_smiles)
        hardener_mol = parse_smiles_safe(hardener_smiles)

        if resin_mol is None:
            raise InvalidMoleculeError(resin_smiles, "Invalid resin SMILES")
        if hardener_mol is None:
            raise InvalidMoleculeError(hardener_smiles, "Invalid hardener SMILES")

        # Identify reactive groups in each component
        resin_groups = self._identify_reactive_groups(resin_mol, "resin")
        hardener_groups = self._identify_reactive_groups(hardener_mol, "hardener")

        # Calculate number of molecules based on stoichiometry
        total_reactive_resin = len(resin_groups)
        total_reactive_hardener = len(hardener_groups)

        if total_reactive_resin == 0 or total_reactive_hardener == 0:
            raise ChemiAccelError("No reactive groups found in resin or hardener")

        # Calculate molecule counts
        n_hardener = int(system_size * stoichiometry / (1 + stoichiometry))
        n_resin = system_size - n_hardener

        # Generate initial system
        reactive_groups = []
        group_id = 0

        # Add resin molecules
        for i in range(n_resin):
            for group_info in resin_groups:
                position = self._generate_random_position()
                group = ReactiveGroup(
                    group_id=group_id,
                    group_type=group_info["type"],
                    reactivity=group_info["reactivity"],
                    steric_factor=group_info["steric_factor"],
                    position=position,
                )
                reactive_groups.append(group)
                group_id += 1

        # Add hardener molecules
        for i in range(n_hardener):
            for group_info in hardener_groups:
                position = self._generate_random_position()
                group = ReactiveGroup(
                    group_id=group_id,
                    group_type=group_info["type"],
                    reactivity=group_info["reactivity"],
                    steric_factor=group_info["steric_factor"],
                    position=position,
                )
                reactive_groups.append(group)
                group_id += 1

        logging.info(
            f"Initial system: {len(reactive_groups)} reactive groups "
            f"({n_resin} resin, {n_hardener} hardener molecules)"
        )

        return reactive_groups

    def simulate_curing(
        self,
        reactive_groups: List[ReactiveGroup],
        mechanism: CuringMechanism,
        simulation_time: float = 3600.0,
        time_step: float = 1.0,
        kinetics: Optional[CuringKinetics] = None,
    ) -> CuringResult:
        """
        Perform Monte Carlo simulation of thermoset curing.

        Args:
            reactive_groups: Initial reactive groups
            mechanism: Curing mechanism
            simulation_time: Total simulation time (seconds)
            time_step: Time step for simulation (seconds)
            kinetics: Kinetic parameters (uses defaults if None)

        Returns:
            CuringResult object with simulation results
        """
        if kinetics is None:
            kinetics = self.default_kinetics.get(mechanism)
            if kinetics is None:
                raise ChemiAccelError(f"No default kinetics available for {mechanism}")

        # Initialize simulation
        time_points = np.arange(0, simulation_time + time_step, time_step)
        conversion_profile = np.zeros(len(time_points))
        bonds = []

        # Track gel point
        gel_point_time = None
        gel_point_conversion = None

        logging.info(
            f"Starting curing simulation: {len(reactive_groups)} groups, "
            f"{simulation_time}s, mechanism: {mechanism.value}"
        )

        start_time = time.time()

        for i, current_time in enumerate(time_points):
            # Perform Monte Carlo step
            new_bonds = self._monte_carlo_step(
                reactive_groups, bonds, kinetics, time_step, current_time
            )
            bonds.extend(new_bonds)

            # Calculate conversion
            total_groups = len(reactive_groups)
            reacted_groups = sum(1 for group in reactive_groups if group.is_reacted)
            conversion = reacted_groups / total_groups if total_groups > 0 else 0.0
            conversion_profile[i] = conversion

            # Check for gel point
            if gel_point_time is None:
                network_topology = self._analyze_network_topology(
                    reactive_groups, bonds
                )
                if self._detect_gel_point(network_topology):
                    gel_point_time = current_time
                    gel_point_conversion = conversion
                    logging.info(
                        f"Gel point detected at t={current_time:.1f}s, "
                        f"conversion={conversion:.3f}"
                    )

            # Progress logging
            if i % max(1, len(time_points) // 10) == 0:
                logging.info(
                    f"Simulation progress: {i/len(time_points)*100:.1f}%, "
                    f"conversion: {conversion:.3f}"
                )

        # Final network analysis
        final_topology = self._analyze_network_topology(reactive_groups, bonds)

        # Predict properties
        mechanical_props = self._predict_mechanical_properties(final_topology, kinetics)
        thermal_props = self._predict_thermal_properties(final_topology, kinetics)

        simulation_time_elapsed = time.time() - start_time

        result = CuringResult(
            time_points=time_points,
            conversion_profile=conversion_profile,
            gel_point_time=gel_point_time,
            gel_point_conversion=gel_point_conversion,
            final_crosslink_density=final_topology.crosslink_density,
            final_gel_fraction=final_topology.gel_fraction,
            network_topology=final_topology,
            mechanical_properties=mechanical_props,
            thermal_properties=thermal_props,
            simulation_metadata={
                "mechanism": mechanism.value,
                "simulation_time_elapsed": simulation_time_elapsed,
                "total_bonds_formed": len(bonds),
                "final_conversion": conversion_profile[-1],
            },
        )

        logging.info(f"Curing simulation completed in {simulation_time_elapsed:.2f}s")
        return result

    def _identify_reactive_groups(
        self, mol: Chem.Mol, component_type: str
    ) -> List[Dict[str, Any]]:
        """Identify reactive groups in a molecule."""
        groups = []

        # Define reactive group patterns
        patterns = {
            "epoxy": "[C]1[O][C]1",
            "primary_amine": "[N]([H])[H]",
            "secondary_amine": "[N]([H])[C]",
            "hydroxyl": "[O][H]",
            "carboxyl": "[C](=O)[O][H]",
            "anhydride": "[C](=O)[O][C](=O)",
            "isocyanate": "[N]=[C]=[O]",
            "vinyl": "[C]=[C]",
        }

        for group_name, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue

            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                # Calculate reactivity based on group type and environment
                reactivity = self._calculate_group_reactivity(mol, match, group_name)
                steric_factor = self._calculate_steric_factor(mol, match)

                groups.append(
                    {
                        "type": group_name,
                        "reactivity": reactivity,
                        "steric_factor": steric_factor,
                        "atom_indices": match,
                    }
                )

        return groups

    def _calculate_group_reactivity(
        self, mol: Chem.Mol, match: Tuple[int, ...], group_type: str
    ) -> float:
        """Calculate reactivity of a functional group."""
        base_reactivities = {
            "epoxy": 0.8,
            "primary_amine": 0.9,
            "secondary_amine": 0.6,
            "hydroxyl": 0.4,
            "carboxyl": 0.5,
            "anhydride": 0.7,
            "isocyanate": 0.95,
            "vinyl": 0.7,
        }

        base_reactivity = base_reactivities.get(group_type, 0.5)

        # Adjust for electronic effects (simplified)
        central_atom_idx = match[0]
        central_atom = mol.GetAtomWithIdx(central_atom_idx)

        # Consider neighboring atoms
        neighbors = central_atom.GetNeighbors()
        electron_withdrawing = sum(
            1 for n in neighbors if n.GetAtomicNum() in [7, 8, 9, 17]
        )
        electron_donating = sum(
            1 for n in neighbors if n.GetAtomicNum() == 6 and n.GetIsAromatic()
        )

        reactivity_modifier = (
            1.0 + 0.1 * electron_withdrawing - 0.05 * electron_donating
        )

        return min(1.0, max(0.1, base_reactivity * reactivity_modifier))

    def _calculate_steric_factor(self, mol: Chem.Mol, match: Tuple[int, ...]) -> float:
        """Calculate steric hindrance factor."""
        central_atom_idx = match[0]
        central_atom = mol.GetAtomWithIdx(central_atom_idx)

        # Count bulky substituents
        neighbors = central_atom.GetNeighbors()
        bulky_groups = 0

        for neighbor in neighbors:
            if neighbor.GetDegree() > 2:  # Branched carbon
                bulky_groups += 1
            elif neighbor.GetAtomicNum() == 6:  # Carbon
                # Check if it's part of a large ring or aromatic system
                if neighbor.IsInRing() and neighbor.GetIsAromatic():
                    bulky_groups += 0.5

        # Steric factor decreases with bulky groups
        steric_factor = max(0.1, 1.0 - 0.2 * bulky_groups)
        return steric_factor

    def _generate_random_position(self) -> Tuple[float, float, float]:
        """Generate random 3D position for a reactive group."""
        # Simple cubic box with side length 100 Å
        return (
            np.random.uniform(0, 100),
            np.random.uniform(0, 100),
            np.random.uniform(0, 100),
        )

    def _monte_carlo_step(
        self,
        reactive_groups: List[ReactiveGroup],
        existing_bonds: List[CrosslinkBond],
        kinetics: CuringKinetics,
        time_step: float,
        current_time: float,
    ) -> List[CrosslinkBond]:
        """Perform one Monte Carlo step in the curing simulation."""
        new_bonds = []

        # Find potential reaction pairs
        unreacted_groups = [g for g in reactive_groups if not g.is_reacted]

        for i, group1 in enumerate(unreacted_groups):
            for group2 in unreacted_groups[i + 1 :]:
                # Check if groups can react
                if self._can_react(group1, group2):
                    # Calculate reaction probability
                    prob = self._calculate_reaction_probability(
                        group1, group2, kinetics, time_step
                    )

                    # Monte Carlo decision
                    if np.random.random() < prob:
                        # Form bond
                        bond = CrosslinkBond(
                            bond_id=len(existing_bonds) + len(new_bonds),
                            group1_id=group1.group_id,
                            group2_id=group2.group_id,
                            bond_strength=self._calculate_bond_strength(group1, group2),
                            formation_time=current_time,
                            bond_type=f"{group1.group_type}-{group2.group_type}",
                        )

                        new_bonds.append(bond)

                        # Update group states
                        group1.is_reacted = True
                        group2.is_reacted = True
                        group1.connected_to.append(group2.group_id)
                        group2.connected_to.append(group1.group_id)

        return new_bonds

    def _can_react(self, group1: ReactiveGroup, group2: ReactiveGroup) -> bool:
        """Check if two groups can react with each other."""
        # Define compatible reaction pairs
        compatible_pairs = {
            ("epoxy", "primary_amine"),
            ("epoxy", "secondary_amine"),
            ("epoxy", "hydroxyl"),
            ("epoxy", "carboxyl"),
            ("anhydride", "primary_amine"),
            ("anhydride", "secondary_amine"),
            ("anhydride", "hydroxyl"),
            ("isocyanate", "primary_amine"),
            ("isocyanate", "secondary_amine"),
            ("isocyanate", "hydroxyl"),
            ("vinyl", "vinyl"),
        }

        pair1 = (group1.group_type, group2.group_type)
        pair2 = (group2.group_type, group1.group_type)

        return pair1 in compatible_pairs or pair2 in compatible_pairs

    def _calculate_reaction_probability(
        self,
        group1: ReactiveGroup,
        group2: ReactiveGroup,
        kinetics: CuringKinetics,
        time_step: float,
    ) -> float:
        """Calculate probability of reaction between two groups."""
        # Distance factor
        distance = np.linalg.norm(np.array(group1.position) - np.array(group2.position))
        distance_factor = np.exp(-distance / 10.0)  # 10 Å characteristic distance

        # Reactivity factor
        reactivity_factor = group1.reactivity * group2.reactivity

        # Steric factor
        steric_factor = group1.steric_factor * group2.steric_factor

        # Temperature-dependent rate constant
        R = 8.314  # Gas constant
        k_base = kinetics.rate_constants.get("k1", 1e-3)
        E_a = kinetics.activation_energies.get("E1", 50000)
        k_eff = k_base * np.exp(-E_a / (R * kinetics.temperature))

        # Overall probability
        prob = k_eff * time_step * distance_factor * reactivity_factor * steric_factor

        return min(1.0, prob)

    def _calculate_bond_strength(
        self, group1: ReactiveGroup, group2: ReactiveGroup
    ) -> float:
        """Calculate strength of the formed bond."""
        # Bond strength based on group types
        bond_strengths = {
            ("epoxy", "primary_amine"): 350.0,  # kJ/mol
            ("epoxy", "secondary_amine"): 320.0,
            ("epoxy", "hydroxyl"): 280.0,
            ("anhydride", "primary_amine"): 300.0,
            ("isocyanate", "hydroxyl"): 330.0,
            ("vinyl", "vinyl"): 250.0,
        }

        pair1 = (group1.group_type, group2.group_type)
        pair2 = (group2.group_type, group1.group_type)

        return bond_strengths.get(pair1, bond_strengths.get(pair2, 250.0))

    def _analyze_network_topology(
        self, reactive_groups: List[ReactiveGroup], bonds: List[CrosslinkBond]
    ) -> NetworkTopology:
        """Analyze the topology of the crosslinked network."""
        if self.use_gpu and CUPY_AVAILABLE:
            return self._analyze_network_topology_gpu(reactive_groups, bonds)
        else:
            return self._analyze_network_topology_cpu(reactive_groups, bonds)

    def _analyze_network_topology_cpu(
        self, reactive_groups: List[ReactiveGroup], bonds: List[CrosslinkBond]
    ) -> NetworkTopology:
        """CPU-based network topology analysis."""
        n_groups = len(reactive_groups)

        # Build adjacency matrix
        adjacency_matrix = np.zeros((n_groups, n_groups), dtype=np.int32)

        for bond in bonds:
            i, j = bond.group1_id, bond.group2_id
            if i < n_groups and j < n_groups:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

        # Use NetworkX for cluster analysis if available
        if NETWORKX_AVAILABLE:
            G = nx.from_numpy_array(adjacency_matrix)
            connected_components = list(nx.connected_components(G))
            cluster_sizes = [len(component) for component in connected_components]
        else:
            # Fallback: simple connected component analysis
            cluster_sizes = self._find_clusters_simple(adjacency_matrix)

        largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0
        gel_fraction = largest_cluster_size / n_groups if n_groups > 0 else 0.0

        # Calculate crosslink density
        total_bonds = len(bonds)
        crosslink_density = total_bonds / n_groups if n_groups > 0 else 0.0

        # Calculate average functionality
        functionalities = [len(group.connected_to) for group in reactive_groups]
        average_functionality = np.mean(functionalities) if functionalities else 0.0

        return NetworkTopology(
            nodes=reactive_groups,
            bonds=bonds,
            adjacency_matrix=adjacency_matrix,
            cluster_sizes=cluster_sizes,
            largest_cluster_size=largest_cluster_size,
            gel_fraction=gel_fraction,
            crosslink_density=crosslink_density,
            average_functionality=average_functionality,
        )

    def _analyze_network_topology_gpu(
        self, reactive_groups: List[ReactiveGroup], bonds: List[CrosslinkBond]
    ) -> NetworkTopology:
        """GPU-accelerated network topology analysis."""
        n_groups = len(reactive_groups)

        # Build adjacency matrix on GPU
        adjacency_matrix_gpu = cp.zeros((n_groups, n_groups), dtype=cp.int32)

        # Transfer bond data to GPU
        if bonds:
            bond_indices = cp.array(
                [[bond.group1_id, bond.group2_id] for bond in bonds]
            )

            # Use advanced indexing to set adjacency matrix
            i_indices = bond_indices[:, 0]
            j_indices = bond_indices[:, 1]

            # Filter valid indices
            valid_mask = (i_indices < n_groups) & (j_indices < n_groups)
            i_valid = i_indices[valid_mask]
            j_valid = j_indices[valid_mask]

            adjacency_matrix_gpu[i_valid, j_valid] = 1
            adjacency_matrix_gpu[j_valid, i_valid] = 1

        # Connected components analysis on GPU
        cluster_sizes = self._find_clusters_gpu(adjacency_matrix_gpu)

        # Transfer results back to CPU
        adjacency_matrix = cp.asnumpy(adjacency_matrix_gpu)
        cluster_sizes = cp.asnumpy(cluster_sizes).tolist()

        largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0
        gel_fraction = largest_cluster_size / n_groups if n_groups > 0 else 0.0

        # Calculate crosslink density
        total_bonds = len(bonds)
        crosslink_density = total_bonds / n_groups if n_groups > 0 else 0.0

        # Calculate average functionality
        functionalities = [len(group.connected_to) for group in reactive_groups]
        average_functionality = np.mean(functionalities) if functionalities else 0.0

        return NetworkTopology(
            nodes=reactive_groups,
            bonds=bonds,
            adjacency_matrix=adjacency_matrix,
            cluster_sizes=cluster_sizes,
            largest_cluster_size=largest_cluster_size,
            gel_fraction=gel_fraction,
            crosslink_density=crosslink_density,
            average_functionality=average_functionality,
        )

    def _find_clusters_simple(self, adjacency_matrix: np.ndarray) -> List[int]:
        """Simple connected component analysis without NetworkX."""
        n = adjacency_matrix.shape[0]
        visited = np.zeros(n, dtype=bool)
        cluster_sizes = []

        for i in range(n):
            if not visited[i]:
                # BFS to find connected component
                cluster_size = 0
                queue = [i]
                visited[i] = True

                while queue:
                    node = queue.pop(0)
                    cluster_size += 1

                    # Find neighbors
                    neighbors = np.where(adjacency_matrix[node] == 1)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)

                cluster_sizes.append(cluster_size)

        return cluster_sizes

    def _find_clusters_gpu(self, adjacency_matrix_gpu):
        """GPU-accelerated connected component analysis."""
        n = adjacency_matrix_gpu.shape[0]

        # Simple iterative approach for connected components on GPU
        labels = cp.arange(n)

        # Iteratively merge connected components
        for _ in range(n):  # Maximum n iterations needed
            old_labels = labels.copy()

            # For each node, take minimum label of neighbors
            for i in range(n):
                neighbors = cp.where(adjacency_matrix_gpu[i] == 1)[0]
                if len(neighbors) > 0:
                    min_label = cp.min(labels[neighbors])
                    labels[i] = min(labels[i], min_label)

            # Check convergence
            if cp.array_equal(labels, old_labels):
                break

        # Count cluster sizes
        unique_labels, counts = cp.unique(labels, return_counts=True)
        return counts

    def _detect_gel_point(self, topology: NetworkTopology) -> bool:
        """Detect if the system has reached the gel point."""
        # Gel point is typically when the largest cluster contains >50% of nodes
        gel_threshold = 0.5
        return topology.gel_fraction > gel_threshold

    def _predict_mechanical_properties(
        self, topology: NetworkTopology, kinetics: CuringKinetics
    ) -> Dict[str, float]:
        """Predict mechanical properties from network topology."""
        properties = {}

        # Elastic modulus (simplified rubber elasticity theory)
        # E = 3 * rho * R * T * nu_e
        # where nu_e is effective crosslink density
        R = 8.314  # Gas constant
        T = kinetics.temperature
        rho = 1.2  # Assumed density (g/cm³)

        # Effective crosslink density (mol/cm³)
        nu_e = topology.crosslink_density * 1e-3  # Convert to mol/cm³

        elastic_modulus = 3 * rho * R * T * nu_e / 1000  # Convert to MPa
        properties["elastic_modulus_mpa"] = max(0.1, elastic_modulus)

        # Glass transition temperature (empirical correlation)
        # Tg increases with crosslink density
        base_tg = 300.0  # Base Tg in K
        tg_increase = topology.crosslink_density * 50.0  # K per crosslink
        properties["glass_transition_temp_k"] = base_tg + tg_increase

        # Tensile strength (empirical)
        tensile_strength = elastic_modulus * 0.1 * topology.gel_fraction
        properties["tensile_strength_mpa"] = max(1.0, tensile_strength)

        # Elongation at break (inversely related to crosslink density)
        elongation = max(0.01, 0.5 / (1 + topology.crosslink_density))
        properties["elongation_at_break"] = elongation

        return properties

    def _predict_thermal_properties(
        self, topology: NetworkTopology, kinetics: CuringKinetics
    ) -> Dict[str, float]:
        """Predict thermal properties from network topology."""
        properties = {}

        # Thermal conductivity (empirical correlation)
        base_conductivity = 0.2  # W/m·K for polymer
        crosslink_enhancement = topology.crosslink_density * 0.05
        properties["thermal_conductivity_w_mk"] = (
            base_conductivity + crosslink_enhancement
        )

        # Coefficient of thermal expansion (decreases with crosslinking)
        base_cte = 80e-6  # 1/K
        cte_reduction = topology.crosslink_density * 10e-6
        properties["thermal_expansion_coeff"] = max(20e-6, base_cte - cte_reduction)

        # Heat capacity (approximately constant)
        properties["heat_capacity_j_gk"] = 1.5  # J/g·K

        # Decomposition temperature (increases with crosslinking)
        base_decomp = 573.15  # 300°C in K
        decomp_increase = topology.crosslink_density * 20.0
        properties["decomposition_temp_k"] = base_decomp + decomp_increase

        return properties


class NetworkTopologyAnalyzer:
    """Advanced network topology analysis for thermoset systems."""

    def __init__(self, use_gpu: bool = True):
        """Initialize the network topology analyzer."""
        self.use_gpu = use_gpu and CUPY_AVAILABLE

    def calculate_network_descriptors(
        self, topology: NetworkTopology
    ) -> Dict[str, float]:
        """Calculate comprehensive network topology descriptors."""
        descriptors = {}

        # Basic descriptors
        descriptors["crosslink_density"] = topology.crosslink_density
        descriptors["gel_fraction"] = topology.gel_fraction
        descriptors["average_functionality"] = topology.average_functionality
        descriptors["largest_cluster_fraction"] = (
            topology.largest_cluster_size / len(topology.nodes) if topology.nodes else 0
        )

        # Advanced descriptors
        if len(topology.cluster_sizes) > 0:
            descriptors["cluster_size_variance"] = np.var(topology.cluster_sizes)
            descriptors["cluster_size_skewness"] = self._calculate_skewness(
                topology.cluster_sizes
            )
            descriptors["number_of_clusters"] = len(topology.cluster_sizes)
        else:
            descriptors["cluster_size_variance"] = 0
            descriptors["cluster_size_skewness"] = 0
            descriptors["number_of_clusters"] = 0

        # Network connectivity descriptors
        adj_matrix = topology.adjacency_matrix
        if adj_matrix.size > 0:
            descriptors["network_density"] = np.sum(adj_matrix) / (
                adj_matrix.shape[0] ** 2
            )
            descriptors["average_degree"] = np.mean(np.sum(adj_matrix, axis=1))
            descriptors["degree_variance"] = np.var(np.sum(adj_matrix, axis=1))
        else:
            descriptors["network_density"] = 0
            descriptors["average_degree"] = 0
            descriptors["degree_variance"] = 0

        return descriptors

    def _calculate_skewness(self, data: List[int]) -> float:
        """Calculate skewness of cluster size distribution."""
        if len(data) < 3:
            return 0.0

        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)

        if std == 0:
            return 0.0

        skewness = np.mean(((data_array - mean) / std) ** 3)
        return skewness


class GelPointDetector:
    """Specialized class for gel point detection and analysis."""

    def __init__(self):
        """Initialize the gel point detector."""
        pass

    def detect_gel_point_percolation(self, topology: NetworkTopology) -> Dict[str, Any]:
        """Detect gel point using percolation theory."""
        result = {
            "is_gel": False,
            "gel_fraction": topology.gel_fraction,
            "percolation_threshold": 0.5,
            "largest_cluster_size": topology.largest_cluster_size,
            "total_nodes": len(topology.nodes),
        }

        # Classical percolation threshold
        if topology.gel_fraction > 0.5:
            result["is_gel"] = True

        # Additional criteria based on cluster size distribution
        if len(topology.cluster_sizes) > 1:
            sorted_sizes = sorted(topology.cluster_sizes, reverse=True)
            if len(sorted_sizes) > 1:
                size_ratio = sorted_sizes[0] / sorted_sizes[1]
                result["size_ratio"] = size_ratio

                # If largest cluster is much larger than second largest
                if size_ratio > 5.0 and topology.gel_fraction > 0.3:
                    result["is_gel"] = True

        return result

    def analyze_gel_evolution(
        self, curing_results: List[CuringResult]
    ) -> Dict[str, Any]:
        """Analyze gel point evolution across multiple curing conditions."""
        gel_points = []

        for result in curing_results:
            if result.gel_point_time is not None:
                gel_points.append(
                    {
                        "time": result.gel_point_time,
                        "conversion": result.gel_point_conversion,
                        "temperature": result.simulation_metadata.get("temperature", 0),
                    }
                )

        analysis = {
            "gel_points": gel_points,
            "average_gel_time": np.mean([gp["time"] for gp in gel_points])
            if gel_points
            else None,
            "average_gel_conversion": np.mean([gp["conversion"] for gp in gel_points])
            if gel_points
            else None,
        }

        return analysis


def validate_against_experimental_data(
    simulation_results: List[CuringResult], experimental_data: Dict[str, Any]
) -> Dict[str, float]:
    """
    Validate simulation results against experimental curing data.

    Args:
        simulation_results: List of simulation results
        experimental_data: Dictionary containing experimental measurements

    Returns:
        Dictionary with validation metrics
    """
    validation_metrics = {}

    # Compare gel point times
    if "gel_point_times" in experimental_data:
        sim_gel_times = [
            r.gel_point_time for r in simulation_results if r.gel_point_time is not None
        ]
        exp_gel_times = experimental_data["gel_point_times"]

        if sim_gel_times and exp_gel_times:
            # Calculate mean absolute error
            min_length = min(len(sim_gel_times), len(exp_gel_times))
            mae_gel_time = np.mean(
                np.abs(
                    np.array(sim_gel_times[:min_length])
                    - np.array(exp_gel_times[:min_length])
                )
            )
            validation_metrics["gel_point_mae"] = mae_gel_time

    # Compare final properties
    if "elastic_modulus" in experimental_data:
        sim_moduli = [
            r.mechanical_properties.get("elastic_modulus_mpa", 0)
            for r in simulation_results
        ]
        exp_moduli = experimental_data["elastic_modulus"]

        if sim_moduli and exp_moduli:
            min_length = min(len(sim_moduli), len(exp_moduli))
            mae_modulus = np.mean(
                np.abs(
                    np.array(sim_moduli[:min_length])
                    - np.array(exp_moduli[:min_length])
                )
            )
            validation_metrics["elastic_modulus_mae"] = mae_modulus

    # Compare glass transition temperatures
    if "glass_transition_temps" in experimental_data:
        sim_tgs = [
            r.thermal_properties.get("glass_transition_temp_k", 0)
            for r in simulation_results
        ]
        exp_tgs = experimental_data["glass_transition_temps"]

        if sim_tgs and exp_tgs:
            min_length = min(len(sim_tgs), len(exp_tgs))
            mae_tg = np.mean(
                np.abs(np.array(sim_tgs[:min_length]) - np.array(exp_tgs[:min_length]))
            )
            validation_metrics["glass_transition_mae"] = mae_tg

    return validation_metrics

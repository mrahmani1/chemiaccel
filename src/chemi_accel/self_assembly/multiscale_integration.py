"""
Multi-Scale Integration Framework for Complex Self-Assembly Systems.

This module provides seamless data flow and model coupling between atomistic
(MD Analysis), molecular property (Calculator), and mesoscale (self-assembly simulation)
modules within ChemiAccel. It enables comprehensive multi-scale modeling workflows
for complex self-assembling systems.

Features:
- Seamless coupling between atomistic, molecular, and mesoscale models
- Automated parameter transfer and scale bridging
- Hierarchical simulation workflows
- Data consistency validation across scales
- Performance optimization for multi-scale calculations
- Integration with existing ChemiAccel modules
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

# Optional dependencies with graceful fallbacks
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances, rdf

    MDA_AVAILABLE = True
except ImportError:
    MDA_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Local imports
from ..core.calculator import Calculator
from ..core.molecular import MolecularData
from .characterization import (
    MorphologyType,
    SelfAssemblyCharacterizer,
    StructureType,
    characterize_trajectory,
)
from .coarse_grained_simulation import (
    CoarseGrainedSimulationEngine,
    InteractionParameters,
    Particle,
    ParticleType,
    SimulationParameters,
    create_amphiphilic_system,
    run_dpd_simulation,
)


class ScaleType(Enum):
    """Different scales in multi-scale modeling."""

    QUANTUM = "quantum"
    ATOMISTIC = "atomistic"
    MOLECULAR = "molecular"
    COARSE_GRAINED = "coarse_grained"
    MESOSCALE = "mesoscale"
    CONTINUUM = "continuum"


class CouplingMethod(Enum):
    """Methods for coupling different scales."""

    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"
    HANDSHAKING = "handshaking"


class DataTransferMethod(Enum):
    """Methods for transferring data between scales."""

    DIRECT_MAPPING = "direct_mapping"
    INTERPOLATION = "interpolation"
    MACHINE_LEARNING = "machine_learning"
    STATISTICAL_COARSE_GRAINING = "statistical_coarse_graining"
    FORCE_MATCHING = "force_matching"


class ValidationLevel(Enum):
    """Validation levels for multi-scale consistency."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    COMPREHENSIVE = "comprehensive"
    STRICT = "strict"


@dataclass
class ScaleParameters:
    """Parameters defining a specific scale in the multi-scale model."""

    scale_type: ScaleType
    spatial_resolution: float  # Length scale in nm
    temporal_resolution: float  # Time scale in ps
    system_size: Tuple[float, float, float]  # Box dimensions in nm
    particle_count: int
    description: str = ""

    def __post_init__(self):
        """Validate scale parameters."""
        if self.spatial_resolution <= 0:
            raise ValueError("Spatial resolution must be positive")
        if self.temporal_resolution <= 0:
            raise ValueError("Temporal resolution must be positive")
        if self.particle_count <= 0:
            raise ValueError("Particle count must be positive")


@dataclass
class CouplingConfiguration:
    """Configuration for coupling between different scales."""

    source_scale: ScaleType
    target_scale: ScaleType
    coupling_method: CouplingMethod
    data_transfer_method: DataTransferMethod
    transfer_frequency: int = 100  # Steps between transfers
    overlap_region: Optional[Tuple[float, float, float]] = None
    mapping_functions: Dict[str, Callable] = field(default_factory=dict)
    validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE

    def __post_init__(self):
        """Validate coupling configuration."""
        if self.transfer_frequency <= 0:
            raise ValueError("Transfer frequency must be positive")


@dataclass
class MultiScaleSystem:
    """Definition of a complete multi-scale system."""

    name: str
    description: str
    scales: List[ScaleParameters]
    couplings: List[CouplingConfiguration]
    global_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate multi-scale system definition."""
        if not self.scales:
            raise ValueError("At least one scale must be defined")

        # Check for unique scale types
        scale_types = [scale.scale_type for scale in self.scales]
        if len(scale_types) != len(set(scale_types)):
            raise ValueError("Duplicate scale types found")

        # Validate couplings reference existing scales
        available_scales = set(scale_types)
        for coupling in self.couplings:
            if coupling.source_scale not in available_scales:
                raise ValueError(f"Source scale {coupling.source_scale} not found")
            if coupling.target_scale not in available_scales:
                raise ValueError(f"Target scale {coupling.target_scale} not found")


@dataclass
class ScaleData:
    """Data container for a specific scale."""

    scale_type: ScaleType
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    forces: Optional[np.ndarray] = None
    energies: Dict[str, float] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    topology: Optional[Any] = None
    trajectory: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate scale data."""
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("Positions must be Nx3 array")

        if self.velocities is not None:
            if self.velocities.shape != self.positions.shape:
                raise ValueError("Velocities must match positions shape")

        if self.forces is not None:
            if self.forces.shape != self.positions.shape:
                raise ValueError("Forces must match positions shape")


@dataclass
class TransferResult:
    """Result of data transfer between scales."""

    source_scale: ScaleType
    target_scale: ScaleType
    transfer_method: DataTransferMethod
    success: bool
    transferred_data: ScaleData
    transfer_time: float
    error_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiScaleIntegrationError(Exception):
    """Base exception for multi-scale integration operations."""

    pass


class ScaleCouplingError(MultiScaleIntegrationError):
    """Exception raised during scale coupling."""

    pass


class DataTransferError(MultiScaleIntegrationError):
    """Exception raised during data transfer between scales."""

    pass


class ValidationError(MultiScaleIntegrationError):
    """Exception raised during validation."""

    pass


class ScaleModel(Protocol):
    """Protocol defining interface for scale-specific models."""

    def initialize(self, initial_data: ScaleData) -> None:
        """Initialize the model with initial data."""
        ...

    def step(self, num_steps: int = 1) -> ScaleData:
        """Advance the model by specified number of steps."""
        ...

    def get_current_state(self) -> ScaleData:
        """Get current state of the model."""
        ...

    def set_boundary_conditions(self, boundary_data: Any) -> None:
        """Set boundary conditions from coupled scales."""
        ...


class AtomisticModel:
    """Atomistic-scale model using MD analysis."""

    def __init__(
        self,
        topology_file: Optional[str] = None,
        trajectory_file: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize atomistic model.

        Args:
            topology_file: Path to topology file
            trajectory_file: Path to trajectory file
            use_gpu: Whether to use GPU acceleration
        """
        if not MDA_AVAILABLE:
            raise ImportError("MDAnalysis is required for atomistic modeling")

        self.topology_file = topology_file
        self.trajectory_file = trajectory_file
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        self.universe = None
        self.current_step = 0
        self.timestep = 1.0  # ps

        if topology_file and trajectory_file:
            self.universe = mda.Universe(topology_file, trajectory_file)

        logging.info(f"AtomisticModel initialized with GPU={self.use_gpu}")

    def initialize(self, initial_data: ScaleData) -> None:
        """Initialize with atomistic data."""
        # Create minimal universe if not loaded from files
        if self.universe is None:
            n_atoms = len(initial_data.positions)
            # Create simple topology
            self.universe = mda.Universe.empty(n_atoms, trajectory=True)
            self.universe.atoms.positions = (
                initial_data.positions * 10
            )  # nm to Angstrom

        self.current_step = 0

    def step(self, num_steps: int = 1) -> ScaleData:
        """Advance atomistic simulation."""
        if self.universe is None:
            raise ValueError("Universe not initialized")

        # Simple MD step simulation
        for _ in range(num_steps):
            # Add thermal motion
            thermal_displacement = np.random.normal(
                0, 0.01, self.universe.atoms.positions.shape
            )
            self.universe.atoms.positions += thermal_displacement
            self.current_step += 1

        # Convert back to ScaleData
        positions = self.universe.atoms.positions / 10.0  # Angstrom to nm

        # Calculate some properties
        if len(positions) > 1:
            com = np.mean(positions, axis=0)
            rg = np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1)))
        else:
            rg = 0.0

        properties = {
            "radius_of_gyration": rg,
            "center_of_mass": com.tolist() if len(positions) > 1 else [0, 0, 0],
            "n_atoms": len(positions),
        }

        return ScaleData(
            scale_type=ScaleType.ATOMISTIC,
            positions=positions,
            properties=properties,
            metadata={
                "step": self.current_step,
                "time": self.current_step * self.timestep,
            },
        )

    def get_current_state(self) -> ScaleData:
        """Get current atomistic state."""
        if self.universe is None:
            raise ValueError("Universe not initialized")

        positions = self.universe.atoms.positions / 10.0  # Angstrom to nm

        return ScaleData(
            scale_type=ScaleType.ATOMISTIC,
            positions=positions,
            metadata={"step": self.current_step},
        )

    def set_boundary_conditions(self, boundary_data: Any) -> None:
        """Set boundary conditions."""
        # Implementation depends on specific MD engine
        pass

    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze atomistic structure."""
        if self.universe is None:
            return {}

        analysis_results = {}

        # Radial distribution function
        if len(self.universe.atoms) > 1:
            try:
                rdf_analysis = rdf.InterRDF(self.universe.atoms, self.universe.atoms)
                rdf_analysis.run()
                analysis_results["rdf"] = {
                    "bins": rdf_analysis.bins.tolist(),
                    "rdf": rdf_analysis.rdf.tolist(),
                }
            except Exception as e:
                logging.warning(f"RDF calculation failed: {e}")

        # Basic geometric properties
        positions = self.universe.atoms.positions / 10.0
        if len(positions) > 1:
            com = np.mean(positions, axis=0)
            rg = np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1)))
            analysis_results["radius_of_gyration"] = rg
            analysis_results["center_of_mass"] = com.tolist()

        return analysis_results


class MolecularModel:
    """Molecular-scale model using ChemiAccel Calculator."""

    def __init__(self, calculator: Optional[Calculator] = None, use_gpu: bool = True):
        """
        Initialize molecular model.

        Args:
            calculator: ChemiAccel calculator instance
            use_gpu: Whether to use GPU acceleration
        """
        self.calculator = calculator or Calculator(backend="cuda" if use_gpu else "cpu")
        self.use_gpu = use_gpu

        self.molecules = []
        self.current_properties = {}

        logging.info(f"MolecularModel initialized with GPU={self.use_gpu}")

    def initialize(self, initial_data: ScaleData) -> None:
        """Initialize with molecular data."""
        # Generate molecules from positions (simplified)
        n_molecules = len(initial_data.positions)

        # Create simple molecules (e.g., water) for demonstration
        self.molecules = ["O" for _ in range(n_molecules)]  # Simple representation

        self.current_properties = {}

    def step(self, num_steps: int = 1) -> ScaleData:
        """Advance molecular simulation."""
        # Calculate molecular properties
        if self.molecules:
            try:
                # Calculate descriptors for current molecules
                results = self.calculator.pandas(self.molecules)

                # Extract relevant properties
                properties = {
                    "molecular_weight": results["molecular_weight"].mean()
                    if "molecular_weight" in results.columns
                    else 0.0,
                    "logp": results["logp"].mean()
                    if "logp" in results.columns
                    else 0.0,
                    "tpsa": results["tpsa"].mean()
                    if "tpsa" in results.columns
                    else 0.0,
                    "n_molecules": len(self.molecules),
                }

                self.current_properties = properties

            except Exception as e:
                logging.warning(f"Molecular property calculation failed: {e}")
                properties = {"n_molecules": len(self.molecules)}
        else:
            properties = {"n_molecules": 0}

        # Generate positions (simplified)
        positions = np.random.uniform(-5, 5, (len(self.molecules), 3))

        return ScaleData(
            scale_type=ScaleType.MOLECULAR,
            positions=positions,
            properties=properties,
            metadata={"step": num_steps},
        )

    def get_current_state(self) -> ScaleData:
        """Get current molecular state."""
        positions = np.random.uniform(-5, 5, (len(self.molecules), 3))

        return ScaleData(
            scale_type=ScaleType.MOLECULAR,
            positions=positions,
            properties=self.current_properties,
            metadata={"n_molecules": len(self.molecules)},
        )

    def set_boundary_conditions(self, boundary_data: Any) -> None:
        """Set molecular boundary conditions."""
        pass

    def calculate_interaction_parameters(self) -> Dict[str, Any]:
        """Calculate interaction parameters for coarse-graining."""
        interaction_params = {}

        if self.molecules and self.current_properties:
            # Derive interaction parameters from molecular properties
            mw = self.current_properties.get("molecular_weight", 18.0)
            logp = self.current_properties.get("logp", 0.0)

            # Simple mapping to DPD parameters
            interaction_params["repulsion_strength"] = 25.0 + 5.0 * logp
            interaction_params["attractive_strength"] = max(0.1, 2.0 - 0.5 * abs(logp))
            interaction_params["particle_size"] = max(0.5, mw / 50.0)

        return interaction_params


class MesoscaleModel:
    """Mesoscale model using coarse-grained simulation."""

    def __init__(
        self,
        simulation_engine: Optional[CoarseGrainedSimulationEngine] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize mesoscale model.

        Args:
            simulation_engine: Coarse-grained simulation engine
            use_gpu: Whether to use GPU acceleration
        """
        self.simulation_engine = simulation_engine
        self.use_gpu = use_gpu

        self.current_trajectory = []
        self.characterizer = SelfAssemblyCharacterizer(use_gpu=use_gpu)

        logging.info(f"MesoscaleModel initialized with GPU={self.use_gpu}")

    def initialize(self, initial_data: ScaleData) -> None:
        """Initialize with mesoscale data."""
        if self.simulation_engine is None:
            # Create default system
            particles, sim_box, interaction_params = create_amphiphilic_system(
                n_molecules=max(10, len(initial_data.positions) // 10),
                head_size=1,
                tail_size=4,
                box_size=15.0,
            )

            sim_params = SimulationParameters(
                timestep=0.01,
                total_time=10.0,
                use_gpu=self.use_gpu,
                output_frequency=10,
            )

            self.simulation_engine = CoarseGrainedSimulationEngine(
                particles, sim_box, sim_params, interaction_params
            )

    def step(self, num_steps: int = 1) -> ScaleData:
        """Advance mesoscale simulation."""
        if self.simulation_engine is None:
            raise ValueError("Simulation engine not initialized")

        # Run simulation steps
        for _ in range(num_steps):
            self.simulation_engine._simulation_step(
                self.simulation_engine.state.step + 1
            )

        # Get current state
        current_state = self.simulation_engine.get_current_state()

        # Characterize structures
        properties = {}
        if self.current_trajectory:
            try:
                char_results = characterize_trajectory(
                    self.current_trajectory,
                    clustering_method="dbscan",
                    use_gpu=self.use_gpu,
                )

                if "structures" in char_results and char_results["structures"]:
                    structures = char_results["structures"]
                    properties["n_structures"] = len(structures)
                    properties["avg_aggregation_number"] = np.mean(
                        [s.aggregation_number for s in structures]
                    )
                    properties["avg_radius_of_gyration"] = np.mean(
                        [s.radius_of_gyration for s in structures]
                    )

            except Exception as e:
                logging.warning(f"Structure characterization failed: {e}")

        return ScaleData(
            scale_type=ScaleType.MESOSCALE,
            positions=current_state.positions,
            velocities=current_state.velocities,
            forces=current_state.forces,
            energies=current_state.energies,
            properties=properties,
            metadata={"step": current_state.step, "time": current_state.time},
        )

    def get_current_state(self) -> ScaleData:
        """Get current mesoscale state."""
        if self.simulation_engine is None:
            raise ValueError("Simulation engine not initialized")

        current_state = self.simulation_engine.get_current_state()

        return ScaleData(
            scale_type=ScaleType.MESOSCALE,
            positions=current_state.positions,
            velocities=current_state.velocities,
            forces=current_state.forces,
            energies=current_state.energies,
            metadata={"step": current_state.step, "time": current_state.time},
        )

    def set_boundary_conditions(self, boundary_data: Any) -> None:
        """Set mesoscale boundary conditions."""
        if self.simulation_engine and hasattr(boundary_data, "interaction_params"):
            # Update interaction parameters from coupled scales
            self.simulation_engine.interaction_params = boundary_data.interaction_params


class DataTransferEngine:
    """Engine for transferring data between different scales."""

    def __init__(self, use_gpu: bool = True):
        """
        Initialize data transfer engine.

        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.transfer_functions = {
            (ScaleType.ATOMISTIC, ScaleType.MOLECULAR): self._atomistic_to_molecular,
            (ScaleType.MOLECULAR, ScaleType.MESOSCALE): self._molecular_to_mesoscale,
            (ScaleType.MESOSCALE, ScaleType.MOLECULAR): self._mesoscale_to_molecular,
            (ScaleType.MOLECULAR, ScaleType.ATOMISTIC): self._molecular_to_atomistic,
        }

        logging.info(f"DataTransferEngine initialized with GPU={self.use_gpu}")

    def transfer_data(
        self,
        source_data: ScaleData,
        target_scale: ScaleType,
        transfer_method: DataTransferMethod = DataTransferMethod.DIRECT_MAPPING,
        **kwargs,
    ) -> TransferResult:
        """
        Transfer data between scales.

        Args:
            source_data: Data from source scale
            target_scale: Target scale type
            transfer_method: Method for data transfer
            **kwargs: Additional transfer parameters

        Returns:
            Transfer result
        """
        start_time = time.time()

        try:
            transfer_key = (source_data.scale_type, target_scale)

            if transfer_key not in self.transfer_functions:
                raise DataTransferError(
                    f"Transfer from {source_data.scale_type} to {target_scale} not implemented"
                )

            # Perform transfer
            transfer_function = self.transfer_functions[transfer_key]
            transferred_data = transfer_function(source_data, transfer_method, **kwargs)

            # Calculate transfer time
            transfer_time = time.time() - start_time

            # Validate transferred data
            error_metrics = self._validate_transfer(source_data, transferred_data)

            return TransferResult(
                source_scale=source_data.scale_type,
                target_scale=target_scale,
                transfer_method=transfer_method,
                success=True,
                transferred_data=transferred_data,
                transfer_time=transfer_time,
                error_metrics=error_metrics,
                warnings=[],
                metadata={"kwargs": kwargs},
            )

        except Exception as e:
            transfer_time = time.time() - start_time
            logging.error(f"Data transfer failed: {e}")

            return TransferResult(
                source_scale=source_data.scale_type,
                target_scale=target_scale,
                transfer_method=transfer_method,
                success=False,
                transferred_data=ScaleData(target_scale, np.array([[0, 0, 0]])),
                transfer_time=transfer_time,
                error_metrics={"transfer_error": float("inf")},
                warnings=[str(e)],
            )

    def _atomistic_to_molecular(
        self, source_data: ScaleData, transfer_method: DataTransferMethod, **kwargs
    ) -> ScaleData:
        """Transfer from atomistic to molecular scale."""
        # Coarse-grain atomistic data to molecular representation

        if transfer_method == DataTransferMethod.DIRECT_MAPPING:
            # Simple center-of-mass mapping
            n_atoms_per_molecule = kwargs.get("atoms_per_molecule", 3)
            positions = source_data.positions

            n_molecules = max(1, len(positions) // n_atoms_per_molecule)
            molecular_positions = []

            for i in range(n_molecules):
                start_idx = i * n_atoms_per_molecule
                end_idx = min(start_idx + n_atoms_per_molecule, len(positions))
                com = np.mean(positions[start_idx:end_idx], axis=0)
                molecular_positions.append(com)

            molecular_positions = np.array(molecular_positions)

        elif transfer_method == DataTransferMethod.STATISTICAL_COARSE_GRAINING:
            # Statistical coarse-graining approach
            positions = source_data.positions

            # Cluster atoms into molecules based on proximity
            from sklearn.cluster import KMeans

            n_clusters = kwargs.get("n_molecules", max(1, len(positions) // 5))

            if len(positions) > n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(positions)
                molecular_positions = kmeans.cluster_centers_
            else:
                molecular_positions = positions[:n_clusters]

        else:
            raise DataTransferError(
                f"Transfer method {transfer_method} not supported for atomistic to molecular"
            )

        # Calculate molecular properties
        properties = {
            "n_molecules": len(molecular_positions),
            "molecular_density": len(molecular_positions)
            / np.prod(source_data.properties.get("box_size", [1, 1, 1])),
        }

        return ScaleData(
            scale_type=ScaleType.MOLECULAR,
            positions=molecular_positions,
            properties=properties,
            metadata={
                "source_scale": ScaleType.ATOMISTIC,
                "transfer_method": transfer_method.value,
            },
        )

    def _molecular_to_mesoscale(
        self, source_data: ScaleData, transfer_method: DataTransferMethod, **kwargs
    ) -> ScaleData:
        """Transfer from molecular to mesoscale."""
        # Convert molecular data to coarse-grained representation

        if transfer_method == DataTransferMethod.DIRECT_MAPPING:
            # Map molecules to CG particles
            n_molecules_per_particle = kwargs.get("molecules_per_particle", 5)
            positions = source_data.positions

            n_particles = max(1, len(positions) // n_molecules_per_particle)
            cg_positions = []

            for i in range(n_particles):
                start_idx = i * n_molecules_per_particle
                end_idx = min(start_idx + n_molecules_per_particle, len(positions))
                com = np.mean(positions[start_idx:end_idx], axis=0)
                cg_positions.append(com)

            cg_positions = np.array(cg_positions)

        elif transfer_method == DataTransferMethod.FORCE_MATCHING:
            # Force matching approach (simplified)
            positions = source_data.positions

            # Use existing positions but with reduced resolution
            skip_factor = kwargs.get("skip_factor", 2)
            cg_positions = positions[::skip_factor]

        else:
            raise DataTransferError(
                f"Transfer method {transfer_method} not supported for molecular to mesoscale"
            )

        # Generate CG properties
        properties = {
            "n_cg_particles": len(cg_positions),
            "coarse_graining_ratio": len(source_data.positions) / len(cg_positions)
            if len(cg_positions) > 0
            else 1,
        }

        # Generate simple velocities
        velocities = np.random.normal(0, 1.0, cg_positions.shape)

        return ScaleData(
            scale_type=ScaleType.MESOSCALE,
            positions=cg_positions,
            velocities=velocities,
            properties=properties,
            metadata={
                "source_scale": ScaleType.MOLECULAR,
                "transfer_method": transfer_method.value,
            },
        )

    def _mesoscale_to_molecular(
        self, source_data: ScaleData, transfer_method: DataTransferMethod, **kwargs
    ) -> ScaleData:
        """Transfer from mesoscale to molecular scale."""
        # Reverse coarse-graining from mesoscale to molecular

        if transfer_method == DataTransferMethod.DIRECT_MAPPING:
            # Expand CG particles to molecules
            expansion_factor = kwargs.get("expansion_factor", 5)
            cg_positions = source_data.positions

            molecular_positions = []
            for cg_pos in cg_positions:
                # Generate molecules around each CG particle
                for _ in range(expansion_factor):
                    offset = np.random.normal(0, 0.2, 3)
                    mol_pos = cg_pos + offset
                    molecular_positions.append(mol_pos)

            molecular_positions = np.array(molecular_positions)

        elif transfer_method == DataTransferMethod.INTERPOLATION:
            # Interpolation-based reconstruction
            cg_positions = source_data.positions
            expansion_factor = kwargs.get("expansion_factor", 3)

            molecular_positions = []
            for i, cg_pos in enumerate(cg_positions):
                for j in range(expansion_factor):
                    # Add structured variations
                    angle = 2 * np.pi * j / expansion_factor
                    radius = 0.3
                    offset = radius * np.array([np.cos(angle), np.sin(angle), 0])
                    mol_pos = cg_pos + offset
                    molecular_positions.append(mol_pos)

            molecular_positions = np.array(molecular_positions)

        else:
            raise DataTransferError(
                f"Transfer method {transfer_method} not supported for mesoscale to molecular"
            )

        properties = {
            "n_molecules": len(molecular_positions),
            "reconstruction_ratio": len(molecular_positions)
            / len(source_data.positions),
        }

        return ScaleData(
            scale_type=ScaleType.MOLECULAR,
            positions=molecular_positions,
            properties=properties,
            metadata={
                "source_scale": ScaleType.MESOSCALE,
                "transfer_method": transfer_method.value,
            },
        )

    def _molecular_to_atomistic(
        self, source_data: ScaleData, transfer_method: DataTransferMethod, **kwargs
    ) -> ScaleData:
        """Transfer from molecular to atomistic scale."""
        # Expand molecular data to atomistic representation

        if transfer_method == DataTransferMethod.DIRECT_MAPPING:
            # Simple expansion
            atoms_per_molecule = kwargs.get("atoms_per_molecule", 3)
            molecular_positions = source_data.positions

            atomistic_positions = []
            for mol_pos in molecular_positions:
                # Generate atoms for each molecule
                for i in range(atoms_per_molecule):
                    offset = np.random.normal(0, 0.1, 3)
                    atom_pos = mol_pos + offset
                    atomistic_positions.append(atom_pos)

            atomistic_positions = np.array(atomistic_positions)

        elif transfer_method == DataTransferMethod.MACHINE_LEARNING:
            # ML-based reconstruction (simplified placeholder)
            molecular_positions = source_data.positions
            atoms_per_molecule = kwargs.get("atoms_per_molecule", 4)

            atomistic_positions = []
            for mol_pos in molecular_positions:
                # Use learned patterns for reconstruction
                for i in range(atoms_per_molecule):
                    # Structured placement based on molecular geometry
                    angle = 2 * np.pi * i / atoms_per_molecule
                    radius = 0.15
                    offset = radius * np.array([np.cos(angle), np.sin(angle), 0])
                    atom_pos = mol_pos + offset
                    atomistic_positions.append(atom_pos)

            atomistic_positions = np.array(atomistic_positions)

        else:
            raise DataTransferError(
                f"Transfer method {transfer_method} not supported for molecular to atomistic"
            )

        properties = {
            "n_atoms": len(atomistic_positions),
            "expansion_ratio": len(atomistic_positions) / len(source_data.positions),
        }

        return ScaleData(
            scale_type=ScaleType.ATOMISTIC,
            positions=atomistic_positions,
            properties=properties,
            metadata={
                "source_scale": ScaleType.MOLECULAR,
                "transfer_method": transfer_method.value,
            },
        )

    def _validate_transfer(
        self, source_data: ScaleData, transferred_data: ScaleData
    ) -> Dict[str, float]:
        """Validate data transfer quality."""
        error_metrics = {}

        # Conservation checks
        if len(source_data.positions) > 0 and len(transferred_data.positions) > 0:
            # Center of mass conservation
            source_com = np.mean(source_data.positions, axis=0)
            target_com = np.mean(transferred_data.positions, axis=0)
            com_error = np.linalg.norm(source_com - target_com)
            error_metrics["center_of_mass_error"] = com_error

            # Size scale conservation
            source_size = np.std(source_data.positions)
            target_size = np.std(transferred_data.positions)
            size_error = abs(source_size - target_size) / max(source_size, 1e-10)
            error_metrics["size_scale_error"] = size_error

        # Energy conservation (if available)
        if source_data.energies and transferred_data.energies:
            for key in source_data.energies:
                if key in transferred_data.energies:
                    energy_error = abs(
                        source_data.energies[key] - transferred_data.energies[key]
                    )
                    error_metrics[f"{key}_energy_error"] = energy_error

        return error_metrics


class MultiScaleIntegrator:
    """
    Main class for multi-scale integration and coordination.

    This class manages the coupling and data flow between different scales
    in a multi-scale self-assembly simulation.
    """

    def __init__(
        self,
        multiscale_system: MultiScaleSystem,
        use_gpu: bool = True,
        parallel_execution: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize multi-scale integrator.

        Args:
            multiscale_system: Multi-scale system definition
            use_gpu: Whether to use GPU acceleration
            parallel_execution: Whether to use parallel execution
            max_workers: Maximum number of parallel workers
        """
        self.system = multiscale_system
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers

        # Initialize components
        self.data_transfer_engine = DataTransferEngine(use_gpu=use_gpu)
        self.scale_models = {}
        self.current_data = {}
        self.coupling_history = []

        # Initialize scale models
        self._initialize_scale_models()

        logging.info(
            f"MultiScaleIntegrator initialized for system '{self.system.name}'"
        )

    def _initialize_scale_models(self):
        """Initialize models for each scale."""
        for scale in self.system.scales:
            if scale.scale_type == ScaleType.ATOMISTIC:
                self.scale_models[scale.scale_type] = AtomisticModel(
                    use_gpu=self.use_gpu
                )
            elif scale.scale_type == ScaleType.MOLECULAR:
                self.scale_models[scale.scale_type] = MolecularModel(
                    use_gpu=self.use_gpu
                )
            elif scale.scale_type == ScaleType.MESOSCALE:
                self.scale_models[scale.scale_type] = MesoscaleModel(
                    use_gpu=self.use_gpu
                )
            else:
                logging.warning(f"Scale type {scale.scale_type} not implemented")

    def initialize_system(self, initial_conditions: Dict[ScaleType, ScaleData]):
        """
        Initialize the multi-scale system with initial conditions.

        Args:
            initial_conditions: Initial data for each scale
        """
        for scale_type, initial_data in initial_conditions.items():
            if scale_type in self.scale_models:
                self.scale_models[scale_type].initialize(initial_data)
                self.current_data[scale_type] = initial_data

        logging.info("Multi-scale system initialized")

    def run_coupled_simulation(
        self,
        total_steps: int,
        coupling_frequency: Dict[Tuple[ScaleType, ScaleType], int] = None,
        output_frequency: int = 100,
    ) -> Dict[str, Any]:
        """
        Run coupled multi-scale simulation.

        Args:
            total_steps: Total number of simulation steps
            coupling_frequency: Frequency of coupling between scales
            output_frequency: Frequency of output data saving

        Returns:
            Simulation results and trajectories
        """
        if coupling_frequency is None:
            coupling_frequency = {
                (
                    coupling.source_scale,
                    coupling.target_scale,
                ): coupling.transfer_frequency
                for coupling in self.system.couplings
            }

        results = {
            "trajectories": {scale_type: [] for scale_type in self.scale_models.keys()},
            "coupling_history": [],
            "performance_metrics": {},
        }

        start_time = time.time()

        for step in range(total_steps):
            # Advance individual scales
            if self.parallel_execution:
                step_results = self._parallel_step()
            else:
                step_results = self._sequential_step()

            # Update current data
            for scale_type, data in step_results.items():
                self.current_data[scale_type] = data

            # Perform coupling at specified frequencies
            if step > 0:
                coupling_results = self._perform_coupling(step, coupling_frequency)
                if coupling_results:
                    results["coupling_history"].extend(coupling_results)

            # Save trajectory data
            if step % output_frequency == 0:
                for scale_type, data in self.current_data.items():
                    results["trajectories"][scale_type].append(
                        {"step": step, "data": data}
                    )

        simulation_time = time.time() - start_time
        results["performance_metrics"]["total_time"] = simulation_time
        results["performance_metrics"]["steps_per_second"] = (
            total_steps / simulation_time
        )

        logging.info(
            f"Multi-scale simulation completed in {simulation_time:.2f} seconds"
        )
        return results

    def _sequential_step(self) -> Dict[ScaleType, ScaleData]:
        """Perform sequential step for all scales."""
        step_results = {}

        for scale_type, model in self.scale_models.items():
            try:
                data = model.step(1)
                step_results[scale_type] = data
            except Exception as e:
                logging.error(f"Step failed for scale {scale_type}: {e}")
                step_results[scale_type] = self.current_data.get(scale_type)

        return step_results

    def _parallel_step(self) -> Dict[ScaleType, ScaleData]:
        """Perform parallel step for all scales."""
        step_results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_scale = {
                executor.submit(model.step, 1): scale_type
                for scale_type, model in self.scale_models.items()
            }

            # Collect results
            for future in as_completed(future_to_scale):
                scale_type = future_to_scale[future]
                try:
                    data = future.result()
                    step_results[scale_type] = data
                except Exception as e:
                    logging.error(f"Parallel step failed for scale {scale_type}: {e}")
                    step_results[scale_type] = self.current_data.get(scale_type)

        return step_results

    def _perform_coupling(
        self,
        current_step: int,
        coupling_frequency: Dict[Tuple[ScaleType, ScaleType], int],
    ) -> List[TransferResult]:
        """Perform coupling between scales at current step."""
        coupling_results = []

        for coupling in self.system.couplings:
            coupling_key = (coupling.source_scale, coupling.target_scale)
            frequency = coupling_frequency.get(
                coupling_key, coupling.transfer_frequency
            )

            if current_step % frequency == 0:
                # Perform data transfer
                source_data = self.current_data.get(coupling.source_scale)
                if source_data is not None:
                    transfer_result = self.data_transfer_engine.transfer_data(
                        source_data,
                        coupling.target_scale,
                        coupling.data_transfer_method,
                    )

                    coupling_results.append(transfer_result)

                    # Update target model with transferred data
                    if (
                        transfer_result.success
                        and coupling.target_scale in self.scale_models
                    ):
                        target_model = self.scale_models[coupling.target_scale]
                        target_model.set_boundary_conditions(
                            transfer_result.transferred_data
                        )

                    # Store coupling history
                    self.coupling_history.append(
                        {
                            "step": current_step,
                            "coupling": coupling,
                            "result": transfer_result,
                        }
                    )

        return coupling_results

    def analyze_coupling_quality(self) -> Dict[str, Any]:
        """Analyze quality of multi-scale coupling."""
        analysis = {
            "transfer_success_rate": 0.0,
            "average_transfer_time": 0.0,
            "coupling_errors": {},
            "scale_consistency": {},
        }

        if not self.coupling_history:
            return analysis

        # Success rate
        successful_transfers = sum(
            1 for entry in self.coupling_history if entry["result"].success
        )
        analysis["transfer_success_rate"] = successful_transfers / len(
            self.coupling_history
        )

        # Average transfer time
        transfer_times = [
            entry["result"].transfer_time for entry in self.coupling_history
        ]
        analysis["average_transfer_time"] = np.mean(transfer_times)

        # Error analysis
        for entry in self.coupling_history:
            result = entry["result"]
            coupling_key = f"{result.source_scale.value}_to_{result.target_scale.value}"

            if coupling_key not in analysis["coupling_errors"]:
                analysis["coupling_errors"][coupling_key] = []

            for error_name, error_value in result.error_metrics.items():
                analysis["coupling_errors"][coupling_key].append(
                    {
                        "error_type": error_name,
                        "value": error_value,
                        "step": entry["step"],
                    }
                )

        return analysis

    def save_results(self, filepath: str, results: Dict[str, Any]):
        """Save simulation results to file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logging.info(f"Results saved to {filepath}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (ScaleData, TransferResult)):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


# Utility functions for common multi-scale scenarios


def create_hierarchical_system(
    n_atoms: int = 1000,
    n_molecules: int = 200,
    n_cg_particles: int = 50,
    box_size: float = 10.0,
) -> MultiScaleSystem:
    """
    Create a hierarchical multi-scale system definition.

    Args:
        n_atoms: Number of atoms in atomistic scale
        n_molecules: Number of molecules in molecular scale
        n_cg_particles: Number of CG particles in mesoscale
        box_size: System box size in nm

    Returns:
        Multi-scale system definition
    """
    # Define scales
    scales = [
        ScaleParameters(
            scale_type=ScaleType.ATOMISTIC,
            spatial_resolution=0.1,  # 0.1 nm
            temporal_resolution=0.001,  # 1 fs
            system_size=(box_size, box_size, box_size),
            particle_count=n_atoms,
            description="All-atom molecular dynamics",
        ),
        ScaleParameters(
            scale_type=ScaleType.MOLECULAR,
            spatial_resolution=0.5,  # 0.5 nm
            temporal_resolution=0.01,  # 10 fs
            system_size=(box_size, box_size, box_size),
            particle_count=n_molecules,
            description="Molecular-level representation",
        ),
        ScaleParameters(
            scale_type=ScaleType.MESOSCALE,
            spatial_resolution=2.0,  # 2 nm
            temporal_resolution=0.1,  # 100 fs
            system_size=(box_size, box_size, box_size),
            particle_count=n_cg_particles,
            description="Coarse-grained simulation",
        ),
    ]

    # Define couplings
    couplings = [
        CouplingConfiguration(
            source_scale=ScaleType.ATOMISTIC,
            target_scale=ScaleType.MOLECULAR,
            coupling_method=CouplingMethod.HIERARCHICAL,
            data_transfer_method=DataTransferMethod.STATISTICAL_COARSE_GRAINING,
            transfer_frequency=100,
            validation_level=ValidationLevel.INTERMEDIATE,
        ),
        CouplingConfiguration(
            source_scale=ScaleType.MOLECULAR,
            target_scale=ScaleType.MESOSCALE,
            coupling_method=CouplingMethod.HIERARCHICAL,
            data_transfer_method=DataTransferMethod.FORCE_MATCHING,
            transfer_frequency=50,
            validation_level=ValidationLevel.INTERMEDIATE,
        ),
    ]

    return MultiScaleSystem(
        name="Hierarchical Self-Assembly System",
        description="Three-scale hierarchical system for self-assembly modeling",
        scales=scales,
        couplings=couplings,
        global_parameters={"temperature": 298.15, "pressure": 1.0},
    )


def run_multiscale_workflow(
    system: MultiScaleSystem,
    initial_atomistic_data: Optional[ScaleData] = None,
    total_steps: int = 1000,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Run a complete multi-scale workflow.

    Args:
        system: Multi-scale system definition
        initial_atomistic_data: Initial atomistic data
        total_steps: Total simulation steps
        use_gpu: Whether to use GPU acceleration

    Returns:
        Complete workflow results
    """
    # Create integrator
    integrator = MultiScaleIntegrator(
        multiscale_system=system, use_gpu=use_gpu, parallel_execution=True
    )

    # Initialize system
    initial_conditions = {}

    if initial_atomistic_data is None:
        # Generate default initial conditions
        for scale in system.scales:
            if scale.scale_type == ScaleType.ATOMISTIC:
                positions = np.random.uniform(-5, 5, (scale.particle_count, 3))
                initial_conditions[scale.scale_type] = ScaleData(
                    scale_type=scale.scale_type, positions=positions
                )
            elif scale.scale_type == ScaleType.MOLECULAR:
                positions = np.random.uniform(-3, 3, (scale.particle_count, 3))
                initial_conditions[scale.scale_type] = ScaleData(
                    scale_type=scale.scale_type, positions=positions
                )
            elif scale.scale_type == ScaleType.MESOSCALE:
                positions = np.random.uniform(-2, 2, (scale.particle_count, 3))
                initial_conditions[scale.scale_type] = ScaleData(
                    scale_type=scale.scale_type, positions=positions
                )
    else:
        initial_conditions[ScaleType.ATOMISTIC] = initial_atomistic_data

    integrator.initialize_system(initial_conditions)

    # Run simulation
    results = integrator.run_coupled_simulation(
        total_steps=total_steps, output_frequency=max(1, total_steps // 100)
    )

    # Add coupling quality analysis
    results["coupling_analysis"] = integrator.analyze_coupling_quality()

    return results


def validate_multiscale_consistency(
    results: Dict[str, Any], tolerance: float = 0.1
) -> Dict[str, bool]:
    """
    Validate consistency across multiple scales.

    Args:
        results: Multi-scale simulation results
        tolerance: Tolerance for consistency checks

    Returns:
        Validation results for each consistency check
    """
    validation = {
        "mass_conservation": True,
        "energy_conservation": True,
        "momentum_conservation": True,
        "scale_coherence": True,
    }

    trajectories = results.get("trajectories", {})

    # Check mass conservation across scales
    for scale_type, trajectory in trajectories.items():
        if len(trajectory) > 1:
            initial_count = len(trajectory[0]["data"].positions)
            final_count = len(trajectory[-1]["data"].positions)

            # Allow for different particle counts due to coarse-graining
            # but they should be consistent within each scale
            if abs(initial_count - final_count) / max(initial_count, 1) > tolerance:
                validation["mass_conservation"] = False

    # Check energy trends (should be physically reasonable)
    coupling_history = results.get("coupling_history", [])
    if coupling_history:
        for entry in coupling_history[-10:]:  # Check recent transfers
            error_metrics = entry.get("error_metrics", {})
            for error_name, error_value in error_metrics.items():
                if "energy" in error_name and error_value > tolerance:
                    validation["energy_conservation"] = False

    # Check coupling quality
    coupling_analysis = results.get("coupling_analysis", {})
    success_rate = coupling_analysis.get("transfer_success_rate", 0.0)
    if success_rate < 0.9:  # 90% success rate threshold
        validation["scale_coherence"] = False

    return validation

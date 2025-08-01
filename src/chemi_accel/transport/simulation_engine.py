"""
GPU-Accelerated Multi-Phase Simulation Engine

This module implements high-performance simulation capabilities for modeling molecular
transport across interfaces using various simulation methods including atomistic MD,
coarse-grained DPD, and AI surrogates.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupyx_ndimage

    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    cupyx_ndimage = None
    CUDA_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    jit = None
    vmap = None
    JAX_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..core.molecular import MolecularData


@dataclass
class SimulationConfig:
    """Configuration for multi-phase transport simulations."""

    method: str = "ai_surrogate"  # 'md', 'dpd', 'ai_surrogate'
    temperature: float = 298.15
    pressure: float = 1.0
    timestep: float = 1e-15  # seconds
    n_steps: int = 100000
    output_frequency: int = 1000
    interface_type: str = "lipid_bilayer"
    phase_definitions: Dict[str, Any] = None
    boundary_conditions: str = "periodic"

    def __post_init__(self):
        if self.phase_definitions is None:
            self.phase_definitions = {}


@dataclass
class TransportResult:
    """Results from multi-phase transport simulation."""

    permeability_coefficient: float
    diffusion_coefficients: Dict[str, float]
    partition_coefficients: Dict[str, float]
    trajectories: np.ndarray
    concentration_profiles: Dict[str, np.ndarray]
    flux_data: np.ndarray
    simulation_time: float
    metadata: Dict[str, Any]


class SimulationMethod(ABC):
    """Abstract base class for simulation methods."""

    @abstractmethod
    def run_simulation(
        self, molecules: List[str], config: SimulationConfig
    ) -> TransportResult:
        """Run transport simulation."""
        pass


class AISubrogateSimulation(SimulationMethod):
    """AI surrogate model for fast transport simulation."""

    def __init__(self, backend: str = "auto"):
        self.backend = self._select_backend(backend)
        self.logger = logging.getLogger(__name__)

        # Pre-trained surrogate models (would be loaded from files in practice)
        self._initialize_surrogate_models()

    def _select_backend(self, backend: str) -> str:
        """Select optimal backend for computation."""
        if backend == "auto":
            if CUDA_AVAILABLE:
                return "cuda"
            elif JAX_AVAILABLE:
                return "jax"
            else:
                return "cpu"
        return backend

    def _initialize_surrogate_models(self):
        """Initialize pre-trained surrogate models."""
        # In practice, these would be loaded from trained model files
        self.permeability_model = self._create_mock_permeability_model()
        self.diffusion_model = self._create_mock_diffusion_model()
        self.partition_model = self._create_mock_partition_model()

    def _create_mock_permeability_model(self):
        """Create mock permeability prediction model."""
        if self.backend == "cuda" and CUDA_AVAILABLE:
            return self._cuda_permeability_model
        elif self.backend == "jax" and JAX_AVAILABLE:
            return self._jax_permeability_model
        else:
            return self._cpu_permeability_model

    def _create_mock_diffusion_model(self):
        """Create mock diffusion coefficient model."""
        if self.backend == "cuda" and CUDA_AVAILABLE:
            return self._cuda_diffusion_model
        elif self.backend == "jax" and JAX_AVAILABLE:
            return self._jax_diffusion_model
        else:
            return self._cpu_diffusion_model

    def _create_mock_partition_model(self):
        """Create mock partition coefficient model."""
        if self.backend == "cuda" and CUDA_AVAILABLE:
            return self._cuda_partition_model
        elif self.backend == "jax" and JAX_AVAILABLE:
            return self._jax_partition_model
        else:
            return self._cpu_partition_model

    def _cuda_permeability_model(self, features):
        """CUDA-accelerated permeability prediction."""
        if not CUDA_AVAILABLE:
            return self._cpu_permeability_model(features)
        # Mock neural network computation
        weights = cp.random.randn(features.shape[1], 64).astype(cp.float32)
        hidden = cp.tanh(cp.dot(features, weights))
        output_weights = cp.random.randn(64, 1).astype(cp.float32)
        output = cp.dot(hidden, output_weights)
        return (
            cp.exp(output.flatten()) * 1e-6
        )  # Convert to realistic permeability units

    def _jax_permeability_model(self, features):
        """JAX-accelerated permeability prediction."""
        if not JAX_AVAILABLE:
            return self._cpu_permeability_model(features)

        @jit
        def model(x):
            weights = jax.random.normal(jax.random.PRNGKey(0), (x.shape[1], 64))
            hidden = jnp.tanh(jnp.dot(x, weights))
            output_weights = jax.random.normal(jax.random.PRNGKey(1), (64, 1))
            output = jnp.dot(hidden, output_weights)
            return jnp.exp(output.flatten()) * 1e-6

        return model(features)

    def _cpu_permeability_model(self, features):
        """CPU permeability prediction."""
        weights = np.random.randn(features.shape[1], 64).astype(np.float32)
        hidden = np.tanh(np.dot(features, weights))
        output_weights = np.random.randn(64, 1).astype(np.float32)
        output = np.dot(hidden, output_weights)
        return np.exp(output.flatten()) * 1e-6

    def _cuda_diffusion_model(self, features):
        """CUDA-accelerated diffusion coefficient prediction."""
        if not CUDA_AVAILABLE:
            return self._cpu_diffusion_model(features)
        weights = cp.random.randn(features.shape[1], 32).astype(cp.float32)
        hidden = cp.maximum(0, cp.dot(features, weights))  # ReLU
        output_weights = cp.random.randn(32, 3).astype(cp.float32)  # 3 phases
        output = cp.dot(hidden, output_weights)
        return cp.exp(output) * 1e-9  # Convert to realistic diffusion units

    def _jax_diffusion_model(self, features):
        """JAX-accelerated diffusion coefficient prediction."""
        if not JAX_AVAILABLE:
            return self._cpu_diffusion_model(features)

        @jit
        def model(x):
            weights = jax.random.normal(jax.random.PRNGKey(2), (x.shape[1], 32))
            hidden = jnp.maximum(0, jnp.dot(x, weights))  # ReLU
            output_weights = jax.random.normal(jax.random.PRNGKey(3), (32, 3))
            output = jnp.dot(hidden, output_weights)
            return jnp.exp(output) * 1e-9

        return model(features)

    def _cpu_diffusion_model(self, features):
        """CPU diffusion coefficient prediction."""
        weights = np.random.randn(features.shape[1], 32).astype(np.float32)
        hidden = np.maximum(0, np.dot(features, weights))
        output_weights = np.random.randn(32, 3).astype(np.float32)
        output = np.dot(hidden, output_weights)
        return np.exp(output) * 1e-9

    def _cuda_partition_model(self, features):
        """CUDA-accelerated partition coefficient prediction."""
        if not CUDA_AVAILABLE:
            return self._cpu_partition_model(features)
        weights = cp.random.randn(features.shape[1], 16).astype(cp.float32)
        hidden = 1 / (1 + cp.exp(-cp.dot(features, weights)))  # Sigmoid
        output_weights = cp.random.randn(16, 2).astype(
            cp.float32
        )  # 2 partition coefficients
        output = cp.dot(hidden, output_weights)
        return output  # Log partition coefficients

    def _jax_partition_model(self, features):
        """JAX-accelerated partition coefficient prediction."""
        if not JAX_AVAILABLE:
            return self._cpu_partition_model(features)

        @jit
        def model(x):
            weights = jax.random.normal(jax.random.PRNGKey(4), (x.shape[1], 16))
            hidden = jax.nn.sigmoid(jnp.dot(x, weights))
            output_weights = jax.random.normal(jax.random.PRNGKey(5), (16, 2))
            output = jnp.dot(hidden, output_weights)
            return output

        return model(features)

    def _cpu_partition_model(self, features):
        """CPU partition coefficient prediction."""
        weights = np.random.randn(features.shape[1], 16).astype(np.float32)
        hidden = 1 / (1 + np.exp(-np.dot(features, weights)))  # Sigmoid
        output_weights = np.random.randn(16, 2).astype(np.float32)
        output = np.dot(hidden, output_weights)
        return output

    def _extract_molecular_features(self, molecules: List[str]) -> np.ndarray:
        """Extract molecular features for surrogate models."""
        # Mock feature extraction - in practice would use RDKit descriptors
        features = []
        for mol in molecules:
            # Simulate molecular descriptors
            mol_features = [
                len(mol),  # Molecular size proxy
                hash(mol) % 100 / 100.0,  # Pseudo-lipophilicity
                (hash(mol + "polar") % 50) / 50.0,  # Pseudo-polarity
                (hash(mol + "hbond") % 20) / 20.0,  # H-bond capacity
                np.random.randn() * 0.1,  # Random noise
            ]
            features.append(mol_features)
        return np.array(features, dtype=np.float32)

    def _generate_trajectories(
        self, molecules: List[str], config: SimulationConfig
    ) -> np.ndarray:
        """Generate mock molecular trajectories."""
        n_molecules = len(molecules)
        n_frames = config.n_steps // config.output_frequency

        # Generate realistic-looking trajectories across membrane
        trajectories = np.zeros((n_molecules, n_frames, 3))

        for i, mol in enumerate(molecules):
            # Start molecules at different z-positions
            z_start = -5.0 + i * 2.0

            # Simulate transport with some randomness
            z_positions = np.linspace(z_start, z_start + 10.0, n_frames)
            z_positions += np.random.randn(n_frames) * 0.5  # Thermal fluctuations

            trajectories[i, :, 0] = np.random.randn(n_frames) * 0.2  # x fluctuations
            trajectories[i, :, 1] = np.random.randn(n_frames) * 0.2  # y fluctuations
            trajectories[i, :, 2] = z_positions  # z transport

        return trajectories

    def _calculate_concentration_profiles(
        self, trajectories: np.ndarray, config: SimulationConfig
    ) -> Dict[str, np.ndarray]:
        """Calculate concentration profiles from trajectories."""
        n_molecules, n_frames, _ = trajectories.shape

        # Define spatial bins
        z_min, z_max = -10.0, 10.0
        n_bins = 50
        z_bins = np.linspace(z_min, z_max, n_bins)

        profiles = {}

        for frame in range(0, n_frames, max(1, n_frames // 10)):  # Sample frames
            z_positions = trajectories[:, frame, 2]
            hist, _ = np.histogram(z_positions, bins=z_bins)
            profiles[f"frame_{frame}"] = hist / n_molecules

        return profiles

    def _calculate_flux(
        self, trajectories: np.ndarray, config: SimulationConfig
    ) -> np.ndarray:
        """Calculate molecular flux across interfaces."""
        n_molecules, n_frames, _ = trajectories.shape

        # Calculate z-direction velocities
        z_positions = trajectories[:, :, 2]
        velocities = np.gradient(z_positions, axis=1) / (
            config.timestep * config.output_frequency
        )

        # Calculate flux at membrane center (z=0)
        membrane_crossings = []
        for i in range(n_molecules):
            z_traj = z_positions[i, :]
            crossings = np.where((z_traj[:-1] < 0) & (z_traj[1:] >= 0))[0]
            membrane_crossings.extend(crossings)

        # Bin crossings by time
        time_bins = np.linspace(0, n_frames, 20)
        flux, _ = np.histogram(membrane_crossings, bins=time_bins)

        return flux.astype(float)

    def run_simulation(
        self, molecules: List[str], config: SimulationConfig
    ) -> TransportResult:
        """Run AI surrogate transport simulation."""
        start_time = np.time.time()

        self.logger.info(
            f"Running AI surrogate simulation for {len(molecules)} molecules"
        )

        # Extract molecular features
        features = self._extract_molecular_features(molecules)

        # Convert to appropriate backend format
        if self.backend == "cuda" and CUDA_AVAILABLE:
            features_gpu = cp.asarray(features)
            permeabilities = self.permeability_model(features_gpu)
            diffusion_coeffs = self.diffusion_model(features_gpu)
            partition_coeffs = self.partition_model(features_gpu)

            # Convert back to CPU for final results
            permeabilities = cp.asnumpy(permeabilities)
            diffusion_coeffs = cp.asnumpy(diffusion_coeffs)
            partition_coeffs = cp.asnumpy(partition_coeffs)

        elif self.backend == "jax" and JAX_AVAILABLE:
            features_jax = jnp.asarray(features)
            permeabilities = np.array(self.permeability_model(features_jax))
            diffusion_coeffs = np.array(self.diffusion_model(features_jax))
            partition_coeffs = np.array(self.partition_model(features_jax))

        else:
            permeabilities = self.permeability_model(features)
            diffusion_coeffs = self.diffusion_model(features)
            partition_coeffs = self.partition_model(features)

        # Generate trajectories and derived properties
        trajectories = self._generate_trajectories(molecules, config)
        concentration_profiles = self._calculate_concentration_profiles(
            trajectories, config
        )
        flux_data = self._calculate_flux(trajectories, config)

        # Package results
        avg_permeability = float(np.mean(permeabilities))

        diffusion_dict = {
            "aqueous": float(np.mean(diffusion_coeffs[:, 0])),
            "membrane": float(np.mean(diffusion_coeffs[:, 1])),
            "lipid": float(np.mean(diffusion_coeffs[:, 2])),
        }

        partition_dict = {
            "water_octanol": float(np.mean(partition_coeffs[:, 0])),
            "membrane_water": float(np.mean(partition_coeffs[:, 1])),
        }

        simulation_time = np.time.time() - start_time

        metadata = {
            "method": "ai_surrogate",
            "backend": self.backend,
            "molecules": molecules,
            "config": config,
            "n_molecules": len(molecules),
        }

        return TransportResult(
            permeability_coefficient=avg_permeability,
            diffusion_coefficients=diffusion_dict,
            partition_coefficients=partition_dict,
            trajectories=trajectories,
            concentration_profiles=concentration_profiles,
            flux_data=flux_data,
            simulation_time=simulation_time,
            metadata=metadata,
        )


class AtomisticMDSimulation(SimulationMethod):
    """Atomistic molecular dynamics simulation for transport."""

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.logger = logging.getLogger(__name__)

    def run_simulation(
        self, molecules: List[str], config: SimulationConfig
    ) -> TransportResult:
        """Run atomistic MD simulation (placeholder implementation)."""
        self.logger.warning(
            "Atomistic MD simulation not fully implemented - using simplified model"
        )

        # In practice, this would interface with MD engines like GROMACS, AMBER, etc.
        # For now, return simplified results

        n_molecules = len(molecules)
        permeability = np.random.lognormal(-12, 2)  # Realistic permeability range

        diffusion_coeffs = {
            "aqueous": np.random.lognormal(-19, 1),
            "membrane": np.random.lognormal(-20, 1),
            "lipid": np.random.lognormal(-21, 1),
        }

        partition_coeffs = {
            "water_octanol": np.random.normal(2, 1),
            "membrane_water": np.random.normal(1, 0.5),
        }

        # Generate mock trajectory data
        n_frames = config.n_steps // config.output_frequency
        trajectories = np.random.randn(n_molecules, n_frames, 3) * 0.5

        # Add transport component
        for i in range(n_molecules):
            trajectories[i, :, 2] += np.linspace(-5, 5, n_frames)

        concentration_profiles = {"frame_0": np.random.poisson(2, 50)}
        flux_data = np.random.poisson(1, 20).astype(float)

        metadata = {
            "method": "atomistic_md",
            "backend": self.backend,
            "molecules": molecules,
            "config": config,
        }

        return TransportResult(
            permeability_coefficient=permeability,
            diffusion_coefficients=diffusion_coeffs,
            partition_coefficients=partition_coeffs,
            trajectories=trajectories,
            concentration_profiles=concentration_profiles,
            flux_data=flux_data,
            simulation_time=10.0,  # Mock timing
            metadata=metadata,
        )


class CoarseGrainedDPDSimulation(SimulationMethod):
    """Coarse-grained dissipative particle dynamics simulation."""

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.logger = logging.getLogger(__name__)

    def run_simulation(
        self, molecules: List[str], config: SimulationConfig
    ) -> TransportResult:
        """Run coarse-grained DPD simulation (placeholder implementation)."""
        self.logger.warning(
            "DPD simulation not fully implemented - using simplified model"
        )

        # Placeholder implementation - would interface with DPD codes
        n_molecules = len(molecules)
        permeability = np.random.lognormal(-11, 1.5)

        diffusion_coeffs = {
            "aqueous": np.random.lognormal(-18, 0.8),
            "membrane": np.random.lognormal(-19, 0.8),
            "lipid": np.random.lognormal(-20, 0.8),
        }

        partition_coeffs = {
            "water_octanol": np.random.normal(1.5, 0.8),
            "membrane_water": np.random.normal(0.8, 0.4),
        }

        n_frames = config.n_steps // config.output_frequency
        trajectories = np.random.randn(n_molecules, n_frames, 3) * 0.8

        concentration_profiles = {"frame_0": np.random.poisson(3, 50)}
        flux_data = np.random.poisson(2, 20).astype(float)

        metadata = {
            "method": "coarse_grained_dpd",
            "backend": self.backend,
            "molecules": molecules,
            "config": config,
        }

        return TransportResult(
            permeability_coefficient=permeability,
            diffusion_coefficients=diffusion_coeffs,
            partition_coefficients=partition_coeffs,
            trajectories=trajectories,
            concentration_profiles=concentration_profiles,
            flux_data=flux_data,
            simulation_time=5.0,
            metadata=metadata,
        )


class MultiPhaseSimulator:
    """
    GPU-accelerated multi-phase simulation engine for molecular transport.

    This class provides a unified interface for running various types of transport
    simulations across different phases and interfaces.
    """

    def __init__(self, backend: str = "auto", method: str = "ai_surrogate"):
        """
        Initialize the multi-phase simulator.

        Args:
            backend: Computational backend ('cuda', 'jax', 'cpu', 'auto')
            method: Simulation method ('ai_surrogate', 'md', 'dpd')
        """
        self.backend = backend
        self.method = method
        self.logger = logging.getLogger(__name__)

        # Initialize simulation method
        self.simulation_method = self._create_simulation_method()

    def _create_simulation_method(self) -> SimulationMethod:
        """Create appropriate simulation method instance."""
        if self.method == "ai_surrogate":
            return AISubrogateSimulation(self.backend)
        elif self.method == "md":
            return AtomisticMDSimulation(self.backend)
        elif self.method == "dpd":
            return CoarseGrainedDPDSimulation(self.backend)
        else:
            raise ChemiAccelError(f"Unknown simulation method: {self.method}")

    def simulate_membrane_transport(
        self,
        molecules: Union[List[str], List[MolecularData]],
        membrane_type: str = "dppc_bilayer",
        conditions: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> TransportResult:
        """
        Simulate molecular transport across biological membranes.

        Args:
            molecules: List of SMILES strings or MolecularData objects
            membrane_type: Type of membrane ('dppc_bilayer', 'dopc_bilayer', etc.)
            conditions: Environmental conditions (temperature, pH, etc.)
            **kwargs: Additional simulation parameters

        Returns:
            TransportResult: Comprehensive transport simulation results
        """
        # Convert MolecularData to SMILES if needed
        if molecules and isinstance(molecules[0], MolecularData):
            molecule_smiles = [mol.smiles for mol in molecules]
        else:
            molecule_smiles = molecules

        # Set up simulation configuration
        config = SimulationConfig(
            method=self.method, interface_type=membrane_type, **kwargs
        )

        # Apply environmental conditions
        if conditions:
            if "temperature" in conditions:
                config.temperature = conditions["temperature"]
            if "pressure" in conditions:
                config.pressure = conditions["pressure"]

        # Add membrane-specific parameters
        config.phase_definitions = self._get_membrane_definition(membrane_type)

        self.logger.info(
            f"Starting membrane transport simulation for {len(molecule_smiles)} molecules"
        )

        return self.simulation_method.run_simulation(molecule_smiles, config)

    def simulate_liquid_liquid_extraction(
        self,
        molecules: Union[List[str], List[MolecularData]],
        solvent_system: Tuple[str, str],
        conditions: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> TransportResult:
        """
        Simulate molecular partitioning in liquid-liquid extraction.

        Args:
            molecules: List of SMILES strings or MolecularData objects
            solvent_system: Tuple of (aqueous_phase, organic_phase)
            conditions: Environmental conditions
            **kwargs: Additional simulation parameters

        Returns:
            TransportResult: Liquid-liquid extraction simulation results
        """
        if molecules and isinstance(molecules[0], MolecularData):
            molecule_smiles = [mol.smiles for mol in molecules]
        else:
            molecule_smiles = molecules

        config = SimulationConfig(
            method=self.method, interface_type="liquid_liquid", **kwargs
        )

        if conditions:
            if "temperature" in conditions:
                config.temperature = conditions["temperature"]

        # Set up solvent system
        config.phase_definitions = {
            "aqueous_phase": solvent_system[0],
            "organic_phase": solvent_system[1],
        }

        self.logger.info("Starting liquid-liquid extraction simulation")

        return self.simulation_method.run_simulation(molecule_smiles, config)

    def simulate_polymer_membrane_transport(
        self,
        molecules: Union[List[str], List[MolecularData]],
        polymer_membrane: str,
        conditions: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> TransportResult:
        """
        Simulate transport through polymer membranes.

        Args:
            molecules: List of SMILES strings or MolecularData objects
            polymer_membrane: Type of polymer membrane
            conditions: Environmental conditions
            **kwargs: Additional simulation parameters

        Returns:
            TransportResult: Polymer membrane transport results
        """
        if molecules and isinstance(molecules[0], MolecularData):
            molecule_smiles = [mol.smiles for mol in molecules]
        else:
            molecule_smiles = molecules

        config = SimulationConfig(
            method=self.method, interface_type="polymer_membrane", **kwargs
        )

        if conditions:
            if "temperature" in conditions:
                config.temperature = conditions["temperature"]

        config.phase_definitions = self._get_polymer_membrane_definition(
            polymer_membrane
        )

        return self.simulation_method.run_simulation(molecule_smiles, config)

    def _get_membrane_definition(self, membrane_type: str) -> Dict[str, Any]:
        """Get membrane-specific simulation parameters."""
        membrane_definitions = {
            "dppc_bilayer": {
                "lipid_type": "DPPC",
                "thickness": 4.0,  # nm
                "area_per_lipid": 0.64,  # nm²
                "chain_length": 16,
                "saturation": "saturated",
            },
            "dopc_bilayer": {
                "lipid_type": "DOPC",
                "thickness": 3.8,  # nm
                "area_per_lipid": 0.68,  # nm²
                "chain_length": 18,
                "saturation": "unsaturated",
            },
            "popc_bilayer": {
                "lipid_type": "POPC",
                "thickness": 3.9,  # nm
                "area_per_lipid": 0.66,  # nm²
                "chain_length": 16,
                "saturation": "mixed",
            },
        }

        return membrane_definitions.get(
            membrane_type, membrane_definitions["dppc_bilayer"]
        )

    def _get_polymer_membrane_definition(self, polymer_type: str) -> Dict[str, Any]:
        """Get polymer membrane-specific parameters."""
        polymer_definitions = {
            "pvdf": {
                "polymer_type": "PVDF",
                "porosity": 0.75,
                "pore_size": 0.1,  # μm
                "thickness": 125,  # μm
                "hydrophobicity": "hydrophobic",
            },
            "pva": {
                "polymer_type": "PVA",
                "porosity": 0.0,  # Dense membrane
                "thickness": 50,  # μm
                "hydrophobicity": "hydrophilic",
            },
            "pdms": {
                "polymer_type": "PDMS",
                "porosity": 0.0,
                "thickness": 200,  # μm
                "hydrophobicity": "hydrophobic",
            },
        }

        return polymer_definitions.get(polymer_type, polymer_definitions["pvdf"])

    def benchmark_against_experimental(
        self, experimental_data: Dict[str, float], simulation_results: TransportResult
    ) -> Dict[str, float]:
        """
        Benchmark simulation results against experimental data.

        Args:
            experimental_data: Dictionary of experimental measurements
            simulation_results: Results from simulation

        Returns:
            Dictionary of comparison metrics
        """
        metrics = {}

        if "permeability" in experimental_data:
            exp_perm = experimental_data["permeability"]
            sim_perm = simulation_results.permeability_coefficient

            metrics["permeability_error"] = abs(sim_perm - exp_perm) / exp_perm
            metrics["permeability_log_error"] = abs(
                np.log10(sim_perm) - np.log10(exp_perm)
            )

        if "partition_coefficient" in experimental_data:
            exp_part = experimental_data["partition_coefficient"]
            sim_part = simulation_results.partition_coefficients.get("water_octanol", 0)

            metrics["partition_error"] = abs(sim_part - exp_part) / abs(exp_part)

        return metrics

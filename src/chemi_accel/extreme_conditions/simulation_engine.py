"""
GPU-Accelerated Non-Equilibrium Simulation Engine

This module implements GPU-accelerated simulation capabilities for non-equilibrium
molecular dynamics (NEMD), reactive force fields, and specialized multi-scale models
to simulate material behavior under extreme conditions.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cp_sparse = None
    CUPY_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from ..core.exceptions import BackendError, ChemiAccelError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ExtremeConditionType(Enum):
    """Types of extreme conditions that can be simulated."""

    HIGH_STRAIN_RATE = "high_strain_rate"
    RADIATION_DAMAGE = "radiation_damage"
    HIGH_TEMPERATURE = "high_temperature"
    HIGH_PRESSURE = "high_pressure"
    SHOCK_LOADING = "shock_loading"
    THERMAL_SHOCK = "thermal_shock"
    CORROSIVE_ENVIRONMENT = "corrosive_environment"
    EXTREME_PH = "extreme_ph"
    RAPID_COOLING = "rapid_cooling"
    FATIGUE_LOADING = "fatigue_loading"


class SimulationBackend(Enum):
    """Available simulation backends."""

    CUDA = "cuda"
    JAX = "jax"
    CPU = "cpu"


@dataclass
class SimulationParameters:
    """Parameters for non-equilibrium simulations."""

    # Time parameters
    time_step: float = 1e-15  # femtoseconds
    total_time: float = 1e-12  # picoseconds
    output_frequency: int = 100

    # Temperature and pressure
    temperature: float = 300.0  # Kelvin
    pressure: float = 1.01325e5  # Pa (1 atm)

    # Extreme condition parameters
    condition_type: ExtremeConditionType = ExtremeConditionType.HIGH_STRAIN_RATE
    strain_rate: float = 1e9  # s^-1
    radiation_dose: float = 0.0  # Gray/s
    shock_velocity: float = 0.0  # m/s

    # Force field parameters
    cutoff_radius: float = 12.0  # Angstroms
    neighbor_list_skin: float = 2.0  # Angstroms

    # Integration parameters
    integrator: str = "velocity_verlet"
    thermostat: str = "nose_hoover"
    barostat: str = "parrinello_rahman"

    # Output parameters
    calculate_stress: bool = True
    calculate_pressure: bool = True
    calculate_damage: bool = True
    track_defects: bool = True


@dataclass
class SimulationResults:
    """Results from non-equilibrium simulations."""

    # Trajectory data
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    forces: np.ndarray = field(default_factory=lambda: np.array([]))

    # Thermodynamic properties
    temperatures: np.ndarray = field(default_factory=lambda: np.array([]))
    pressures: np.ndarray = field(default_factory=lambda: np.array([]))
    energies: np.ndarray = field(default_factory=lambda: np.array([]))

    # Stress and strain
    stress_tensor: np.ndarray = field(default_factory=lambda: np.array([]))
    strain_tensor: np.ndarray = field(default_factory=lambda: np.array([]))

    # Damage and failure metrics
    damage_fraction: np.ndarray = field(default_factory=lambda: np.array([]))
    defect_count: np.ndarray = field(default_factory=lambda: np.array([]))
    crack_length: np.ndarray = field(default_factory=lambda: np.array([]))

    # Timing and metadata
    computation_time: float = 0.0
    simulation_metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSimulator(ABC):
    """Base class for all extreme condition simulators."""

    def __init__(self, backend: SimulationBackend = SimulationBackend.CUDA):
        self.backend = backend
        self.logger = get_logger(self.__class__.__name__)
        self._validate_backend()

    def _validate_backend(self):
        """Validate that the requested backend is available."""
        if self.backend == SimulationBackend.CUDA and not CUPY_AVAILABLE:
            raise BackendError("CUDA backend requested but CuPy not available")
        elif self.backend == SimulationBackend.JAX and not JAX_AVAILABLE:
            raise BackendError("JAX backend requested but JAX not available")

    @abstractmethod
    def run_simulation(
        self, initial_structure: Dict[str, Any], parameters: SimulationParameters
    ) -> SimulationResults:
        """Run the simulation with given parameters."""
        pass

    @abstractmethod
    def calculate_properties(self, results: SimulationResults) -> Dict[str, float]:
        """Calculate material properties from simulation results."""
        pass


class NEMDSimulator(BaseSimulator):
    """Non-Equilibrium Molecular Dynamics simulator for extreme conditions."""

    def __init__(self, backend: SimulationBackend = SimulationBackend.CUDA):
        super().__init__(backend)
        self.force_calculator = None
        self._setup_force_calculator()

    def _setup_force_calculator(self):
        """Initialize the force calculation method based on backend."""
        if self.backend == SimulationBackend.CUDA:
            self.force_calculator = self._cuda_force_calculation
        elif self.backend == SimulationBackend.JAX:
            self.force_calculator = jit(self._jax_force_calculation)
        else:
            self.force_calculator = self._cpu_force_calculation

    def run_simulation(
        self, initial_structure: Dict[str, Any], parameters: SimulationParameters
    ) -> SimulationResults:
        """Run NEMD simulation under extreme conditions."""

        start_time = time.time()
        self.logger.info(
            f"Starting NEMD simulation for {parameters.condition_type.value}"
        )

        # Extract initial conditions
        positions = np.array(initial_structure["positions"])
        velocities = np.array(
            initial_structure.get("velocities", np.random.normal(0, 1, positions.shape))
        )
        masses = np.array(initial_structure.get("masses", np.ones(positions.shape[0])))

        # Initialize trajectory storage
        n_atoms = positions.shape[0]
        n_steps = int(parameters.total_time / parameters.time_step)
        output_steps = n_steps // parameters.output_frequency

        trajectory_positions = np.zeros((output_steps, n_atoms, 3))
        trajectory_velocities = np.zeros((output_steps, n_atoms, 3))
        trajectory_forces = np.zeros((output_steps, n_atoms, 3))

        # Thermodynamic properties
        temperatures = np.zeros(output_steps)
        pressures = np.zeros(output_steps)
        energies = np.zeros(output_steps)

        # Stress/strain tensors
        stress_tensors = np.zeros((output_steps, 3, 3))
        strain_tensors = np.zeros((output_steps, 3, 3))

        # Damage metrics
        damage_fractions = np.zeros(output_steps)
        defect_counts = np.zeros(output_steps)
        crack_lengths = np.zeros(output_steps)

        # Move arrays to GPU if using CUDA
        if self.backend == SimulationBackend.CUDA:
            positions = cp.asarray(positions)
            velocities = cp.asarray(velocities)
            masses = cp.asarray(masses)

        # Main simulation loop
        output_counter = 0
        for step in range(n_steps):
            # Apply extreme condition modifications
            self._apply_extreme_conditions(positions, velocities, step, parameters)

            # Calculate forces
            forces = self.force_calculator(positions, parameters)

            # Velocity Verlet integration
            velocities += 0.5 * parameters.time_step * forces / masses[:, np.newaxis]
            positions += parameters.time_step * velocities

            # Update forces with new positions
            forces = self.force_calculator(positions, parameters)
            velocities += 0.5 * parameters.time_step * forces / masses[:, np.newaxis]

            # Apply thermostat and barostat
            self._apply_thermostat(velocities, parameters)
            self._apply_barostat(positions, parameters)

            # Store trajectory data
            if step % parameters.output_frequency == 0:
                if self.backend == SimulationBackend.CUDA:
                    trajectory_positions[output_counter] = cp.asnumpy(positions)
                    trajectory_velocities[output_counter] = cp.asnumpy(velocities)
                    trajectory_forces[output_counter] = cp.asnumpy(forces)
                else:
                    trajectory_positions[output_counter] = positions.copy()
                    trajectory_velocities[output_counter] = velocities.copy()
                    trajectory_forces[output_counter] = forces.copy()

                # Calculate properties
                temperatures[output_counter] = self._calculate_temperature(
                    velocities, masses
                )
                pressures[output_counter] = self._calculate_pressure(
                    positions, forces, parameters
                )
                energies[output_counter] = self._calculate_energy(
                    positions, velocities, masses, parameters
                )

                # Calculate stress and strain
                stress_tensors[output_counter] = self._calculate_stress_tensor(
                    positions, forces, parameters
                )
                strain_tensors[output_counter] = self._calculate_strain_tensor(
                    positions, step, parameters
                )

                # Calculate damage metrics
                damage_fractions[output_counter] = self._calculate_damage_fraction(
                    positions, forces, parameters
                )
                defect_counts[output_counter] = self._count_defects(
                    positions, parameters
                )
                crack_lengths[output_counter] = self._measure_crack_length(
                    positions, parameters
                )

                output_counter += 1

                if output_counter % 100 == 0:
                    self.logger.info(
                        f"Completed {output_counter}/{output_steps} output steps"
                    )

        # Create results object
        results = SimulationResults(
            positions=trajectory_positions,
            velocities=trajectory_velocities,
            forces=trajectory_forces,
            temperatures=temperatures,
            pressures=pressures,
            energies=energies,
            stress_tensor=stress_tensors,
            strain_tensor=strain_tensors,
            damage_fraction=damage_fractions,
            defect_count=defect_counts,
            crack_length=crack_lengths,
            computation_time=time.time() - start_time,
            simulation_metadata={
                "backend": self.backend.value,
                "condition_type": parameters.condition_type.value,
                "n_atoms": n_atoms,
                "n_steps": n_steps,
            },
        )

        self.logger.info(
            f"NEMD simulation completed in {results.computation_time:.2f} seconds"
        )
        return results

    def _cuda_force_calculation(
        self, positions: cp.ndarray, parameters: SimulationParameters
    ) -> cp.ndarray:
        """CUDA-accelerated force calculation."""
        n_atoms = positions.shape[0]
        forces = cp.zeros_like(positions)

        # Lennard-Jones potential with GPU acceleration
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = positions[j] - positions[i]
                r = cp.linalg.norm(r_vec)

                if r < parameters.cutoff_radius:
                    # Lennard-Jones force
                    sigma = 3.4  # Angstroms
                    epsilon = 0.238  # kcal/mol

                    r6 = (sigma / r) ** 6
                    r12 = r6**2

                    f_magnitude = 24 * epsilon * (2 * r12 - r6) / (r**2)
                    f_vec = f_magnitude * r_vec / r

                    forces[i] += f_vec
                    forces[j] -= f_vec

        return forces

    def _jax_force_calculation(
        self, positions: jnp.ndarray, parameters: SimulationParameters
    ) -> jnp.ndarray:
        """JAX-accelerated force calculation with automatic differentiation."""

        def potential_energy(pos):
            n_atoms = pos.shape[0]
            energy = 0.0

            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r_vec = pos[j] - pos[i]
                    r = jnp.linalg.norm(r_vec)

                    # Lennard-Jones potential
                    sigma = 3.4
                    epsilon = 0.238

                    if r < parameters.cutoff_radius:
                        r6 = (sigma / r) ** 6
                        r12 = r6**2
                        energy += 4 * epsilon * (r12 - r6)

            return energy

        # Use automatic differentiation to get forces
        grad_fn = grad(potential_energy)
        return -grad_fn(positions)

    def _cpu_force_calculation(
        self, positions: np.ndarray, parameters: SimulationParameters
    ) -> np.ndarray:
        """CPU-based force calculation."""
        n_atoms = positions.shape[0]
        forces = np.zeros_like(positions)

        # Simple Lennard-Jones potential
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)

                if r < parameters.cutoff_radius:
                    sigma = 3.4
                    epsilon = 0.238

                    r6 = (sigma / r) ** 6
                    r12 = r6**2

                    f_magnitude = 24 * epsilon * (2 * r12 - r6) / (r**2)
                    f_vec = f_magnitude * r_vec / r

                    forces[i] += f_vec
                    forces[j] -= f_vec

        return forces

    def _apply_extreme_conditions(
        self,
        positions: Union[np.ndarray, cp.ndarray],
        velocities: Union[np.ndarray, cp.ndarray],
        step: int,
        parameters: SimulationParameters,
    ):
        """Apply extreme condition modifications to the system."""

        if parameters.condition_type == ExtremeConditionType.HIGH_STRAIN_RATE:
            # Apply high strain rate deformation
            strain = parameters.strain_rate * parameters.time_step * step
            deformation_matrix = np.array(
                [[1 + strain, 0, 0], [0, 1 - 0.5 * strain, 0], [0, 0, 1 - 0.5 * strain]]
            )

            if self.backend == SimulationBackend.CUDA:
                deformation_matrix = cp.asarray(deformation_matrix)

            positions[:] = positions @ deformation_matrix.T

        elif parameters.condition_type == ExtremeConditionType.RADIATION_DAMAGE:
            # Randomly displace atoms to simulate radiation damage
            if step % 1000 == 0:  # Apply damage periodically
                damage_probability = parameters.radiation_dose * 1e-10
                n_atoms = positions.shape[0]

                if self.backend == SimulationBackend.CUDA:
                    random_mask = cp.random.random(n_atoms) < damage_probability
                    displacement = cp.random.normal(0, 0.5, (n_atoms, 3))
                else:
                    random_mask = np.random.random(n_atoms) < damage_probability
                    displacement = np.random.normal(0, 0.5, (n_atoms, 3))

                positions[random_mask] += displacement[random_mask]

        elif parameters.condition_type == ExtremeConditionType.SHOCK_LOADING:
            # Apply shock wave propagation
            shock_front_position = (
                parameters.shock_velocity * parameters.time_step * step
            )

            # Apply velocity kick to atoms behind shock front
            shock_mask = positions[:, 0] < shock_front_position
            if self.backend == SimulationBackend.CUDA:
                velocities[shock_mask, 0] += cp.random.normal(
                    0, 100, cp.sum(shock_mask)
                )
            else:
                velocities[shock_mask, 0] += np.random.normal(
                    0, 100, np.sum(shock_mask)
                )

    def _apply_thermostat(
        self,
        velocities: Union[np.ndarray, cp.ndarray],
        parameters: SimulationParameters,
    ):
        """Apply thermostat to maintain temperature."""
        current_temp = self._calculate_temperature(velocities, None)
        scale_factor = np.sqrt(parameters.temperature / current_temp)

        if self.backend == SimulationBackend.CUDA:
            scale_factor = cp.asarray(scale_factor)

        velocities *= scale_factor

    def _apply_barostat(
        self, positions: Union[np.ndarray, cp.ndarray], parameters: SimulationParameters
    ):
        """Apply barostat to maintain pressure."""
        # Simple volume scaling barostat
        current_pressure = 1e5  # Placeholder
        pressure_ratio = parameters.pressure / current_pressure
        scale_factor = pressure_ratio ** (1 / 3)

        if self.backend == SimulationBackend.CUDA:
            scale_factor = cp.asarray(scale_factor)

        positions *= scale_factor

    def _calculate_temperature(
        self,
        velocities: Union[np.ndarray, cp.ndarray],
        masses: Optional[Union[np.ndarray, cp.ndarray]],
    ) -> float:
        """Calculate instantaneous temperature."""
        if masses is None:
            if self.backend == SimulationBackend.CUDA:
                masses = cp.ones(velocities.shape[0])
            else:
                masses = np.ones(velocities.shape[0])

        kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
        n_dof = 3 * velocities.shape[0] - 3  # Remove center of mass motion
        kb = 8.314e-3  # kJ/(mol·K)

        if self.backend == SimulationBackend.CUDA:
            kinetic_energy = cp.asnumpy(kinetic_energy)

        return 2 * kinetic_energy / (n_dof * kb)

    def _calculate_pressure(
        self,
        positions: Union[np.ndarray, cp.ndarray],
        forces: Union[np.ndarray, cp.ndarray],
        parameters: SimulationParameters,
    ) -> float:
        """Calculate instantaneous pressure using virial theorem."""
        # Simplified pressure calculation
        if self.backend == SimulationBackend.CUDA:
            virial = cp.sum(positions * forces)
            volume = cp.prod(cp.max(positions, axis=0) - cp.min(positions, axis=0))
            virial = cp.asnumpy(virial)
            volume = cp.asnumpy(volume)
        else:
            virial = np.sum(positions * forces)
            volume = np.prod(np.max(positions, axis=0) - np.min(positions, axis=0))

        return virial / (3 * volume)

    def _calculate_energy(
        self,
        positions: Union[np.ndarray, cp.ndarray],
        velocities: Union[np.ndarray, cp.ndarray],
        masses: Union[np.ndarray, cp.ndarray],
        parameters: SimulationParameters,
    ) -> float:
        """Calculate total energy."""
        kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
        potential = 0.0  # Placeholder for potential energy calculation

        if self.backend == SimulationBackend.CUDA:
            kinetic = cp.asnumpy(kinetic)

        return kinetic + potential

    def _calculate_stress_tensor(
        self,
        positions: Union[np.ndarray, cp.ndarray],
        forces: Union[np.ndarray, cp.ndarray],
        parameters: SimulationParameters,
    ) -> np.ndarray:
        """Calculate stress tensor."""
        stress = np.zeros((3, 3))

        # Virial stress calculation
        for i in range(positions.shape[0]):
            stress += np.outer(positions[i], forces[i])

        if self.backend == SimulationBackend.CUDA:
            stress = cp.asnumpy(stress)

        # Normalize by volume
        volume = 1000.0  # Placeholder
        return stress / volume

    def _calculate_strain_tensor(
        self,
        positions: Union[np.ndarray, cp.ndarray],
        step: int,
        parameters: SimulationParameters,
    ) -> np.ndarray:
        """Calculate strain tensor."""
        strain = np.zeros((3, 3))

        if parameters.condition_type == ExtremeConditionType.HIGH_STRAIN_RATE:
            total_strain = parameters.strain_rate * parameters.time_step * step
            strain[0, 0] = total_strain
            strain[1, 1] = -0.5 * total_strain
            strain[2, 2] = -0.5 * total_strain

        return strain

    def _calculate_damage_fraction(
        self,
        positions: Union[np.ndarray, cp.ndarray],
        forces: Union[np.ndarray, cp.ndarray],
        parameters: SimulationParameters,
    ) -> float:
        """Calculate fraction of damaged material."""
        # Simplified damage model based on force magnitude
        if self.backend == SimulationBackend.CUDA:
            force_magnitudes = cp.linalg.norm(forces, axis=1)
            damage_threshold = 10.0
            damaged_atoms = cp.sum(force_magnitudes > damage_threshold)
            return cp.asnumpy(damaged_atoms) / positions.shape[0]
        else:
            force_magnitudes = np.linalg.norm(forces, axis=1)
            damage_threshold = 10.0
            damaged_atoms = np.sum(force_magnitudes > damage_threshold)
            return damaged_atoms / positions.shape[0]

    def _count_defects(
        self, positions: Union[np.ndarray, cp.ndarray], parameters: SimulationParameters
    ) -> int:
        """Count crystal defects."""
        # Simplified defect counting based on nearest neighbor analysis
        return 0  # Placeholder

    def _measure_crack_length(
        self, positions: Union[np.ndarray, cp.ndarray], parameters: SimulationParameters
    ) -> float:
        """Measure total crack length."""
        # Simplified crack detection
        return 0.0  # Placeholder

    def calculate_properties(self, results: SimulationResults) -> Dict[str, float]:
        """Calculate material properties from simulation results."""
        properties = {}

        # Average properties
        properties["avg_temperature"] = np.mean(results.temperatures)
        properties["avg_pressure"] = np.mean(results.pressures)
        properties["avg_energy"] = np.mean(results.energies)

        # Final damage state
        properties["final_damage_fraction"] = results.damage_fraction[-1]
        properties["final_defect_count"] = results.defect_count[-1]
        properties["final_crack_length"] = results.crack_length[-1]

        # Stress/strain properties
        final_stress = results.stress_tensor[-1]
        final_strain = results.strain_tensor[-1]

        properties["von_mises_stress"] = self._calculate_von_mises_stress(final_stress)
        properties["equivalent_strain"] = self._calculate_equivalent_strain(
            final_strain
        )

        return properties

    def _calculate_von_mises_stress(self, stress_tensor: np.ndarray) -> float:
        """Calculate von Mises equivalent stress."""
        s11, s22, s33 = stress_tensor[0, 0], stress_tensor[1, 1], stress_tensor[2, 2]
        s12, s13, s23 = stress_tensor[0, 1], stress_tensor[0, 2], stress_tensor[1, 2]

        return np.sqrt(
            0.5
            * (
                (s11 - s22) ** 2
                + (s22 - s33) ** 2
                + (s33 - s11) ** 2
                + 6 * (s12**2 + s13**2 + s23**2)
            )
        )

    def _calculate_equivalent_strain(self, strain_tensor: np.ndarray) -> float:
        """Calculate equivalent strain."""
        e11, e22, e33 = strain_tensor[0, 0], strain_tensor[1, 1], strain_tensor[2, 2]
        e12, e13, e23 = strain_tensor[0, 1], strain_tensor[0, 2], strain_tensor[1, 2]

        return np.sqrt(
            (2 / 3)
            * (e11**2 + e22**2 + e33**2 + 2 * (e12**2 + e13**2 + e23**2))
        )


class ReactiveForceFieldSimulator(BaseSimulator):
    """Simulator for reactive force fields under extreme conditions."""

    def __init__(self, backend: SimulationBackend = SimulationBackend.CUDA):
        super().__init__(backend)
        self.force_field_type = "ReaxFF"  # Default to ReaxFF

    def run_simulation(
        self, initial_structure: Dict[str, Any], parameters: SimulationParameters
    ) -> SimulationResults:
        """Run reactive force field simulation."""
        self.logger.info(f"Running {self.force_field_type} simulation")

        # Placeholder implementation - would interface with actual ReaxFF
        # or similar reactive force field implementations

        # For now, return empty results
        return SimulationResults(
            computation_time=0.1,
            simulation_metadata={
                "backend": self.backend.value,
                "force_field": self.force_field_type,
                "condition_type": parameters.condition_type.value,
            },
        )

    def calculate_properties(self, results: SimulationResults) -> Dict[str, float]:
        """Calculate properties from reactive simulation."""
        return {
            "bond_breaking_events": 0,
            "bond_formation_events": 0,
            "chemical_reaction_rate": 0.0,
        }


class MultiScaleSimulator(BaseSimulator):
    """Multi-scale simulator combining different physics scales."""

    def __init__(self, backend: SimulationBackend = SimulationBackend.CUDA):
        super().__init__(backend)
        self.md_simulator = NEMDSimulator(backend)
        self.continuum_solver = None  # Would be implemented for continuum mechanics

    def run_simulation(
        self, initial_structure: Dict[str, Any], parameters: SimulationParameters
    ) -> SimulationResults:
        """Run multi-scale simulation."""
        self.logger.info("Running multi-scale simulation")

        # Run molecular dynamics at small scale
        md_results = self.md_simulator.run_simulation(initial_structure, parameters)

        # Extract homogenized properties for continuum scale
        # (This would be implemented in a full multi-scale framework)

        return md_results

    def calculate_properties(self, results: SimulationResults) -> Dict[str, float]:
        """Calculate multi-scale properties."""
        md_props = self.md_simulator.calculate_properties(results)

        # Add continuum-scale properties
        md_props.update(
            {
                "macroscopic_stress": 0.0,
                "macroscopic_strain": 0.0,
                "homogenized_stiffness": 0.0,
            }
        )

        return md_props


class NonEquilibriumSimulationEngine:
    """Main engine for coordinating non-equilibrium simulations."""

    def __init__(self, backend: SimulationBackend = SimulationBackend.CUDA):
        self.backend = backend
        self.logger = get_logger(self.__class__.__name__)

        # Initialize simulators
        self.nemd_simulator = NEMDSimulator(backend)
        self.reactive_simulator = ReactiveForceFieldSimulator(backend)
        self.multiscale_simulator = MultiScaleSimulator(backend)

        self.available_simulators = {
            "nemd": self.nemd_simulator,
            "reactive": self.reactive_simulator,
            "multiscale": self.multiscale_simulator,
        }

    def run_extreme_condition_simulation(
        self,
        structure: Dict[str, Any],
        condition_type: ExtremeConditionType,
        simulator_type: str = "nemd",
        **kwargs,
    ) -> SimulationResults:
        """Run simulation under specified extreme conditions."""

        if simulator_type not in self.available_simulators:
            raise ValueError(f"Unknown simulator type: {simulator_type}")

        # Create simulation parameters
        parameters = SimulationParameters(condition_type=condition_type, **kwargs)

        simulator = self.available_simulators[simulator_type]

        self.logger.info(
            f"Running {simulator_type} simulation for {condition_type.value}"
        )

        results = simulator.run_simulation(structure, parameters)

        # Calculate derived properties
        properties = simulator.calculate_properties(results)
        results.simulation_metadata["calculated_properties"] = properties

        return results

    def benchmark_simulation_performance(
        self,
        structure: Dict[str, Any],
        condition_types: List[ExtremeConditionType],
        simulator_types: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark performance across different conditions and simulators."""

        benchmark_results = {}

        for condition in condition_types:
            benchmark_results[condition.value] = {}

            for sim_type in simulator_types:
                if sim_type in self.available_simulators:
                    try:
                        results = self.run_extreme_condition_simulation(
                            structure,
                            condition,
                            sim_type,
                            total_time=1e-13,  # Short simulation for benchmarking
                            output_frequency=10,
                        )

                        benchmark_results[condition.value][sim_type] = {
                            "computation_time": results.computation_time,
                            "atoms_per_second": (
                                len(structure["positions"]) / results.computation_time
                            ),
                        }

                    except Exception as e:
                        self.logger.warning(
                            f"Benchmark failed for {condition.value}/{sim_type}: {e}"
                        )
                        benchmark_results[condition.value][sim_type] = {
                            "computation_time": float("inf"),
                            "atoms_per_second": 0.0,
                        }

        return benchmark_results

    def get_available_conditions(self) -> List[ExtremeConditionType]:
        """Get list of available extreme conditions."""
        return list(ExtremeConditionType)

    def get_available_simulators(self) -> List[str]:
        """Get list of available simulator types."""
        return list(self.available_simulators.keys())

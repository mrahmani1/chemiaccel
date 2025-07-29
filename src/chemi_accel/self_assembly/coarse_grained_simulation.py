"""
GPU-Accelerated Coarse-Grained and Dissipative Particle Dynamics (DPD) Simulation Engine.

This module provides high-performance GPU-accelerated simulation engines for coarse-grained
molecular dynamics and DPD simulations to model mesoscale self-assembly phenomena.
It implements efficient particle interaction potentials and GPU kernels optimized for
self-assembling systems.
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Optional GPU acceleration dependencies
try:
    import cupy as cp
    import cupyx.scipy.spatial.distance as cupy_distance

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cupy_distance = None
    CUPY_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

# Core scientific dependencies
try:
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, squareform

    SCIPY_AVAILABLE = True
except ImportError:
    pdist = None
    squareform = None
    minimize = None
    SCIPY_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class SimulationMethod(Enum):
    """Enumeration of available simulation methods."""

    COARSE_GRAINED_MD = "coarse_grained_md"
    DISSIPATIVE_PARTICLE_DYNAMICS = "dpd"
    BROWNIAN_DYNAMICS = "brownian_dynamics"
    LANGEVIN_DYNAMICS = "langevin_dynamics"


class InteractionType(Enum):
    """Types of particle interaction potentials."""

    LENNARD_JONES = "lennard_jones"
    DPD_CONSERVATIVE = "dpd_conservative"
    DPD_DISSIPATIVE = "dpd_dissipative"
    DPD_RANDOM = "dpd_random"
    SOFT_REPULSIVE = "soft_repulsive"
    HARMONIC_BOND = "harmonic_bond"
    ANGLE_POTENTIAL = "angle_potential"
    ELECTROSTATIC = "electrostatic"


class BoundaryCondition(Enum):
    """Boundary condition types for simulation box."""

    PERIODIC = "periodic"
    REFLECTING = "reflecting"
    ABSORBING = "absorbing"
    FIXED = "fixed"


class IntegratorType(Enum):
    """Numerical integration schemes."""

    VERLET = "verlet"
    VELOCITY_VERLET = "velocity_verlet"
    DPD_INTEGRATOR = "dpd_integrator"
    LANGEVIN_INTEGRATOR = "langevin_integrator"
    RUNGE_KUTTA = "runge_kutta"


@dataclass
class ParticleType:
    """Definition of a particle type with interaction parameters."""

    name: str
    mass: float
    radius: float
    charge: float = 0.0
    epsilon: float = 1.0  # LJ energy parameter
    sigma: float = 1.0  # LJ length parameter
    color: str = "blue"  # For visualization
    hydrophobic: bool = False

    def __post_init__(self):
        if self.mass <= 0:
            raise ValueError("Particle mass must be positive")
        if self.radius <= 0:
            raise ValueError("Particle radius must be positive")


@dataclass
class Particle:
    """Individual particle with position, velocity, and type information."""

    particle_id: int
    particle_type: ParticleType
    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    force: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    bonds: List[int] = field(default_factory=list)

    def __post_init__(self):
        if len(self.position) != 3:
            raise ValueError("Position must be 3D vector")
        if len(self.velocity) != 3:
            raise ValueError("Velocity must be 3D vector")


@dataclass
class Bond:
    """Bond between two particles."""

    particle1_id: int
    particle2_id: int
    equilibrium_length: float
    spring_constant: float
    bond_type: str = "harmonic"


@dataclass
class SimulationBox:
    """Simulation box with boundary conditions."""

    dimensions: npt.NDArray[np.float64]
    boundary_conditions: List[BoundaryCondition]
    periodic_box: bool = True

    def __post_init__(self):
        if len(self.dimensions) != 3:
            raise ValueError("Box dimensions must be 3D")
        if len(self.boundary_conditions) != 3:
            self.boundary_conditions = [BoundaryCondition.PERIODIC] * 3


@dataclass
class SimulationParameters:
    """Parameters for coarse-grained simulation."""

    method: SimulationMethod = SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS
    integrator: IntegratorType = IntegratorType.DPD_INTEGRATOR
    timestep: float = 0.01
    total_time: float = 100.0
    temperature: float = 298.15
    pressure: float = 1.0
    cutoff_radius: float = 2.5
    neighbor_list_skin: float = 0.3
    update_neighbor_frequency: int = 10
    output_frequency: int = 100
    use_gpu: bool = True
    gpu_device_id: int = 0
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.timestep <= 0:
            raise ValueError("Timestep must be positive")
        if self.total_time <= 0:
            raise ValueError("Total simulation time must be positive")
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")


@dataclass
class InteractionParameters:
    """Parameters for particle interactions."""

    interaction_matrix: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=dict
    )
    dpd_gamma: float = 4.5  # DPD friction coefficient
    dpd_sigma: float = 3.0  # DPD noise amplitude
    bond_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    angle_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add_interaction(
        self,
        type1: str,
        type2: str,
        interaction_type: InteractionType,
        parameters: Dict[str, float],
    ):
        """Add interaction parameters between two particle types."""
        key = tuple(sorted([type1, type2]))
        if key not in self.interaction_matrix:
            self.interaction_matrix[key] = {}
        self.interaction_matrix[key][interaction_type.value] = parameters

    def get_interaction(
        self, type1: str, type2: str, interaction_type: InteractionType
    ) -> Optional[Dict[str, float]]:
        """Get interaction parameters between two particle types."""
        key = tuple(sorted([type1, type2]))
        return self.interaction_matrix.get(key, {}).get(interaction_type.value)


@dataclass
class SimulationState:
    """Current state of the simulation."""

    step: int = 0
    time: float = 0.0
    total_energy: float = 0.0
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    temperature: float = 0.0
    pressure: float = 0.0
    volume: float = 0.0
    box_dimensions: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))


@dataclass
class TrajectoryFrame:
    """Single frame of simulation trajectory."""

    step: int
    time: float
    positions: npt.NDArray[np.float64]
    velocities: npt.NDArray[np.float64]
    forces: npt.NDArray[np.float64]
    energies: Dict[str, float]
    box_dimensions: npt.NDArray[np.float64]


class CoarseGrainedSimulationError(Exception):
    """Base exception for coarse-grained simulation errors."""

    pass


class GPUMemoryError(CoarseGrainedSimulationError):
    """Exception raised when GPU memory is insufficient."""

    pass


class IntegrationError(CoarseGrainedSimulationError):
    """Exception raised during numerical integration."""

    pass


class ParticleInteractionCalculator:
    """GPU-accelerated calculator for particle interactions."""

    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device_id = device_id

        if self.use_gpu:
            cp.cuda.Device(device_id).use()
            logger.info(f"Using GPU device {device_id} for particle interactions")
        else:
            logger.info("Using CPU for particle interactions")

    def calculate_lennard_jones_forces(
        self,
        positions: npt.NDArray[np.float64],
        particle_types: List[str],
        interaction_params: InteractionParameters,
        box_dimensions: npt.NDArray[np.float64],
        cutoff: float,
    ) -> npt.NDArray[np.float64]:
        """Calculate Lennard-Jones forces between particles."""
        n_particles = len(positions)

        if self.use_gpu:
            return self._calculate_lj_forces_gpu(
                positions, particle_types, interaction_params, box_dimensions, cutoff
            )
        else:
            return self._calculate_lj_forces_cpu(
                positions, particle_types, interaction_params, box_dimensions, cutoff
            )

    def _calculate_lj_forces_gpu(
        self,
        positions: npt.NDArray[np.float64],
        particle_types: List[str],
        interaction_params: InteractionParameters,
        box_dimensions: npt.NDArray[np.float64],
        cutoff: float,
    ) -> npt.NDArray[np.float64]:
        """GPU implementation of Lennard-Jones force calculation."""
        n_particles = len(positions)
        forces = cp.zeros((n_particles, 3), dtype=cp.float64)
        positions_gpu = cp.asarray(positions, dtype=cp.float64)
        box_gpu = cp.asarray(box_dimensions, dtype=cp.float64)

        # Custom CUDA kernel for LJ force calculation
        lj_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void calculate_lj_forces(double* positions, double* forces, double* box,
                                int n_particles, double cutoff, double* epsilon_matrix,
                                double* sigma_matrix) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_particles) return;

            double fi[3] = {0.0, 0.0, 0.0};
            double pos_i[3] = {positions[3*i], positions[3*i+1], positions[3*i+2]};

            for (int j = i + 1; j < n_particles; j++) {
                double pos_j[3] = {positions[3*j], positions[3*j+1], positions[3*j+2]};
                double dr[3];
                double r2 = 0.0;

                // Apply periodic boundary conditions
                for (int k = 0; k < 3; k++) {
                    dr[k] = pos_i[k] - pos_j[k];
                    dr[k] = dr[k] - box[k] * round(dr[k] / box[k]);
                    r2 += dr[k] * dr[k];
                }

                double r = sqrt(r2);
                if (r < cutoff && r > 0.0) {
                    // Get interaction parameters (simplified)
                    double epsilon = epsilon_matrix[i * n_particles + j];
                    double sigma = sigma_matrix[i * n_particles + j];

                    // Lennard-Jones potential
                    double sr6 = pow(sigma / r, 6.0);
                    double sr12 = sr6 * sr6;
                    double force_magnitude = 24.0 * epsilon * (2.0 * sr12 - sr6) / (r * r);

                    for (int k = 0; k < 3; k++) {
                        double f = force_magnitude * dr[k] / r;
                        fi[k] += f;
                        atomicAdd(&forces[3*j + k], -f);
                    }
                }
            }

            for (int k = 0; k < 3; k++) {
                forces[3*i + k] += fi[k];
            }
        }
        """,
            "calculate_lj_forces",
        )

        # Prepare parameter matrices
        epsilon_matrix = cp.ones((n_particles, n_particles), dtype=cp.float64)
        sigma_matrix = cp.ones((n_particles, n_particles), dtype=cp.float64)

        # Launch kernel
        block_size = 256
        grid_size = (n_particles + block_size - 1) // block_size
        lj_kernel(
            (grid_size,),
            (block_size,),
            (
                positions_gpu,
                forces,
                box_gpu,
                n_particles,
                cutoff,
                epsilon_matrix,
                sigma_matrix,
            ),
        )

        return cp.asnumpy(forces)

    def _calculate_lj_forces_cpu(
        self,
        positions: npt.NDArray[np.float64],
        particle_types: List[str],
        interaction_params: InteractionParameters,
        box_dimensions: npt.NDArray[np.float64],
        cutoff: float,
    ) -> npt.NDArray[np.float64]:
        """CPU implementation of Lennard-Jones force calculation."""
        n_particles = len(positions)
        forces = np.zeros((n_particles, 3), dtype=np.float64)

        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                dr = positions[i] - positions[j]
                # Apply periodic boundary conditions
                dr = dr - box_dimensions * np.round(dr / box_dimensions)
                r2 = np.sum(dr**2)
                r = np.sqrt(r2)

                if r < cutoff and r > 0:
                    # Get interaction parameters
                    type_i, type_j = particle_types[i], particle_types[j]
                    params = interaction_params.get_interaction(
                        type_i, type_j, InteractionType.LENNARD_JONES
                    )

                    if params:
                        epsilon = params.get("epsilon", 1.0)
                        sigma = params.get("sigma", 1.0)

                        # Lennard-Jones force
                        sr6 = (sigma / r) ** 6
                        sr12 = sr6**2
                        force_magnitude = 24 * epsilon * (2 * sr12 - sr6) / r2
                        force_vector = force_magnitude * dr / r

                        forces[i] += force_vector
                        forces[j] -= force_vector

        return forces

    def calculate_dpd_forces(
        self,
        positions: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64],
        particle_types: List[str],
        interaction_params: InteractionParameters,
        box_dimensions: npt.NDArray[np.float64],
        cutoff: float,
        temperature: float,
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Calculate DPD (conservative, dissipative, and random) forces."""
        if self.use_gpu:
            return self._calculate_dpd_forces_gpu(
                positions,
                velocities,
                particle_types,
                interaction_params,
                box_dimensions,
                cutoff,
                temperature,
                dt,
            )
        else:
            return self._calculate_dpd_forces_cpu(
                positions,
                velocities,
                particle_types,
                interaction_params,
                box_dimensions,
                cutoff,
                temperature,
                dt,
            )

    def _calculate_dpd_forces_gpu(
        self,
        positions: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64],
        particle_types: List[str],
        interaction_params: InteractionParameters,
        box_dimensions: npt.NDArray[np.float64],
        cutoff: float,
        temperature: float,
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """GPU implementation of DPD force calculation."""
        n_particles = len(positions)
        forces = cp.zeros((n_particles, 3), dtype=cp.float64)
        positions_gpu = cp.asarray(positions, dtype=cp.float64)
        velocities_gpu = cp.asarray(velocities, dtype=cp.float64)
        box_gpu = cp.asarray(box_dimensions, dtype=cp.float64)

        # Generate random numbers for stochastic forces
        random_forces = cp.random.normal(
            0, 1, (n_particles, n_particles, 3), dtype=cp.float64
        )

        # DPD force calculation kernel
        dpd_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void calculate_dpd_forces(double* positions, double* velocities, double* forces,
                                 double* box, double* random_forces, int n_particles,
                                 double cutoff, double gamma, double sigma, double dt,
                                 double temperature, double* a_matrix) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n_particles) return;

            double fi[3] = {0.0, 0.0, 0.0};
            double pos_i[3] = {positions[3*i], positions[3*i+1], positions[3*i+2]};
            double vel_i[3] = {velocities[3*i], velocities[3*i+1], velocities[3*i+2]};

            for (int j = i + 1; j < n_particles; j++) {
                double pos_j[3] = {positions[3*j], positions[3*j+1], positions[3*j+2]};
                double vel_j[3] = {velocities[3*j], velocities[3*j+1], velocities[3*j+2]};
                double dr[3], dv[3];
                double r2 = 0.0;

                // Apply periodic boundary conditions
                for (int k = 0; k < 3; k++) {
                    dr[k] = pos_i[k] - pos_j[k];
                    dr[k] = dr[k] - box[k] * round(dr[k] / box[k]);
                    dv[k] = vel_i[k] - vel_j[k];
                    r2 += dr[k] * dr[k];
                }

                double r = sqrt(r2);
                if (r < cutoff && r > 0.0) {
                    double w_c = 1.0 - r / cutoff;  // Conservative weight
                    double w_d = w_c * w_c;         // Dissipative weight
                    double w_r = w_c;               // Random weight

                    // Conservative force
                    double a_ij = a_matrix[i * n_particles + j];
                    double f_c_mag = a_ij * w_c;

                    // Dissipative force
                    double rdv = 0.0;
                    for (int k = 0; k < 3; k++) {
                        rdv += dr[k] * dv[k];
                    }
                    double f_d_mag = -gamma * w_d * rdv / r;

                    // Random force
                    double xi = random_forces[i * n_particles * 3 + j * 3];
                    double f_r_mag = sigma * w_r * xi / sqrt(dt);

                    // Total force
                    for (int k = 0; k < 3; k++) {
                        double e_ij = dr[k] / r;  // Unit vector
                        double f_total = (f_c_mag + f_d_mag + f_r_mag) * e_ij;
                        fi[k] += f_total;
                        atomicAdd(&forces[3*j + k], -f_total);
                    }
                }
            }

            for (int k = 0; k < 3; k++) {
                forces[3*i + k] += fi[k];
            }
        }
        """,
            "calculate_dpd_forces",
        )

        # Prepare parameter matrices
        a_matrix = (
            cp.ones((n_particles, n_particles), dtype=cp.float64) * 25.0
        )  # Default repulsion
        gamma = interaction_params.dpd_gamma
        sigma = interaction_params.dpd_sigma

        # Launch kernel
        block_size = 256
        grid_size = (n_particles + block_size - 1) // block_size
        dpd_kernel(
            (grid_size,),
            (block_size,),
            (
                positions_gpu,
                velocities_gpu,
                forces,
                box_gpu,
                random_forces,
                n_particles,
                cutoff,
                gamma,
                sigma,
                dt,
                temperature,
                a_matrix,
            ),
        )

        return cp.asnumpy(forces)

    def _calculate_dpd_forces_cpu(
        self,
        positions: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64],
        particle_types: List[str],
        interaction_params: InteractionParameters,
        box_dimensions: npt.NDArray[np.float64],
        cutoff: float,
        temperature: float,
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """CPU implementation of DPD force calculation."""
        n_particles = len(positions)
        forces = np.zeros((n_particles, 3), dtype=np.float64)
        gamma = interaction_params.dpd_gamma
        sigma = interaction_params.dpd_sigma

        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                dr = positions[i] - positions[j]
                dv = velocities[i] - velocities[j]

                # Apply periodic boundary conditions
                dr = dr - box_dimensions * np.round(dr / box_dimensions)
                r = np.linalg.norm(dr)

                if r < cutoff and r > 0:
                    # Weight functions
                    w_c = 1.0 - r / cutoff
                    w_d = w_c**2
                    w_r = w_c

                    # Unit vector
                    e_ij = dr / r

                    # Conservative force
                    type_i, type_j = particle_types[i], particle_types[j]
                    params = interaction_params.get_interaction(
                        type_i, type_j, InteractionType.DPD_CONSERVATIVE
                    )
                    a_ij = params.get("a", 25.0) if params else 25.0
                    f_c = a_ij * w_c * e_ij

                    # Dissipative force
                    f_d = -gamma * w_d * np.dot(e_ij, dv) * e_ij

                    # Random force
                    xi = np.random.normal(0, 1, 3)
                    f_r = sigma * w_r * xi / np.sqrt(dt)

                    # Total force
                    f_total = f_c + f_d + f_r
                    forces[i] += f_total
                    forces[j] -= f_total

        return forces


class MDIntegrator:
    """Numerical integrators for molecular dynamics."""

    @staticmethod
    def velocity_verlet_step(
        positions: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64],
        forces: npt.NDArray[np.float64],
        masses: npt.NDArray[np.float64],
        dt: float,
        box_dimensions: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Perform a velocity-Verlet integration step."""
        # Update positions
        new_positions = (
            positions + velocities * dt + 0.5 * forces / masses[:, np.newaxis] * dt**2
        )

        # Apply periodic boundary conditions
        new_positions = new_positions % box_dimensions

        # Update velocities (first half)
        new_velocities = velocities + 0.5 * forces / masses[:, np.newaxis] * dt

        return new_positions, new_velocities

    @staticmethod
    def dpd_integrator_step(
        positions: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64],
        forces: npt.NDArray[np.float64],
        masses: npt.NDArray[np.float64],
        dt: float,
        box_dimensions: npt.NDArray[np.float64],
        temperature: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Perform a DPD-specific integration step."""
        # Modified velocity-Verlet for DPD
        new_positions = (
            positions + velocities * dt + 0.5 * forces / masses[:, np.newaxis] * dt**2
        )
        new_positions = new_positions % box_dimensions

        new_velocities = velocities + forces / masses[:, np.newaxis] * dt

        return new_positions, new_velocities


class CoarseGrainedSimulationEngine:
    """Main GPU-accelerated coarse-grained simulation engine."""

    def __init__(
        self,
        particles: List[Particle],
        simulation_box: SimulationBox,
        simulation_params: SimulationParameters,
        interaction_params: InteractionParameters,
    ):
        self.particles = particles
        self.simulation_box = simulation_box
        self.simulation_params = simulation_params
        self.interaction_params = interaction_params

        # Initialize state
        self.state = SimulationState()
        self.trajectory = []

        # Initialize calculators
        self.force_calculator = ParticleInteractionCalculator(
            use_gpu=simulation_params.use_gpu, device_id=simulation_params.gpu_device_id
        )

        # Prepare arrays
        self._prepare_simulation_arrays()

        # Set random seed
        if simulation_params.random_seed is not None:
            np.random.seed(simulation_params.random_seed)
            if CUPY_AVAILABLE and simulation_params.use_gpu:
                cp.random.seed(simulation_params.random_seed)

        logger.info(
            f"Initialized coarse-grained simulation with {len(particles)} particles"
        )

    def _prepare_simulation_arrays(self):
        """Prepare numpy arrays for simulation."""
        n_particles = len(self.particles)

        self.positions = np.array(
            [p.position for p in self.particles], dtype=np.float64
        )
        self.velocities = np.array(
            [p.velocity for p in self.particles], dtype=np.float64
        )
        self.forces = np.zeros((n_particles, 3), dtype=np.float64)
        self.masses = np.array(
            [p.particle_type.mass for p in self.particles], dtype=np.float64
        )
        self.particle_types = [p.particle_type.name for p in self.particles]

        # Initialize box dimensions
        self.state.box_dimensions = self.simulation_box.dimensions.copy()
        self.state.volume = np.prod(self.simulation_box.dimensions)

    def run_simulation(self) -> List[TrajectoryFrame]:
        """Run the complete simulation."""
        logger.info("Starting coarse-grained simulation")
        start_time = time.time()

        n_steps = int(
            self.simulation_params.total_time / self.simulation_params.timestep
        )

        try:
            for step in range(n_steps):
                self._simulation_step(step)

                # Save trajectory frame
                if step % self.simulation_params.output_frequency == 0:
                    self._save_trajectory_frame(step)

                # Log progress
                if step % (n_steps // 10) == 0:
                    progress = 100 * step / n_steps
                    logger.info(f"Simulation progress: {progress:.1f}%")

        except Exception as e:
            logger.error(f"Simulation failed at step {step}: {e}")
            raise CoarseGrainedSimulationError(f"Simulation failed: {e}")

        end_time = time.time()
        logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds")

        return self.trajectory

    def _simulation_step(self, step: int):
        """Perform a single simulation step."""
        dt = self.simulation_params.timestep

        # Calculate forces
        self._calculate_forces()

        # Integrate equations of motion
        if self.simulation_params.integrator == IntegratorType.VELOCITY_VERLET:
            self.positions, self.velocities = MDIntegrator.velocity_verlet_step(
                self.positions,
                self.velocities,
                self.forces,
                self.masses,
                dt,
                self.state.box_dimensions,
            )
        elif self.simulation_params.integrator == IntegratorType.DPD_INTEGRATOR:
            self.positions, self.velocities = MDIntegrator.dpd_integrator_step(
                self.positions,
                self.velocities,
                self.forces,
                self.masses,
                dt,
                self.state.box_dimensions,
                self.simulation_params.temperature,
            )

        # Update state
        self.state.step = step
        self.state.time = step * dt
        self._calculate_energies()
        self._calculate_thermodynamic_properties()

    def _calculate_forces(self):
        """Calculate all forces acting on particles."""
        self.forces.fill(0.0)

        if (
            self.simulation_params.method
            == SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS
        ):
            # DPD forces
            dpd_forces = self.force_calculator.calculate_dpd_forces(
                self.positions,
                self.velocities,
                self.particle_types,
                self.interaction_params,
                self.state.box_dimensions,
                self.simulation_params.cutoff_radius,
                self.simulation_params.temperature,
                self.simulation_params.timestep,
            )
            self.forces += dpd_forces

        elif self.simulation_params.method == SimulationMethod.COARSE_GRAINED_MD:
            # Lennard-Jones forces
            lj_forces = self.force_calculator.calculate_lennard_jones_forces(
                self.positions,
                self.particle_types,
                self.interaction_params,
                self.state.box_dimensions,
                self.simulation_params.cutoff_radius,
            )
            self.forces += lj_forces

        # Add bond forces if present
        self._calculate_bond_forces()

    def _calculate_bond_forces(self):
        """Calculate forces from bonded interactions."""
        for particle in self.particles:
            for bond_partner_id in particle.bonds:
                if bond_partner_id > particle.particle_id:  # Avoid double counting
                    # Simple harmonic bond
                    dr = (
                        self.positions[particle.particle_id]
                        - self.positions[bond_partner_id]
                    )
                    dr = dr - self.state.box_dimensions * np.round(
                        dr / self.state.box_dimensions
                    )
                    r = np.linalg.norm(dr)

                    # Default bond parameters
                    k_bond = 100.0  # Spring constant
                    r0 = 1.0  # Equilibrium length

                    if r > 0:
                        force_magnitude = -k_bond * (r - r0)
                        force_vector = force_magnitude * dr / r

                        self.forces[particle.particle_id] += force_vector
                        self.forces[bond_partner_id] -= force_vector

    def _calculate_energies(self):
        """Calculate kinetic and potential energies."""
        # Kinetic energy
        ke = 0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities**2)
        self.state.kinetic_energy = ke

        # Potential energy (simplified)
        pe = 0.0  # Would need to recalculate interactions
        self.state.potential_energy = pe
        self.state.total_energy = ke + pe

    def _calculate_thermodynamic_properties(self):
        """Calculate temperature and pressure."""
        # Temperature from kinetic energy
        n_particles = len(self.particles)
        self.state.temperature = (
            (2.0 / 3.0) * self.state.kinetic_energy / (n_particles * 1.380649e-23)
        )

        # Pressure (simplified ideal gas approximation)
        self.state.pressure = (
            n_particles * 1.380649e-23 * self.state.temperature / self.state.volume
        )

    def _save_trajectory_frame(self, step: int):
        """Save current state as trajectory frame."""
        frame = TrajectoryFrame(
            step=step,
            time=self.state.time,
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            forces=self.forces.copy(),
            energies={
                "kinetic": self.state.kinetic_energy,
                "potential": self.state.potential_energy,
                "total": self.state.total_energy,
            },
            box_dimensions=self.state.box_dimensions.copy(),
        )
        self.trajectory.append(frame)

    def get_current_state(self) -> SimulationState:
        """Get current simulation state."""
        return self.state

    def get_trajectory(self) -> List[TrajectoryFrame]:
        """Get complete trajectory."""
        return self.trajectory


def create_amphiphilic_system(
    n_molecules: int = 100,
    head_size: int = 1,
    tail_size: int = 8,
    box_size: float = 20.0,
) -> Tuple[List[Particle], SimulationBox, InteractionParameters]:
    """Create a system of amphiphilic molecules for self-assembly simulation."""

    # Define particle types
    head_type = ParticleType(name="HEAD", mass=1.0, radius=0.5, hydrophobic=False)
    tail_type = ParticleType(name="TAIL", mass=1.0, radius=0.5, hydrophobic=True)
    water_type = ParticleType(name="WATER", mass=1.0, radius=0.3, hydrophobic=False)

    particles = []
    particle_id = 0

    # Create amphiphilic molecules
    for mol_id in range(n_molecules):
        # Random position
        mol_center = np.random.uniform(0, box_size, 3)

        # Create head particle
        head_pos = mol_center + np.random.normal(0, 0.1, 3)
        head_particle = Particle(
            particle_id=particle_id,
            particle_type=head_type,
            position=head_pos,
            velocity=np.random.normal(0, 1, 3),
        )
        particles.append(head_particle)
        head_id = particle_id
        particle_id += 1

        # Create tail particles
        tail_ids = []
        for tail_idx in range(tail_size):
            tail_pos = (
                head_pos
                + (tail_idx + 1) * np.array([0.5, 0, 0])
                + np.random.normal(0, 0.05, 3)
            )
            tail_particle = Particle(
                particle_id=particle_id,
                particle_type=tail_type,
                position=tail_pos,
                velocity=np.random.normal(0, 1, 3),
            )
            particles.append(tail_particle)
            tail_ids.append(particle_id)
            particle_id += 1

        # Add bonds within molecule
        particles[head_id].bonds.append(tail_ids[0])
        for i in range(len(tail_ids) - 1):
            particles[tail_ids[i]].bonds.append(tail_ids[i + 1])

    # Add water particles
    n_water = n_molecules * 20  # Approximate water density
    for _ in range(n_water):
        water_pos = np.random.uniform(0, box_size, 3)
        water_particle = Particle(
            particle_id=particle_id,
            particle_type=water_type,
            position=water_pos,
            velocity=np.random.normal(0, 1, 3),
        )
        particles.append(water_particle)
        particle_id += 1

    # Create simulation box
    simulation_box = SimulationBox(
        dimensions=np.array([box_size, box_size, box_size]),
        boundary_conditions=[BoundaryCondition.PERIODIC] * 3,
    )

    # Set up interactions
    interaction_params = InteractionParameters()

    # DPD interactions (repulsion parameters)
    interaction_params.add_interaction(
        "HEAD", "HEAD", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
    )
    interaction_params.add_interaction(
        "HEAD", "TAIL", InteractionType.DPD_CONSERVATIVE, {"a": 40.0}
    )
    interaction_params.add_interaction(
        "HEAD", "WATER", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
    )
    interaction_params.add_interaction(
        "TAIL", "TAIL", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
    )
    interaction_params.add_interaction(
        "TAIL", "WATER", InteractionType.DPD_CONSERVATIVE, {"a": 80.0}
    )  # Strong repulsion
    interaction_params.add_interaction(
        "WATER", "WATER", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
    )

    return particles, simulation_box, interaction_params


# High-level convenience function
def run_dpd_simulation(
    particles: List[Particle],
    simulation_box: SimulationBox,
    interaction_params: InteractionParameters,
    total_time: float = 100.0,
    timestep: float = 0.01,
    temperature: float = 298.15,
    use_gpu: bool = True,
) -> List[TrajectoryFrame]:
    """Run a DPD simulation with default parameters."""

    sim_params = SimulationParameters(
        method=SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS,
        integrator=IntegratorType.DPD_INTEGRATOR,
        timestep=timestep,
        total_time=total_time,
        temperature=temperature,
        use_gpu=use_gpu,
    )

    engine = CoarseGrainedSimulationEngine(
        particles=particles,
        simulation_box=simulation_box,
        simulation_params=sim_params,
        interaction_params=interaction_params,
    )

    return engine.run_simulation()

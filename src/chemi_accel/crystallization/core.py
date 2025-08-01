"""
Core data structures and base classes for crystallization engineering.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class CrystallizationMethod(Enum):
    """Crystallization methods supported by the system."""

    COOLING = "cooling"
    ANTISOLVENT = "antisolvent"
    EVAPORATION = "evaporation"
    REACTIVE = "reactive"
    SALTING_OUT = "salting_out"
    SEEDED = "seeded"


class PolymorphType(Enum):
    """Types of polymorphs that can be predicted."""

    FORM_I = "form_i"
    FORM_II = "form_ii"
    FORM_III = "form_iii"
    HYDRATE = "hydrate"
    SOLVATE = "solvate"
    AMORPHOUS = "amorphous"


@dataclass
class ProcessConditions:
    """
    Comprehensive process conditions for crystallization.

    Attributes:
        temperature_profile: Temperature vs time profile (K, s)
        solvent_composition: Solvent mixture composition
        stirring_rate: Stirring rate (rpm)
        seeding_conditions: Seeding parameters
        supersaturation_profile: Supersaturation vs time
        pressure: Process pressure (Pa)
        ph_profile: pH vs time profile
        antisolvent_addition_rate: Rate of antisolvent addition (mL/min)
        nucleation_promoters: Nucleation promoter concentrations
        impurity_levels: Impurity concentrations (ppm)
    """

    temperature_profile: List[Tuple[float, float]] = field(default_factory=list)
    solvent_composition: Dict[str, float] = field(default_factory=dict)
    stirring_rate: float = 200.0
    seeding_conditions: Optional[Dict[str, Any]] = None
    supersaturation_profile: List[Tuple[float, float]] = field(default_factory=list)
    pressure: float = 101325.0  # Pa
    ph_profile: List[Tuple[float, float]] = field(default_factory=list)
    antisolvent_addition_rate: Optional[float] = None
    nucleation_promoters: Dict[str, float] = field(default_factory=dict)
    impurity_levels: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "temperature_profile": self.temperature_profile,
            "solvent_composition": self.solvent_composition,
            "stirring_rate": self.stirring_rate,
            "seeding_conditions": self.seeding_conditions,
            "supersaturation_profile": self.supersaturation_profile,
            "pressure": self.pressure,
            "ph_profile": self.ph_profile,
            "antisolvent_addition_rate": self.antisolvent_addition_rate,
            "nucleation_promoters": self.nucleation_promoters,
            "impurity_levels": self.impurity_levels,
        }


@dataclass
class CrystalProperties:
    """
    Properties of crystallized material.

    Attributes:
        polymorph_composition: Distribution of polymorphs
        particle_size_distribution: PSD parameters
        crystal_habit: Shape descriptors
        surface_area: Specific surface area (m²/g)
        dissolution_rate: Dissolution rate constant
        flowability_index: Flowability measurements
        purity: Chemical purity (%)
        yield_: Crystallization yield (%)
        crystal_structure: Unit cell parameters
        mechanical_properties: Hardness, compressibility
    """

    polymorph_composition: Dict[PolymorphType, float] = field(default_factory=dict)
    particle_size_distribution: Dict[str, float] = field(default_factory=dict)
    crystal_habit: Dict[str, float] = field(default_factory=dict)
    surface_area: float = 0.0
    dissolution_rate: float = 0.0
    flowability_index: float = 0.0
    purity: float = 100.0
    yield_: float = 0.0
    crystal_structure: Dict[str, float] = field(default_factory=dict)
    mechanical_properties: Dict[str, float] = field(default_factory=dict)

    def get_dominant_polymorph(self) -> Optional[PolymorphType]:
        """Get the dominant polymorph form."""
        if not self.polymorph_composition:
            return None
        return max(self.polymorph_composition, key=self.polymorph_composition.get)


@dataclass
class SimulationConfig:
    """
    Configuration for crystallization simulation.

    Attributes:
        time_span: Simulation time range (start, end) in seconds
        time_steps: Number of simulation time steps
        spatial_discretization: Spatial grid parameters
        gpu_acceleration: Whether to use GPU acceleration
        backend: Computational backend ('cuda', 'jax', 'cpu')
        precision: Numerical precision ('float32', 'float64')
        batch_size: Batch size for parallel processing
        convergence_criteria: Convergence tolerance
        max_iterations: Maximum simulation iterations
        output_frequency: Frequency of output generation
    """

    time_span: Tuple[float, float] = (0.0, 3600.0)  # 1 hour default
    time_steps: int = 1000
    spatial_discretization: Dict[str, int] = field(
        default_factory=lambda: {"nx": 50, "ny": 50, "nz": 50}
    )
    gpu_acceleration: bool = True
    backend: str = "cuda"
    precision: str = "float32"
    batch_size: int = 32
    convergence_criteria: float = 1e-6
    max_iterations: int = 10000
    output_frequency: int = 100

    def validate(self) -> bool:
        """Validate simulation configuration."""
        if self.time_span[1] <= self.time_span[0]:
            raise ValueError("End time must be greater than start time")
        if self.time_steps <= 0:
            raise ValueError("Time steps must be positive")
        if self.backend not in ["cuda", "jax", "cpu"]:
            raise ValueError("Backend must be 'cuda', 'jax', or 'cpu'")
        return True


class CrystallizationCore(ABC):
    """
    Abstract base class for crystallization components.

    Provides common functionality and interface for all crystallization
    modeling and optimization components.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize crystallization core.

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self.config.validate()
        self._initialized = False
        self._simulation_data = {}

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component."""
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        pass

    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized

    def get_simulation_data(self) -> Dict[str, Any]:
        """Get current simulation data."""
        return self._simulation_data.copy()

    def update_config(self, new_config: SimulationConfig) -> None:
        """Update simulation configuration."""
        new_config.validate()
        self.config = new_config
        self._initialized = False  # Require re-initialization


@dataclass
class KineticsParameters:
    """
    Parameters for crystallization kinetics modeling.

    Attributes:
        nucleation_rate_constant: Primary nucleation rate constant
        growth_rate_constant: Crystal growth rate constant
        secondary_nucleation_constant: Secondary nucleation constant
        agglomeration_kernel: Agglomeration rate kernel
        breakage_kernel: Crystal breakage rate kernel
        surface_energy: Crystal-solution interfacial energy
        molecular_volume: Molecular volume in crystal
        diffusion_coefficient: Solute diffusion coefficient
        activation_energies: Activation energies for various processes
    """

    nucleation_rate_constant: float = 1e6
    growth_rate_constant: float = 1e-8
    secondary_nucleation_constant: float = 1e4
    agglomeration_kernel: float = 1e-17
    breakage_kernel: float = 1e-3
    surface_energy: float = 0.01  # J/m²
    molecular_volume: float = 1e-28  # m³
    diffusion_coefficient: float = 1e-9  # m²/s
    activation_energies: Dict[str, float] = field(
        default_factory=lambda: {
            "nucleation": 50000.0,  # J/mol
            "growth": 30000.0,  # J/mol
            "dissolution": 40000.0,  # J/mol
        }
    )


@dataclass
class RealTimeData:
    """
    Real-time process data structure.

    Attributes:
        timestamp: Data timestamp
        temperature: Current temperature (K)
        turbidity: Solution turbidity
        raman_spectrum: Raman spectroscopy data
        pvm_images: Particle vision measurement images
        ph: Solution pH
        conductivity: Solution conductivity
        supersaturation: Calculated supersaturation
        particle_count: Particle count data
        stirring_rate: Current stirring rate
    """

    timestamp: float
    temperature: float
    turbidity: Optional[float] = None
    raman_spectrum: Optional[np.ndarray] = None
    pvm_images: Optional[List[np.ndarray]] = None
    ph: Optional[float] = None
    conductivity: Optional[float] = None
    supersaturation: Optional[float] = None
    particle_count: Optional[Dict[str, int]] = None
    stirring_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = {
            "timestamp": self.timestamp,
            "temperature": self.temperature,
            "turbidity": self.turbidity,
            "ph": self.ph,
            "conductivity": self.conductivity,
            "supersaturation": self.supersaturation,
            "particle_count": self.particle_count,
            "stirring_rate": self.stirring_rate,
        }

        # Handle numpy arrays
        if self.raman_spectrum is not None:
            data["raman_spectrum"] = self.raman_spectrum.tolist()
        if self.pvm_images is not None:
            data["pvm_images"] = [img.tolist() for img in self.pvm_images]

        return data


class OptimizationObjective(Enum):
    """Optimization objectives for crystallization."""

    MAXIMIZE_YIELD = "maximize_yield"
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_PURITY = "maximize_purity"
    CONTROL_PSD = "control_psd"
    OPTIMIZE_HABIT = "optimize_habit"
    MINIMIZE_IMPURITIES = "minimize_impurities"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"


@dataclass
class OptimizationConstraints:
    """
    Constraints for crystallization optimization.

    Attributes:
        temperature_bounds: Min/max temperature bounds (K)
        time_bounds: Min/max process time bounds (s)
        solvent_constraints: Solvent composition constraints
        equipment_limits: Equipment operational limits
        quality_requirements: Product quality requirements
        safety_constraints: Safety-related constraints
    """

    temperature_bounds: Tuple[float, float] = (273.15, 373.15)
    time_bounds: Tuple[float, float] = (300.0, 7200.0)  # 5 min to 2 hours
    solvent_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    equipment_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    safety_constraints: Dict[str, float] = field(default_factory=dict)

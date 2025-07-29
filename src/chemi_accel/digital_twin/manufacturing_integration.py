"""
Manufacturing Process Integration for Digital Twin

Extends the multi-scale material digital twin to incorporate manufacturing
process parameters as inputs and outputs, with interfaces to process
engineering simulation tools.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.exceptions import ChemiAccelError
from .multiscale_framework import LengthScale, MaterialComponent, MultiScaleFramework

logger = logging.getLogger(__name__)


class ProcessType(Enum):
    """Manufacturing process types."""

    EXTRUSION = "extrusion"
    INJECTION_MOLDING = "injection_molding"
    COMPRESSION_MOLDING = "compression_molding"
    BLOW_MOLDING = "blow_molding"
    THERMOFORMING = "thermoforming"
    ADDITIVE_MANUFACTURING = "additive_manufacturing"
    COATING = "coating"
    CASTING = "casting"
    SPINNING = "spinning"
    PULTRUSION = "pultrusion"


@dataclass
class ProcessConditions:
    """Manufacturing process conditions."""

    temperature: Optional[float] = None  # K
    pressure: Optional[float] = None  # Pa
    flow_rate: Optional[float] = None  # kg/s
    shear_rate: Optional[float] = None  # 1/s
    residence_time: Optional[float] = None  # s
    cooling_rate: Optional[float] = None  # K/s
    feed_rate: Optional[float] = None  # kg/h
    screw_speed: Optional[float] = None  # rpm
    melt_temperature: Optional[float] = None  # K
    mold_temperature: Optional[float] = None  # K
    cycle_time: Optional[float] = None  # s
    pressure_profile: Optional[List[float]] = field(default_factory=list)
    temperature_profile: Optional[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temperature": self.temperature,
            "pressure": self.pressure,
            "flow_rate": self.flow_rate,
            "shear_rate": self.shear_rate,
            "residence_time": self.residence_time,
            "cooling_rate": self.cooling_rate,
            "feed_rate": self.feed_rate,
            "screw_speed": self.screw_speed,
            "melt_temperature": self.melt_temperature,
            "mold_temperature": self.mold_temperature,
            "cycle_time": self.cycle_time,
            "pressure_profile": self.pressure_profile,
            "temperature_profile": self.temperature_profile,
        }


@dataclass
class MaterialFlowProperties:
    """Material flow properties during processing."""

    viscosity: float = 1000.0  # Pa·s
    density: float = 1000.0  # kg/m³
    thermal_conductivity: float = 0.2  # W/m·K
    specific_heat: float = 2000.0  # J/kg·K
    thermal_expansion: float = 1e-4  # 1/K
    elastic_modulus: float = 1e9  # Pa
    yield_strength: float = 50e6  # Pa
    crystallization_kinetics: Dict[str, float] = field(default_factory=dict)
    degradation_kinetics: Dict[str, float] = field(default_factory=dict)

    def update_with_temperature(self, temperature: float) -> None:
        """Update properties based on temperature."""
        # Temperature-dependent viscosity (Arrhenius-like)
        if temperature > 0:
            reference_temp = 473.15  # 200°C
            activation_energy = 50000  # J/mol
            gas_constant = 8.314

            viscosity_ratio = np.exp(
                (activation_energy / gas_constant)
                * (1 / temperature - 1 / reference_temp)
            )
            self.viscosity = 1000.0 * viscosity_ratio

        # Temperature-dependent density
        thermal_expansion_effect = 1 - self.thermal_expansion * (temperature - 293.15)
        self.density = 1000.0 / thermal_expansion_effect


@dataclass
class ProcessOutput:
    """Manufacturing process output characteristics."""

    part_quality: float = 0.0  # Quality score (0-1)
    dimensional_accuracy: float = 0.0  # Dimensional accuracy (0-1)
    surface_finish: float = 0.0  # Surface finish quality (0-1)
    mechanical_properties: Dict[str, float] = field(default_factory=dict)
    morphology: Dict[str, Any] = field(default_factory=dict)
    residual_stress: float = 0.0  # Pa
    crystallinity: float = 0.0  # Fraction (0-1)
    orientation: Dict[str, float] = field(default_factory=dict)
    defects: List[str] = field(default_factory=list)
    processing_time: float = 0.0  # s
    energy_consumption: float = 0.0  # J

    def calculate_overall_quality(self) -> float:
        """Calculate overall quality score."""
        quality_factors = [
            self.part_quality,
            self.dimensional_accuracy,
            self.surface_finish,
        ]

        valid_factors = [f for f in quality_factors if f > 0]
        if valid_factors:
            return sum(valid_factors) / len(valid_factors)
        return 0.0


class ProcessSimulationInterface(ABC):
    """Abstract interface for process simulation tools."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the simulation tool."""
        pass

    @abstractmethod
    def set_material_properties(self, properties: MaterialFlowProperties) -> None:
        """Set material properties for simulation."""
        pass

    @abstractmethod
    def set_process_conditions(self, conditions: ProcessConditions) -> None:
        """Set process conditions."""
        pass

    @abstractmethod
    def run_simulation(self) -> ProcessOutput:
        """Run the process simulation."""
        pass

    @abstractmethod
    def get_field_data(self, field_name: str) -> np.ndarray:
        """Get field data (pressure, temperature, velocity, etc.)."""
        pass


class FluidDynamicsSimulator(ProcessSimulationInterface):
    """Fluid dynamics simulation interface."""

    def __init__(self, solver_type: str = "cfd"):
        self.solver_type = solver_type
        self.material_properties = None
        self.process_conditions = None
        self.mesh = None
        self.velocity_field = None
        self.pressure_field = None
        self.temperature_field = None
        self.viscosity_field = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize CFD simulation."""
        mesh_size = config.get("mesh_size", [50, 50, 50])
        domain_size = config.get("domain_size", [1.0, 1.0, 1.0])

        self.mesh = self._create_mesh(mesh_size, domain_size)

        # Initialize fields
        field_shape = tuple(mesh_size)
        self.velocity_field = np.zeros(field_shape + (3,))  # 3D velocity
        self.pressure_field = np.zeros(field_shape)
        self.temperature_field = np.full(field_shape, 293.15)
        self.viscosity_field = np.full(field_shape, 1000.0)

        logger.info(f"Initialized CFD simulation with mesh {mesh_size}")

    def set_material_properties(self, properties: MaterialFlowProperties) -> None:
        """Set material properties."""
        self.material_properties = properties

        # Update viscosity field
        if self.viscosity_field is not None:
            self.viscosity_field.fill(properties.viscosity)

    def set_process_conditions(self, conditions: ProcessConditions) -> None:
        """Set process conditions."""
        self.process_conditions = conditions

        # Apply boundary conditions
        if conditions.temperature and self.temperature_field is not None:
            self.temperature_field.fill(conditions.temperature)

        if conditions.flow_rate and self.velocity_field is not None:
            # Simplified: uniform velocity in x-direction
            inlet_velocity = self._calculate_inlet_velocity(conditions.flow_rate)
            self.velocity_field[:, :, 0, 0] = inlet_velocity

    def run_simulation(self) -> ProcessOutput:
        """Run CFD simulation."""
        if not self.material_properties or not self.process_conditions:
            raise ChemiAccelError(
                "Material properties and process conditions must be set"
            )

        # Simplified CFD solver (placeholder for real CFD)
        self._solve_momentum_equation()
        self._solve_continuity_equation()
        self._solve_energy_equation()

        # Calculate process outputs
        output = self._calculate_process_output()

        logger.info("CFD simulation completed")
        return output

    def get_field_data(self, field_name: str) -> np.ndarray:
        """Get field data."""
        field_map = {
            "velocity": self.velocity_field,
            "pressure": self.pressure_field,
            "temperature": self.temperature_field,
            "viscosity": self.viscosity_field,
        }

        if field_name not in field_map:
            raise ValueError(f"Unknown field: {field_name}")

        return field_map[field_name].copy()

    def _create_mesh(
        self, mesh_size: List[int], domain_size: List[float]
    ) -> Dict[str, Any]:
        """Create computational mesh."""
        return {
            "size": mesh_size,
            "domain": domain_size,
            "dx": domain_size[0] / mesh_size[0],
            "dy": domain_size[1] / mesh_size[1],
            "dz": domain_size[2] / mesh_size[2],
        }

    def _calculate_inlet_velocity(self, flow_rate: float) -> float:
        """Calculate inlet velocity from flow rate."""
        if not self.material_properties:
            return 0.0

        # Simplified: assume circular cross-section
        diameter = 0.1  # m
        area = np.pi * (diameter / 2) ** 2
        density = self.material_properties.density

        return flow_rate / (density * area)

    def _solve_momentum_equation(self) -> None:
        """Solve momentum equation (simplified)."""
        if self.velocity_field is None or self.pressure_field is None:
            return

        # Simplified momentum solver
        dt = 0.001  # Time step
        dx = self.mesh["dx"]

        # Update velocity field (simplified)
        for i in range(1, self.velocity_field.shape[0] - 1):
            for j in range(1, self.velocity_field.shape[1] - 1):
                for k in range(1, self.velocity_field.shape[2] - 1):
                    # Simplified pressure gradient term
                    dp_dx = (
                        self.pressure_field[i + 1, j, k]
                        - self.pressure_field[i - 1, j, k]
                    ) / (2 * dx)

                    # Viscous term (simplified)
                    viscosity = self.viscosity_field[i, j, k]
                    d2u_dx2 = (
                        self.velocity_field[i + 1, j, k, 0]
                        - 2 * self.velocity_field[i, j, k, 0]
                        + self.velocity_field[i - 1, j, k, 0]
                    ) / (dx**2)

                    # Update velocity
                    density = self.material_properties.density
                    self.velocity_field[i, j, k, 0] += dt * (
                        -dp_dx / density + viscosity * d2u_dx2 / density
                    )

    def _solve_continuity_equation(self) -> None:
        """Solve continuity equation (simplified)."""
        # Simplified: adjust pressure to satisfy continuity
        if self.velocity_field is None or self.pressure_field is None:
            return

        # Calculate velocity divergence
        dx = self.mesh["dx"]
        for i in range(1, self.pressure_field.shape[0] - 1):
            for j in range(1, self.pressure_field.shape[1] - 1):
                for k in range(1, self.pressure_field.shape[2] - 1):
                    div_u = (
                        self.velocity_field[i + 1, j, k, 0]
                        - self.velocity_field[i - 1, j, k, 0]
                    ) / (2 * dx)

                    # Adjust pressure to reduce divergence
                    self.pressure_field[i, j, k] -= 1000.0 * div_u

    def _solve_energy_equation(self) -> None:
        """Solve energy equation (simplified)."""
        if self.temperature_field is None or self.velocity_field is None:
            return

        # Simplified energy equation with convection and diffusion
        dt = 0.001
        dx = self.mesh["dx"]

        thermal_conductivity = self.material_properties.thermal_conductivity
        density = self.material_properties.density
        specific_heat = self.material_properties.specific_heat

        alpha = thermal_conductivity / (density * specific_heat)  # Thermal diffusivity

        for i in range(1, self.temperature_field.shape[0] - 1):
            for j in range(1, self.temperature_field.shape[1] - 1):
                for k in range(1, self.temperature_field.shape[2] - 1):
                    # Convection term
                    dT_dx = (
                        self.temperature_field[i + 1, j, k]
                        - self.temperature_field[i - 1, j, k]
                    ) / (2 * dx)
                    convection = self.velocity_field[i, j, k, 0] * dT_dx

                    # Diffusion term
                    d2T_dx2 = (
                        self.temperature_field[i + 1, j, k]
                        - 2 * self.temperature_field[i, j, k]
                        + self.temperature_field[i - 1, j, k]
                    ) / (dx**2)
                    diffusion = alpha * d2T_dx2

                    # Update temperature
                    self.temperature_field[i, j, k] += dt * (-convection + diffusion)

    def _calculate_process_output(self) -> ProcessOutput:
        """Calculate process output from simulation results."""
        # Calculate quality metrics based on simulation results
        avg_velocity = np.mean(self.velocity_field[:, :, :, 0])
        max_velocity = np.max(self.velocity_field[:, :, :, 0])
        avg_temperature = np.mean(self.temperature_field)
        max_pressure = np.max(self.pressure_field)

        # Quality assessment (simplified)
        velocity_uniformity = (
            1.0 - (max_velocity - avg_velocity) / max_velocity
            if max_velocity > 0
            else 0.0
        )
        temperature_uniformity = (
            1.0 - np.std(self.temperature_field) / avg_temperature
            if avg_temperature > 0
            else 0.0
        )

        output = ProcessOutput(
            part_quality=velocity_uniformity * 0.5 + temperature_uniformity * 0.5,
            dimensional_accuracy=velocity_uniformity,
            surface_finish=temperature_uniformity,
            mechanical_properties={
                "flow_stress": max_pressure * 0.1,
                "orientation_factor": velocity_uniformity,
            },
            morphology={
                "flow_pattern": "uniform"
                if velocity_uniformity > 0.8
                else "non_uniform",
                "temperature_distribution": "uniform"
                if temperature_uniformity > 0.8
                else "non_uniform",
            },
            processing_time=self.process_conditions.cycle_time or 60.0,
            energy_consumption=self._calculate_energy_consumption(),
        )

        return output

    def _calculate_energy_consumption(self) -> float:
        """Calculate energy consumption."""
        # Simplified energy calculation
        if not self.process_conditions:
            return 0.0

        # Power for heating
        heating_power = 10000.0  # W (simplified)

        # Power for pumping
        avg_pressure = np.mean(self.pressure_field)
        flow_rate = self.process_conditions.flow_rate or 0.1
        pumping_power = avg_pressure * flow_rate / 1000.0  # Simplified

        total_power = heating_power + pumping_power
        processing_time = self.process_conditions.cycle_time or 60.0

        return total_power * processing_time


class HeatTransferSimulator(ProcessSimulationInterface):
    """Heat transfer simulation interface."""

    def __init__(self, solver_type: str = "fem"):
        self.solver_type = solver_type
        self.material_properties = None
        self.process_conditions = None
        self.mesh = None
        self.temperature_field = None
        self.heat_flux_field = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize heat transfer simulation."""
        mesh_size = config.get("mesh_size", [30, 30, 30])
        domain_size = config.get("domain_size", [0.1, 0.1, 0.1])

        self.mesh = self._create_mesh(mesh_size, domain_size)

        # Initialize fields
        field_shape = tuple(mesh_size)
        self.temperature_field = np.full(field_shape, 293.15)
        self.heat_flux_field = np.zeros(field_shape + (3,))  # 3D heat flux

        logger.info(f"Initialized heat transfer simulation with mesh {mesh_size}")

    def set_material_properties(self, properties: MaterialFlowProperties) -> None:
        """Set material properties."""
        self.material_properties = properties

    def set_process_conditions(self, conditions: ProcessConditions) -> None:
        """Set process conditions."""
        self.process_conditions = conditions

        # Apply thermal boundary conditions
        if conditions.temperature and self.temperature_field is not None:
            # Set boundary temperature
            self.temperature_field[0, :, :] = conditions.temperature  # Left boundary
            self.temperature_field[-1, :, :] = (
                conditions.mold_temperature or 293.15
            )  # Right boundary

    def run_simulation(self) -> ProcessOutput:
        """Run heat transfer simulation."""
        if not self.material_properties or not self.process_conditions:
            raise ChemiAccelError(
                "Material properties and process conditions must be set"
            )

        # Solve heat equation
        self._solve_heat_equation()
        self._calculate_heat_flux()

        # Calculate process outputs
        output = self._calculate_thermal_output()

        logger.info("Heat transfer simulation completed")
        return output

    def get_field_data(self, field_name: str) -> np.ndarray:
        """Get field data."""
        field_map = {
            "temperature": self.temperature_field,
            "heat_flux": self.heat_flux_field,
        }

        if field_name not in field_map:
            raise ValueError(f"Unknown field: {field_name}")

        return field_map[field_name].copy()

    def _create_mesh(
        self, mesh_size: List[int], domain_size: List[float]
    ) -> Dict[str, Any]:
        """Create computational mesh."""
        return {
            "size": mesh_size,
            "domain": domain_size,
            "dx": domain_size[0] / mesh_size[0],
            "dy": domain_size[1] / mesh_size[1],
            "dz": domain_size[2] / mesh_size[2],
        }

    def _solve_heat_equation(self) -> None:
        """Solve heat equation using finite differences."""
        if self.temperature_field is None:
            return

        # Heat equation: ∂T/∂t = α ∇²T
        dt = 0.001  # Time step
        dx = self.mesh["dx"]

        thermal_conductivity = self.material_properties.thermal_conductivity
        density = self.material_properties.density
        specific_heat = self.material_properties.specific_heat

        alpha = thermal_conductivity / (density * specific_heat)  # Thermal diffusivity

        # Stability condition
        dt = min(dt, 0.25 * dx**2 / alpha)

        # Number of time steps
        total_time = self.process_conditions.cycle_time or 60.0
        num_steps = int(total_time / dt)

        for step in range(num_steps):
            T_old = self.temperature_field.copy()

            # Interior points
            for i in range(1, self.temperature_field.shape[0] - 1):
                for j in range(1, self.temperature_field.shape[1] - 1):
                    for k in range(1, self.temperature_field.shape[2] - 1):
                        # Laplacian
                        d2T_dx2 = (
                            T_old[i + 1, j, k] - 2 * T_old[i, j, k] + T_old[i - 1, j, k]
                        ) / dx**2
                        d2T_dy2 = (
                            T_old[i, j + 1, k] - 2 * T_old[i, j, k] + T_old[i, j - 1, k]
                        ) / dx**2
                        d2T_dz2 = (
                            T_old[i, j, k + 1] - 2 * T_old[i, j, k] + T_old[i, j, k - 1]
                        ) / dx**2

                        laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2

                        self.temperature_field[i, j, k] = (
                            T_old[i, j, k] + alpha * dt * laplacian
                        )

    def _calculate_heat_flux(self) -> None:
        """Calculate heat flux field."""
        if self.temperature_field is None or self.heat_flux_field is None:
            return

        dx = self.mesh["dx"]
        thermal_conductivity = self.material_properties.thermal_conductivity

        # Calculate heat flux: q = -k ∇T
        for i in range(1, self.temperature_field.shape[0] - 1):
            for j in range(1, self.temperature_field.shape[1] - 1):
                for k in range(1, self.temperature_field.shape[2] - 1):
                    # Temperature gradients
                    dT_dx = (
                        self.temperature_field[i + 1, j, k]
                        - self.temperature_field[i - 1, j, k]
                    ) / (2 * dx)
                    dT_dy = (
                        self.temperature_field[i, j + 1, k]
                        - self.temperature_field[i, j - 1, k]
                    ) / (2 * dx)
                    dT_dz = (
                        self.temperature_field[i, j, k + 1]
                        - self.temperature_field[i, j, k - 1]
                    ) / (2 * dx)

                    # Heat flux components
                    self.heat_flux_field[i, j, k, 0] = -thermal_conductivity * dT_dx
                    self.heat_flux_field[i, j, k, 1] = -thermal_conductivity * dT_dy
                    self.heat_flux_field[i, j, k, 2] = -thermal_conductivity * dT_dz

    def _calculate_thermal_output(self) -> ProcessOutput:
        """Calculate thermal process output."""
        # Temperature uniformity
        avg_temp = np.mean(self.temperature_field)
        temp_std = np.std(self.temperature_field)
        temperature_uniformity = 1.0 - temp_std / avg_temp if avg_temp > 0 else 0.0

        # Cooling rate analysis
        cooling_rate = self._estimate_cooling_rate()

        # Quality based on thermal uniformity
        thermal_quality = temperature_uniformity

        output = ProcessOutput(
            part_quality=thermal_quality,
            dimensional_accuracy=temperature_uniformity,
            surface_finish=0.8
            if cooling_rate < 10.0
            else 0.6,  # Slower cooling = better surface
            mechanical_properties={
                "thermal_stress": temp_std * 1e6,  # Simplified thermal stress
                "crystallinity": self._estimate_crystallinity(cooling_rate),
            },
            morphology={
                "cooling_pattern": "uniform"
                if temperature_uniformity > 0.8
                else "non_uniform",
                "thermal_gradient": temp_std,
            },
            processing_time=self.process_conditions.cycle_time or 60.0,
            energy_consumption=self._calculate_thermal_energy(),
        )

        return output

    def _estimate_cooling_rate(self) -> float:
        """Estimate average cooling rate."""
        if not self.process_conditions:
            return 1.0

        initial_temp = self.process_conditions.melt_temperature or 473.15
        final_temp = self.process_conditions.mold_temperature or 293.15
        cooling_time = self.process_conditions.cycle_time or 60.0

        return (initial_temp - final_temp) / cooling_time

    def _estimate_crystallinity(self, cooling_rate: float) -> float:
        """Estimate crystallinity based on cooling rate."""
        # Simplified Avrami-like model
        # Slower cooling = higher crystallinity
        max_crystallinity = 0.6
        rate_factor = np.exp(-cooling_rate / 5.0)
        return max_crystallinity * rate_factor

    def _calculate_thermal_energy(self) -> float:
        """Calculate thermal energy consumption."""
        if not self.process_conditions:
            return 0.0

        # Energy for heating
        mass = 1.0  # kg (simplified)
        specific_heat = self.material_properties.specific_heat
        temp_rise = (self.process_conditions.melt_temperature or 473.15) - 293.15

        heating_energy = mass * specific_heat * temp_rise

        # Energy for cooling
        cooling_energy = heating_energy * 0.3  # Simplified

        return heating_energy + cooling_energy


class ManufacturingDigitalTwin:
    """
    Enhanced digital twin with manufacturing process integration.

    Extends the multi-scale framework to include manufacturing process
    parameters and simulation interfaces.
    """

    def __init__(self, multiscale_framework: MultiScaleFramework):
        self.multiscale_framework = multiscale_framework
        self.process_simulators = {}
        self.process_conditions = None
        self.material_flow_properties = MaterialFlowProperties()
        self.process_history = []

    def add_process_simulator(
        self,
        process_type: ProcessType,
        simulator: ProcessSimulationInterface,
        simulator_id: str = None,
    ) -> str:
        """Add a process simulation interface."""
        if simulator_id is None:
            simulator_id = f"{process_type.value}_simulator"

        self.process_simulators[simulator_id] = {
            "simulator": simulator,
            "process_type": process_type,
            "active": True,
        }

        logger.info(f"Added {process_type.value} simulator: {simulator_id}")
        return simulator_id

    def set_process_conditions(self, conditions: ProcessConditions) -> None:
        """Set manufacturing process conditions."""
        self.process_conditions = conditions

        # Update material properties based on process conditions
        if conditions.temperature:
            self.material_flow_properties.update_with_temperature(
                conditions.temperature
            )

        # Propagate conditions to all simulators
        for simulator_info in self.process_simulators.values():
            if simulator_info["active"]:
                simulator_info["simulator"].set_process_conditions(conditions)
                simulator_info["simulator"].set_material_properties(
                    self.material_flow_properties
                )

        logger.info("Process conditions updated")

    def run_manufacturing_simulation(
        self, process_type: ProcessType, simulation_config: Dict[str, Any] = None
    ) -> ProcessOutput:
        """Run manufacturing process simulation."""
        if not self.process_conditions:
            raise ChemiAccelError("Process conditions must be set before simulation")

        # Find appropriate simulator
        simulator_id = None
        for sid, sim_info in self.process_simulators.items():
            if sim_info["process_type"] == process_type and sim_info["active"]:
                simulator_id = sid
                break

        if not simulator_id:
            raise ChemiAccelError(
                f"No active simulator found for process type: {process_type.value}"
            )

        simulator = self.process_simulators[simulator_id]["simulator"]

        # Initialize simulator if needed
        if simulation_config:
            simulator.initialize(simulation_config)

        # Update material properties from multi-scale models
        self._update_material_properties_from_multiscale()
        simulator.set_material_properties(self.material_flow_properties)

        # Run simulation
        output = simulator.run_simulation()

        # Store results
        self.process_history.append(
            {
                "process_type": process_type.value,
                "conditions": self.process_conditions.to_dict(),
                "output": output,
                "timestamp": len(self.process_history),
            }
        )

        logger.info(f"Manufacturing simulation completed for {process_type.value}")
        return output

    def run_coupled_simulation(
        self,
        multiscale_dt: float = 0.001,
        multiscale_steps: int = 100,
        process_type: ProcessType = ProcessType.EXTRUSION,
    ) -> Tuple[List[Dict[str, Any]], ProcessOutput]:
        """Run coupled multi-scale and manufacturing simulation."""

        # Run multi-scale simulation
        multiscale_results = []
        for step in range(multiscale_steps):
            ms_result = self.multiscale_framework.step(multiscale_dt)
            multiscale_results.append(ms_result)

            # Update material properties based on multi-scale results
            if step % 10 == 0:  # Update every 10 steps
                self._update_material_properties_from_multiscale()

        # Run manufacturing process simulation
        process_output = self.run_manufacturing_simulation(process_type)

        # Feedback manufacturing results to multi-scale models
        self._feedback_manufacturing_results(process_output)

        return multiscale_results, process_output

    def optimize_process_conditions(
        self,
        process_type: ProcessType,
        target_properties: Dict[str, float],
        parameter_ranges: Dict[str, Tuple[float, float]],
        num_iterations: int = 20,
    ) -> Tuple[ProcessConditions, ProcessOutput]:
        """Optimize process conditions to achieve target properties."""

        best_conditions = None
        best_output = None
        best_score = float("-inf")

        for iteration in range(num_iterations):
            # Generate random conditions within ranges
            test_conditions = self._generate_random_conditions(parameter_ranges)
            self.set_process_conditions(test_conditions)

            # Run simulation
            try:
                output = self.run_manufacturing_simulation(process_type)

                # Calculate fitness score
                score = self._calculate_fitness_score(output, target_properties)

                if score > best_score:
                    best_score = score
                    best_conditions = test_conditions
                    best_output = output

            except Exception as e:
                logger.warning(f"Simulation failed in iteration {iteration}: {e}")
                continue

        logger.info(f"Optimization completed. Best score: {best_score:.3f}")
        return best_conditions, best_output

    def get_process_field_data(self, simulator_id: str, field_name: str) -> np.ndarray:
        """Get field data from process simulator."""
        if simulator_id not in self.process_simulators:
            raise ValueError(f"Simulator {simulator_id} not found")

        simulator = self.process_simulators[simulator_id]["simulator"]
        return simulator.get_field_data(field_name)

    def _update_material_properties_from_multiscale(self) -> None:
        """Update material flow properties from multi-scale models."""
        # Get properties from multi-scale framework
        ms_properties = self.multiscale_framework.get_material_properties()

        # Update flow properties based on multi-scale results
        for model_id, properties in ms_properties.items():
            if "density" in properties:
                self.material_flow_properties.density = properties["density"]

            if "thermal_conductivity" in properties:
                self.material_flow_properties.thermal_conductivity = properties[
                    "thermal_conductivity"
                ]

            if "elastic_modulus" in properties:
                self.material_flow_properties.elastic_modulus = properties[
                    "elastic_modulus"
                ]

            if "specific_heat" in properties:
                self.material_flow_properties.specific_heat = properties[
                    "specific_heat"
                ]

        logger.debug("Material properties updated from multi-scale models")

    def _feedback_manufacturing_results(self, process_output: ProcessOutput) -> None:
        """Feedback manufacturing results to multi-scale models."""
        # Update multi-scale models based on manufacturing results

        # Example: Update material state based on residual stress
        if process_output.residual_stress > 0:
            for model_id in self.multiscale_framework.models:
                try:
                    state = self.multiscale_framework.get_model_state(model_id)

                    # Add residual stress effects
                    if "stress" in state:
                        state["stress"] += process_output.residual_stress
                    else:
                        state["residual_stress"] = process_output.residual_stress

                    self.multiscale_framework.set_model_state(model_id, state)

                except Exception as e:
                    logger.warning(f"Failed to update model {model_id}: {e}")

        logger.debug("Manufacturing results fed back to multi-scale models")

    def _generate_random_conditions(
        self, parameter_ranges: Dict[str, Tuple[float, float]]
    ) -> ProcessConditions:
        """Generate random process conditions within specified ranges."""
        conditions = ProcessConditions()

        for param, (min_val, max_val) in parameter_ranges.items():
            value = np.random.uniform(min_val, max_val)

            if hasattr(conditions, param):
                setattr(conditions, param, value)

        return conditions

    def _calculate_fitness_score(
        self, output: ProcessOutput, target_properties: Dict[str, float]
    ) -> float:
        """Calculate fitness score for optimization."""
        score = 0.0
        num_targets = 0

        # Compare output properties with targets
        output_dict = {
            "part_quality": output.part_quality,
            "dimensional_accuracy": output.dimensional_accuracy,
            "surface_finish": output.surface_finish,
            "crystallinity": output.crystallinity,
            "processing_time": output.processing_time,
            "energy_consumption": output.energy_consumption,
        }

        for prop_name, target_value in target_properties.items():
            if prop_name in output_dict:
                actual_value = output_dict[prop_name]

                # Calculate normalized error
                if target_value > 0:
                    error = abs(actual_value - target_value) / target_value
                    score += max(0, 1.0 - error)  # Higher score for lower error
                    num_targets += 1

        # Add overall quality bonus
        if output.part_quality > 0:
            score += output.part_quality * 0.5

        return score / max(num_targets, 1)

    def get_manufacturing_summary(self) -> Dict[str, Any]:
        """Get summary of manufacturing simulation results."""
        if not self.process_history:
            return {"message": "No manufacturing simulations run yet"}

        latest_result = self.process_history[-1]

        return {
            "total_simulations": len(self.process_history),
            "latest_process": latest_result["process_type"],
            "latest_quality": latest_result["output"].part_quality,
            "latest_processing_time": latest_result["output"].processing_time,
            "average_quality": np.mean(
                [r["output"].part_quality for r in self.process_history]
            ),
            "available_simulators": list(self.process_simulators.keys()),
            "current_conditions": self.process_conditions.to_dict()
            if self.process_conditions
            else None,
        }

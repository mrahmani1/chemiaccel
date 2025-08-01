"""
GPU-Accelerated Crystallization Kinetics Modeling

This module implements advanced crystallization kinetics models combining:
- GPU-accelerated multi-scale modeling
- Classical nucleation theory
- Population balance models
- Molecular dynamics insights integration
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.integrate as cp_integrate

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cp_integrate = None
    CUPY_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from .core import (
    CrystallizationCore,
    KineticsParameters,
    ProcessConditions,
    RealTimeData,
    SimulationConfig,
)

logger = logging.getLogger(__name__)


class NucleationModel(ABC):
    """Abstract base class for nucleation models."""

    @abstractmethod
    def calculate_nucleation_rate(
        self, supersaturation: float, temperature: float
    ) -> float:
        """Calculate nucleation rate."""
        pass


class ClassicalNucleationTheory(NucleationModel):
    """
    Classical nucleation theory implementation.

    Implements the classical nucleation theory with corrections for:
    - Surface energy effects
    - Temperature dependence
    - Supersaturation effects
    """

    def __init__(self, surface_energy: float = 0.01, molecular_volume: float = 1e-28):
        """
        Initialize CNT model.

        Args:
            surface_energy: Crystal-solution interfacial energy (J/m²)
            molecular_volume: Molecular volume in crystal (m³)
        """
        self.surface_energy = surface_energy
        self.molecular_volume = molecular_volume
        self.k_boltzmann = 1.38064852e-23  # J/K

    def calculate_nucleation_rate(
        self, supersaturation: float, temperature: float
    ) -> float:
        """
        Calculate nucleation rate using CNT.

        Args:
            supersaturation: Supersaturation ratio
            temperature: Temperature (K)

        Returns:
            Nucleation rate (nuclei/m³/s)
        """
        if supersaturation <= 1.0:
            return 0.0

        # Critical radius
        ln_s = np.log(supersaturation)
        r_critical = (
            2
            * self.surface_energy
            * self.molecular_volume
            / (self.k_boltzmann * temperature * ln_s)
        )

        # Critical work of nucleation
        w_critical = (
            16
            * np.pi
            * self.surface_energy**3
            * self.molecular_volume**2
            / (3 * (self.k_boltzmann * temperature * ln_s) ** 2)
        )

        # Pre-exponential factor (simplified)
        a_factor = 1e20  # nuclei/m³/s

        # Nucleation rate
        j_nucleation = a_factor * np.exp(-w_critical / (self.k_boltzmann * temperature))

        return j_nucleation


class GrowthModel(ABC):
    """Abstract base class for crystal growth models."""

    @abstractmethod
    def calculate_growth_rate(
        self, supersaturation: float, temperature: float, size: float
    ) -> float:
        """Calculate growth rate."""
        pass


class DiffusionLimitedGrowth(GrowthModel):
    """
    Diffusion-limited crystal growth model.

    Implements growth kinetics based on:
    - Mass transfer limitations
    - Surface integration kinetics
    - Size-dependent effects
    """

    def __init__(
        self, growth_constant: float = 1e-8, diffusion_coefficient: float = 1e-9
    ):
        """
        Initialize growth model.

        Args:
            growth_constant: Growth rate constant (m/s)
            diffusion_coefficient: Solute diffusion coefficient (m²/s)
        """
        self.growth_constant = growth_constant
        self.diffusion_coefficient = diffusion_coefficient

    def calculate_growth_rate(
        self, supersaturation: float, temperature: float, size: float
    ) -> float:
        """
        Calculate growth rate.

        Args:
            supersaturation: Supersaturation ratio
            temperature: Temperature (K)
            size: Crystal size (m)

        Returns:
            Growth rate (m/s)
        """
        if supersaturation <= 1.0:
            return 0.0

        # Temperature dependence (Arrhenius)
        activation_energy = 30000.0  # J/mol
        r_gas = 8.314  # J/mol/K
        temperature_factor = np.exp(-activation_energy / (r_gas * temperature))

        # Supersaturation driving force
        driving_force = supersaturation - 1.0

        # Size-dependent mass transfer coefficient
        mass_transfer_coeff = (
            self.diffusion_coefficient / size
            if size > 0
            else self.diffusion_coefficient / 1e-6
        )

        # Growth rate
        growth_rate = (
            self.growth_constant
            * temperature_factor
            * driving_force
            * mass_transfer_coeff
        )

        return growth_rate


class PopulationBalanceModel:
    """
    Population balance model for crystal size distribution evolution.

    Solves the population balance equation with:
    - Nucleation
    - Growth
    - Agglomeration
    - Breakage
    """

    def __init__(self, size_grid: np.ndarray, kinetics_params: KineticsParameters):
        """
        Initialize population balance model.

        Args:
            size_grid: Size grid for discretization (m)
            kinetics_params: Kinetics parameters
        """
        self.size_grid = size_grid
        self.kinetics_params = kinetics_params
        self.nucleation_model = ClassicalNucleationTheory(
            kinetics_params.surface_energy, kinetics_params.molecular_volume
        )
        self.growth_model = DiffusionLimitedGrowth(
            kinetics_params.growth_rate_constant, kinetics_params.diffusion_coefficient
        )

    def calculate_birth_rate(self, size: float, number_density: np.ndarray) -> float:
        """Calculate birth rate due to agglomeration."""
        birth_rate = 0.0

        for i, size_i in enumerate(self.size_grid):
            if size_i >= size:
                break
            size_j = size - size_i
            j = np.searchsorted(self.size_grid, size_j)

            if j < len(self.size_grid) and j < len(number_density):
                kernel = self.kinetics_params.agglomeration_kernel * (size_i + size_j)
                birth_rate += 0.5 * kernel * number_density[i] * number_density[j]

        return birth_rate

    def calculate_death_rate(
        self, size: float, number_density: np.ndarray, size_idx: int
    ) -> float:
        """Calculate death rate due to agglomeration."""
        death_rate = 0.0

        for j, size_j in enumerate(self.size_grid):
            if j < len(number_density):
                kernel = self.kinetics_params.agglomeration_kernel * (size + size_j)
                death_rate += kernel * number_density[size_idx] * number_density[j]

        return death_rate

    def solve_population_balance(
        self,
        initial_distribution: np.ndarray,
        time_span: Tuple[float, float],
        conditions: ProcessConditions,
        n_steps: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve population balance equation.

        Args:
            initial_distribution: Initial number density distribution
            time_span: Time span for simulation (s)
            conditions: Process conditions
            n_steps: Number of time steps

        Returns:
            Tuple of (time_points, distribution_evolution)
        """
        time_points = np.linspace(time_span[0], time_span[1], n_steps)
        dt = (time_span[1] - time_span[0]) / n_steps

        distribution_evolution = np.zeros((n_steps, len(self.size_grid)))
        distribution_evolution[0] = initial_distribution.copy()

        for t_idx in range(1, n_steps):
            current_time = time_points[t_idx]
            current_dist = distribution_evolution[t_idx - 1].copy()

            # Get current temperature and supersaturation
            temperature = self._get_temperature_at_time(current_time, conditions)
            supersaturation = self._get_supersaturation_at_time(
                current_time, conditions
            )

            # Calculate rates
            new_dist = current_dist.copy()

            for i, size in enumerate(self.size_grid):
                # Nucleation (only for smallest size)
                if i == 0:
                    nucleation_rate = self.nucleation_model.calculate_nucleation_rate(
                        supersaturation, temperature
                    )
                    new_dist[i] += nucleation_rate * dt

                # Growth
                growth_rate = self.growth_model.calculate_growth_rate(
                    supersaturation, temperature, size
                )

                # Simple growth discretization (upwind scheme)
                if i > 0 and growth_rate > 0:
                    size_diff = self.size_grid[i] - self.size_grid[i - 1]
                    flux_in = growth_rate * current_dist[i - 1] / size_diff
                    flux_out = growth_rate * current_dist[i] / size_diff
                    new_dist[i] += (flux_in - flux_out) * dt

                # Agglomeration
                birth_rate = self.calculate_birth_rate(size, current_dist)
                death_rate = self.calculate_death_rate(size, current_dist, i)
                new_dist[i] += (birth_rate - death_rate) * dt

                # Breakage (simplified)
                breakage_rate = self.kinetics_params.breakage_kernel * size**2
                new_dist[i] -= breakage_rate * current_dist[i] * dt

                # Ensure non-negative
                new_dist[i] = max(0.0, new_dist[i])

            distribution_evolution[t_idx] = new_dist

        return time_points, distribution_evolution

    def _get_temperature_at_time(
        self, time: float, conditions: ProcessConditions
    ) -> float:
        """Interpolate temperature at given time."""
        if not conditions.temperature_profile:
            return 298.15  # Default room temperature

        times = [point[1] for point in conditions.temperature_profile]
        temps = [point[0] for point in conditions.temperature_profile]

        return np.interp(time, times, temps)

    def _get_supersaturation_at_time(
        self, time: float, conditions: ProcessConditions
    ) -> float:
        """Interpolate supersaturation at given time."""
        if not conditions.supersaturation_profile:
            return 1.5  # Default moderate supersaturation

        times = [point[1] for point in conditions.supersaturation_profile]
        supersats = [point[0] for point in conditions.supersaturation_profile]

        return np.interp(time, times, supersats)


class GPUAcceleratedKinetics:
    """
    GPU-accelerated implementation of crystallization kinetics.

    Uses CuPy for NVIDIA GPU acceleration of kinetics calculations.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize GPU-accelerated kinetics.

        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

        if self.use_gpu:
            logger.info("GPU acceleration enabled for kinetics modeling")
        else:
            logger.info("Using CPU for kinetics modeling")

    def calculate_nucleation_rates_batch(
        self,
        supersaturations: Union[np.ndarray, Any],
        temperatures: Union[np.ndarray, Any],
        kinetics_params: KineticsParameters,
    ) -> Union[np.ndarray, Any]:
        """
        Calculate nucleation rates for batch of conditions.

        Args:
            supersaturations: Array of supersaturation values
            temperatures: Array of temperature values (K)
            kinetics_params: Kinetics parameters

        Returns:
            Array of nucleation rates
        """
        if self.use_gpu:
            supersaturations = cp.asarray(supersaturations)
            temperatures = cp.asarray(temperatures)

        # Constants
        k_b = 1.38064852e-23
        surface_energy = kinetics_params.surface_energy
        molecular_volume = kinetics_params.molecular_volume

        # Avoid log of values <= 1
        mask = supersaturations > 1.0
        ln_s = self.xp.log(self.xp.maximum(supersaturations, 1.001))

        # Critical work of nucleation
        w_critical = (16 * self.xp.pi * surface_energy**3 * molecular_volume**2) / (
            3 * (k_b * temperatures * ln_s) ** 2
        )

        # Nucleation rate
        a_factor = kinetics_params.nucleation_rate_constant
        nucleation_rates = a_factor * self.xp.exp(-w_critical / (k_b * temperatures))

        # Apply mask
        nucleation_rates = self.xp.where(mask, nucleation_rates, 0.0)

        if self.use_gpu:
            return cp.asnumpy(nucleation_rates)
        else:
            return nucleation_rates

    def calculate_growth_rates_batch(
        self,
        supersaturations: Union[np.ndarray, Any],
        temperatures: Union[np.ndarray, Any],
        sizes: Union[np.ndarray, Any],
        kinetics_params: KineticsParameters,
    ) -> Union[np.ndarray, Any]:
        """
        Calculate growth rates for batch of conditions.

        Args:
            supersaturations: Array of supersaturation values
            temperatures: Array of temperature values (K)
            sizes: Array of crystal sizes (m)
            kinetics_params: Kinetics parameters

        Returns:
            Array of growth rates
        """
        if self.use_gpu:
            supersaturations = cp.asarray(supersaturations)
            temperatures = cp.asarray(temperatures)
            sizes = cp.asarray(sizes)

        # Temperature dependence
        activation_energy = kinetics_params.activation_energies.get("growth", 30000.0)
        r_gas = 8.314
        temperature_factor = self.xp.exp(-activation_energy / (r_gas * temperatures))

        # Driving force
        driving_force = self.xp.maximum(supersaturations - 1.0, 0.0)

        # Mass transfer coefficient
        diffusion_coeff = kinetics_params.diffusion_coefficient
        mass_transfer_coeff = diffusion_coeff / self.xp.maximum(sizes, 1e-9)

        # Growth rate
        growth_rates = (
            kinetics_params.growth_rate_constant
            * temperature_factor
            * driving_force
            * mass_transfer_coeff
        )

        if self.use_gpu:
            return cp.asnumpy(growth_rates)
        else:
            return growth_rates


class MultiScaleKineticsModel(CrystallizationCore):
    """
    Multi-scale crystallization kinetics model.

    Combines:
    - Atomistic insights from MD simulations
    - Mesoscale population balance modeling
    - Macroscale process dynamics
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize multi-scale model.

        Args:
            config: Simulation configuration
        """
        super().__init__(config)
        self.population_balance = None
        self.gpu_kinetics = None
        self.kinetics_params = KineticsParameters()

    def initialize(self) -> None:
        """Initialize the multi-scale model."""
        # Create size grid
        size_min = 1e-9  # 1 nm
        size_max = 1e-3  # 1 mm
        n_sizes = 100
        self.size_grid = np.logspace(np.log10(size_min), np.log10(size_max), n_sizes)

        # Initialize population balance model
        self.population_balance = PopulationBalanceModel(
            self.size_grid, self.kinetics_params
        )

        # Initialize GPU acceleration
        self.gpu_kinetics = GPUAcceleratedKinetics(
            use_gpu=self.config.gpu_acceleration and self.config.backend == "cuda"
        )

        self._initialized = True
        logger.info("Multi-scale kinetics model initialized")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        required_keys = ["conditions", "initial_distribution"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required input: {key}")

        conditions = kwargs["conditions"]
        if not isinstance(conditions, ProcessConditions):
            raise TypeError("conditions must be ProcessConditions instance")

        initial_distribution = kwargs["initial_distribution"]
        if not isinstance(initial_distribution, np.ndarray):
            raise TypeError("initial_distribution must be numpy array")

        if len(initial_distribution) != len(self.size_grid):
            raise ValueError("initial_distribution length must match size_grid")

        return True

    def simulate_crystallization(
        self,
        conditions: ProcessConditions,
        initial_distribution: Optional[np.ndarray] = None,
        md_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Simulate crystallization process using multi-scale model.

        Args:
            conditions: Process conditions
            initial_distribution: Initial crystal size distribution
            md_insights: Molecular dynamics simulation insights

        Returns:
            Simulation results dictionary
        """
        if not self.is_initialized():
            self.initialize()

        # Default initial distribution (delta function at nucleation size)
        if initial_distribution is None:
            initial_distribution = np.zeros(len(self.size_grid))
            initial_distribution[0] = 1e10  # Initial nuclei concentration

        self.validate_inputs(
            conditions=conditions, initial_distribution=initial_distribution
        )

        # Incorporate MD insights if available
        if md_insights:
            self._incorporate_md_insights(md_insights)

        # Solve population balance
        (
            time_points,
            distribution_evolution,
        ) = self.population_balance.solve_population_balance(
            initial_distribution,
            self.config.time_span,
            conditions,
            self.config.time_steps,
        )

        # Calculate derived properties
        results = self._calculate_derived_properties(
            time_points, distribution_evolution, conditions
        )

        # Store simulation data
        self._simulation_data = {
            "time_points": time_points,
            "distribution_evolution": distribution_evolution,
            "size_grid": self.size_grid,
            "conditions": conditions,
            "results": results,
        }

        return results

    def _incorporate_md_insights(self, md_insights: Dict[str, Any]) -> None:
        """Incorporate molecular dynamics insights into kinetics parameters."""
        if "surface_energy" in md_insights:
            self.kinetics_params.surface_energy = md_insights["surface_energy"]

        if "molecular_volume" in md_insights:
            self.kinetics_params.molecular_volume = md_insights["molecular_volume"]

        if "diffusion_coefficient" in md_insights:
            self.kinetics_params.diffusion_coefficient = md_insights[
                "diffusion_coefficient"
            ]

        if "activation_energies" in md_insights:
            self.kinetics_params.activation_energies.update(
                md_insights["activation_energies"]
            )

        logger.info("Incorporated MD insights into kinetics parameters")

    def _calculate_derived_properties(
        self,
        time_points: np.ndarray,
        distribution_evolution: np.ndarray,
        conditions: ProcessConditions,
    ) -> Dict[str, Any]:
        """Calculate derived properties from population balance solution."""
        results = {}

        # Calculate moments of distribution
        results["total_number"] = np.trapz(
            distribution_evolution, self.size_grid, axis=1
        )
        results["mean_size"] = np.array(
            [
                np.trapz(dist * self.size_grid, self.size_grid)
                / max(1e-10, np.trapz(dist, self.size_grid))
                for dist in distribution_evolution
            ]
        )

        # Calculate mass-based properties
        masses = (
            (4 / 3) * np.pi * (self.size_grid / 2) ** 3 * 1000
        )  # Assuming density ~ 1000 kg/m³
        total_mass = np.array(
            [np.trapz(dist * masses, self.size_grid) for dist in distribution_evolution]
        )
        results["total_mass"] = total_mass
        results["yield_evolution"] = total_mass / total_mass[-1] * 100  # Percentage

        # Calculate size distribution statistics
        results["d10"] = []
        results["d50"] = []
        results["d90"] = []

        for dist in distribution_evolution:
            cumulative = np.cumsum(dist * np.gradient(self.size_grid))
            cumulative_normalized = cumulative / cumulative[-1]

            results["d10"].append(np.interp(0.1, cumulative_normalized, self.size_grid))
            results["d50"].append(np.interp(0.5, cumulative_normalized, self.size_grid))
            results["d90"].append(np.interp(0.9, cumulative_normalized, self.size_grid))

        results["d10"] = np.array(results["d10"])
        results["d50"] = np.array(results["d50"])
        results["d90"] = np.array(results["d90"])

        # Calculate supersaturation evolution
        results["supersaturation_evolution"] = np.array(
            [
                self.population_balance._get_supersaturation_at_time(t, conditions)
                for t in time_points
            ]
        )

        return results


class CrystallizationKineticsModel(MultiScaleKineticsModel):
    """
    Main crystallization kinetics model class.

    Provides high-level interface for crystallization kinetics modeling
    with GPU acceleration and multi-scale capabilities.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize crystallization kinetics model.

        Args:
            config: Simulation configuration
        """
        super().__init__(config)

    def predict_crystallization_outcome(
        self,
        conditions: ProcessConditions,
        compound_properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Predict crystallization outcomes for given conditions.

        Args:
            conditions: Process conditions
            compound_properties: Compound-specific properties

        Returns:
            Predicted crystallization outcomes
        """
        if not self.is_initialized():
            self.initialize()

        # Update kinetics parameters based on compound properties
        if compound_properties:
            self._update_kinetics_from_properties(compound_properties)

        # Run simulation
        results = self.simulate_crystallization(conditions)

        # Extract key outcomes
        final_distribution = results["results"]["total_mass"][-1]
        final_mean_size = results["results"]["mean_size"][-1]
        final_yield = results["results"]["yield_evolution"][-1]

        outcomes = {
            "final_yield": final_yield,
            "final_mean_size": final_mean_size,
            "final_d50": results["results"]["d50"][-1],
            "size_distribution_width": results["results"]["d90"][-1]
            - results["results"]["d10"][-1],
            "crystallization_time": self.config.time_span[1] - self.config.time_span[0],
            "process_efficiency": final_yield
            / (self.config.time_span[1] / 3600),  # yield per hour
            "quality_metrics": {
                "uniformity": 1.0
                / (
                    1.0
                    + (results["results"]["d90"][-1] / results["results"]["d10"][-1])
                ),
                "productivity": final_distribution / (self.config.time_span[1] / 3600),
            },
        }

        return outcomes

    def _update_kinetics_from_properties(self, properties: Dict[str, Any]) -> None:
        """Update kinetics parameters from compound properties."""
        if "molecular_weight" in properties:
            # Estimate molecular volume from molecular weight
            mw = properties["molecular_weight"]
            density_estimate = 1200  # kg/m³ typical for organics
            avogadro = 6.022e23
            self.kinetics_params.molecular_volume = mw / (density_estimate * avogadro)

        if "solubility" in properties:
            # Adjust nucleation parameters based on solubility
            solubility = properties["solubility"]
            self.kinetics_params.nucleation_rate_constant *= 1.0 / solubility

        if "crystal_habit_energy" in properties:
            self.kinetics_params.surface_energy = properties["crystal_habit_energy"]

        logger.info("Updated kinetics parameters from compound properties")

"""
Dynamic process integration for formulations.

Integrates with AI-driven in-operando characterization and autonomous
materials process optimization to predict and control evolution of
microstructure and properties during dynamic processes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.backend import Backend
from ..core.exceptions import ChemiAccelError
from .core import ComponentData, MixtureCalculator, MixtureData


@dataclass
class ProcessConditions:
    """Dynamic process conditions."""

    temperature_profile: Optional[Callable[[float], float]] = None
    pressure_profile: Optional[Callable[[float], float]] = None
    shear_rate_profile: Optional[Callable[[float], float]] = None
    time_steps: Optional[np.ndarray] = None
    boundary_conditions: Optional[Dict[str, Any]] = None
    control_parameters: Optional[Dict[str, Any]] = None

    def get_conditions_at_time(self, t: float) -> Dict[str, float]:
        """Get process conditions at specific time."""
        conditions = {}

        if self.temperature_profile:
            conditions["temperature"] = self.temperature_profile(t)
        if self.pressure_profile:
            conditions["pressure"] = self.pressure_profile(t)
        if self.shear_rate_profile:
            conditions["shear_rate"] = self.shear_rate_profile(t)

        return conditions


@dataclass
class FormulationProcess:
    """Results from dynamic process simulation."""

    process_type: str
    time_points: np.ndarray
    mixture_evolution: List[MixtureData]
    property_evolution: List[Dict[str, float]]
    microstructure_evolution: List[Dict[str, Any]]
    process_conditions: ProcessConditions
    control_actions: Optional[List[Dict[str, Any]]] = None

    @property
    def final_state(self) -> MixtureData:
        """Final mixture state."""
        return self.mixture_evolution[-1] if self.mixture_evolution else None

    @property
    def final_properties(self) -> Dict[str, float]:
        """Final mixture properties."""
        return self.property_evolution[-1] if self.property_evolution else {}


class ProcessModel(ABC):
    """Abstract base class for dynamic process models."""

    @abstractmethod
    def simulate_step(
        self, mixture: MixtureData, conditions: Dict[str, float], dt: float, **kwargs
    ) -> Tuple[MixtureData, Dict[str, float], Dict[str, Any]]:
        """Simulate single process step."""
        pass

    @abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass


class CrystallizationModel(ProcessModel):
    """Model for crystallization processes."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.nucleation_rate_constant = 1e6  # 1/m³/s
        self.growth_rate_constant = 1e-8  # m/s
        self.surface_energy = 0.1  # J/m²

    def simulate_step(
        self, mixture: MixtureData, conditions: Dict[str, float], dt: float, **kwargs
    ) -> Tuple[MixtureData, Dict[str, float], Dict[str, Any]]:
        """Simulate crystallization step."""

        T = conditions.get("temperature", 298.15)
        supersaturation = self._calculate_supersaturation(mixture, T)

        # Update crystal size distribution
        microstructure = self._update_crystal_distribution(
            mixture, supersaturation, T, dt
        )

        # Update mixture composition (some solute crystallizes)
        new_mixture = self._update_mixture_composition(mixture, microstructure, dt)

        # Calculate updated properties
        properties = self._calculate_crystallization_properties(
            new_mixture, microstructure, T
        )

        return new_mixture, properties, microstructure

    def _calculate_supersaturation(self, mixture: MixtureData, T: float) -> float:
        """Calculate supersaturation ratio."""
        # Simplified supersaturation calculation
        # In practice, would use solubility models

        # Assume first component is solute, second is solvent
        if len(mixture.components) < 2:
            return 1.0

        solute_fraction = mixture.components[0].mol_fraction
        # Simplified temperature-dependent solubility
        solubility = 0.1 * np.exp(-1000.0 / T)  # Temperature dependent

        return solute_fraction / solubility if solubility > 0 else 1.0

    def _update_crystal_distribution(
        self, mixture: MixtureData, supersaturation: float, T: float, dt: float
    ) -> Dict[str, Any]:
        """Update crystal size distribution."""

        # Get current crystal data or initialize
        current_crystals = mixture.mixture_properties.get(
            "crystals",
            {
                "number_density": 0.0,
                "mean_size": 0.0,
                "size_distribution": np.array([]),
            },
        )

        # Nucleation rate (simplified)
        if supersaturation > 1.0:
            nucleation_rate = (
                self.nucleation_rate_constant
                * (supersaturation - 1.0) ** 2
                * np.exp(
                    -16
                    * np.pi
                    * self.surface_energy**3
                    / (3 * (8.314 * T) ** 3 * np.log(supersaturation) ** 2)
                )
            )
        else:
            nucleation_rate = 0.0

        # Growth rate (simplified)
        if supersaturation > 1.0:
            growth_rate = self.growth_rate_constant * (supersaturation - 1.0)
        else:
            growth_rate = 0.0

        # Update crystal distribution
        new_crystals = current_crystals.copy()

        # Add new nuclei
        new_nuclei = nucleation_rate * dt
        new_crystals["number_density"] += new_nuclei

        # Grow existing crystals
        if new_crystals["mean_size"] > 0:
            new_crystals["mean_size"] += growth_rate * dt
        else:
            new_crystals["mean_size"] = growth_rate * dt

        return new_crystals

    def _update_mixture_composition(
        self, mixture: MixtureData, microstructure: Dict[str, Any], dt: float
    ) -> MixtureData:
        """Update mixture composition based on crystallization."""

        # Calculate amount crystallized
        crystal_volume = (
            microstructure["number_density"]
            * (4 / 3)
            * np.pi
            * (microstructure["mean_size"] / 2) ** 3
        )

        # Assume 10% volume fraction affects composition significantly
        if crystal_volume > 0.1:
            # Create new mixture with updated composition
            new_components = []
            for i, comp in enumerate(mixture.components):
                new_fraction = comp.mol_fraction
                if i == 0:  # Assume first component crystallizes
                    new_fraction *= 0.95  # 5% crystallizes
                else:  # Solvent fraction increases relatively
                    new_fraction *= 1.05

                new_components.append(
                    ComponentData(
                        smiles=comp.smiles,
                        mol_fraction=new_fraction,
                        molecular_weight=comp.molecular_weight,
                        density=comp.density,
                        properties=comp.properties.copy(),
                    )
                )

            # Normalize fractions
            total = sum(comp.mol_fraction for comp in new_components)
            for comp in new_components:
                comp.mol_fraction /= total

            new_mixture = MixtureData(
                components=new_components,
                temperature=mixture.temperature,
                pressure=mixture.pressure,
                mixture_properties=mixture.mixture_properties.copy(),
            )
            new_mixture.mixture_properties["crystals"] = microstructure

            return new_mixture

        # No significant change
        mixture.mixture_properties["crystals"] = microstructure
        return mixture

    def _calculate_crystallization_properties(
        self, mixture: MixtureData, microstructure: Dict[str, Any], T: float
    ) -> Dict[str, float]:
        """Calculate properties during crystallization."""

        crystal_fraction = (
            microstructure["number_density"]
            * (4 / 3)
            * np.pi
            * (microstructure["mean_size"] / 2) ** 3
        )

        return {
            "crystal_fraction": crystal_fraction,
            "mean_crystal_size": microstructure["mean_size"],
            "nucleation_rate": microstructure.get("nucleation_rate", 0.0),
            "growth_rate": microstructure.get("growth_rate", 0.0),
            "viscosity": self._calculate_suspension_viscosity(
                mixture, crystal_fraction
            ),
        }

    def _calculate_suspension_viscosity(
        self, mixture: MixtureData, crystal_fraction: float
    ) -> float:
        """Calculate suspension viscosity using Einstein equation."""

        # Base viscosity (simplified)
        base_viscosity = 0.001  # Pa·s (water-like)

        # Einstein equation for dilute suspensions
        if crystal_fraction < 0.1:
            viscosity = base_viscosity * (1 + 2.5 * crystal_fraction)
        else:
            # Krieger-Dougherty equation for concentrated suspensions
            phi_max = 0.64  # Maximum packing fraction
            if crystal_fraction < phi_max:
                viscosity = base_viscosity * (1 - crystal_fraction / phi_max) ** (
                    -2.5 * phi_max
                )
            else:
                viscosity = float("inf")  # Jammed state

        return viscosity

    def get_model_parameters(self) -> Dict[str, Any]:
        """Get crystallization model parameters."""
        return {
            "nucleation_rate_constant": self.nucleation_rate_constant,
            "growth_rate_constant": self.growth_rate_constant,
            "surface_energy": self.surface_energy,
        }


class MixingModel(ProcessModel):
    """Model for mixing processes."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.diffusion_coefficients = {}
        self.mixing_efficiency = 0.8

    def simulate_step(
        self, mixture: MixtureData, conditions: Dict[str, float], dt: float, **kwargs
    ) -> Tuple[MixtureData, Dict[str, float], Dict[str, Any]]:
        """Simulate mixing step."""

        shear_rate = conditions.get("shear_rate", 0.0)
        T = conditions.get("temperature", 298.15)

        # Update mixing state
        microstructure = self._update_mixing_state(mixture, shear_rate, T, dt)

        # Mixture composition doesn't change during mixing
        new_mixture = mixture
        new_mixture.mixture_properties.update(microstructure)

        # Calculate mixing properties
        properties = self._calculate_mixing_properties(
            new_mixture, microstructure, shear_rate, T
        )

        return new_mixture, properties, microstructure

    def _update_mixing_state(
        self, mixture: MixtureData, shear_rate: float, T: float, dt: float
    ) -> Dict[str, Any]:
        """Update mixing state."""

        # Get current mixing state
        current_state = mixture.mixture_properties.get(
            "mixing_state",
            {
                "homogeneity": 0.0,  # 0 = unmixed, 1 = perfectly mixed
                "segregation_index": 1.0,  # 1 = completely segregated, 0 = homogeneous
                "mixing_time": 0.0,
            },
        )

        # Calculate mixing progress
        mixing_rate = self._calculate_mixing_rate(shear_rate, T)

        # Update homogeneity
        new_homogeneity = current_state["homogeneity"] + mixing_rate * dt
        new_homogeneity = min(new_homogeneity, 1.0)  # Cap at perfect mixing

        # Update segregation index (inverse of homogeneity)
        new_segregation = 1.0 - new_homogeneity

        # Update mixing time
        new_mixing_time = current_state["mixing_time"] + dt

        return {
            "homogeneity": new_homogeneity,
            "segregation_index": new_segregation,
            "mixing_time": new_mixing_time,
            "mixing_rate": mixing_rate,
        }

    def _calculate_mixing_rate(self, shear_rate: float, T: float) -> float:
        """Calculate mixing rate based on shear rate and temperature."""

        # Simple mixing rate model
        # Higher shear rate and temperature increase mixing
        base_rate = 0.1  # 1/s
        shear_factor = 1.0 + 0.01 * shear_rate  # Linear dependence on shear
        temp_factor = np.exp((T - 298.15) / 100.0)  # Exponential temperature dependence

        return base_rate * shear_factor * temp_factor * self.mixing_efficiency

    def _calculate_mixing_properties(
        self,
        mixture: MixtureData,
        microstructure: Dict[str, Any],
        shear_rate: float,
        T: float,
    ) -> Dict[str, float]:
        """Calculate properties during mixing."""

        homogeneity = microstructure["homogeneity"]

        return {
            "homogeneity": homogeneity,
            "segregation_index": microstructure["segregation_index"],
            "mixing_energy": shear_rate**2 * 0.001,  # Simplified energy dissipation
            "effective_viscosity": self._calculate_effective_viscosity(
                mixture, shear_rate
            ),
            "mixing_efficiency": self.mixing_efficiency,
        }

    def _calculate_effective_viscosity(
        self, mixture: MixtureData, shear_rate: float
    ) -> float:
        """Calculate effective viscosity during mixing."""

        # Simple mixing rule for viscosity
        base_viscosity = 0.001  # Pa·s

        # Shear-thinning behavior
        if shear_rate > 0:
            viscosity = base_viscosity * (1 + 100.0 / shear_rate)
        else:
            viscosity = base_viscosity * 100.0  # High viscosity at rest

        return viscosity

    def get_model_parameters(self) -> Dict[str, Any]:
        """Get mixing model parameters."""
        return {
            "mixing_efficiency": self.mixing_efficiency,
            "diffusion_coefficients": self.diffusion_coefficients,
        }


class PhaseSeparationModel(ProcessModel):
    """Model for phase separation processes."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.interfacial_tension = 0.01  # N/m
        self.mobility = 1e-12  # m²/s/Pa

    def simulate_step(
        self, mixture: MixtureData, conditions: Dict[str, float], dt: float, **kwargs
    ) -> Tuple[MixtureData, Dict[str, float], Dict[str, Any]]:
        """Simulate phase separation step."""

        T = conditions.get("temperature", 298.15)

        # Calculate phase separation driving force
        separation_force = self._calculate_separation_force(mixture, T)

        # Update phase morphology
        microstructure = self._update_phase_morphology(mixture, separation_force, T, dt)

        # Update mixture with new phase information
        new_mixture = self._update_mixture_phases(mixture, microstructure)

        # Calculate phase separation properties
        properties = self._calculate_separation_properties(
            new_mixture, microstructure, T
        )

        return new_mixture, properties, microstructure

    def _calculate_separation_force(self, mixture: MixtureData, T: float) -> float:
        """Calculate thermodynamic driving force for phase separation."""

        # Simplified Flory-Huggins model for phase separation
        if len(mixture.components) < 2:
            return 0.0

        x1 = mixture.components[0].mol_fraction
        x2 = mixture.components[1].mol_fraction

        # Critical value for phase separation (simplified)
        chi_critical = 2.0  # Flory-Huggins parameter
        chi_actual = chi_critical * (
            1.0 + (T - 298.15) / 100.0
        )  # Temperature dependence

        # Spinodal condition
        if chi_actual > chi_critical and 0.1 < x1 < 0.9:
            # In unstable region - phase separation occurs
            separation_force = chi_actual * x1 * x2
        else:
            separation_force = 0.0

        return separation_force

    def _update_phase_morphology(
        self, mixture: MixtureData, separation_force: float, T: float, dt: float
    ) -> Dict[str, Any]:
        """Update phase morphology during separation."""

        # Get current morphology or initialize
        current_morphology = mixture.mixture_properties.get(
            "phase_morphology",
            {
                "domain_size": 0.0,  # Characteristic domain size (m)
                "phase_fraction": [0.5, 0.5],  # Volume fractions of phases
                "interfacial_area": 0.0,  # Interfacial area per unit volume
                "separation_progress": 0.0,  # 0 = mixed, 1 = fully separated
            },
        )

        if separation_force > 0:
            # Calculate growth rate of domains
            growth_rate = self.mobility * separation_force / self.interfacial_tension

            # Update domain size
            new_domain_size = current_morphology["domain_size"] + growth_rate * dt
            new_domain_size = max(new_domain_size, 1e-9)  # Minimum size 1 nm

            # Update separation progress
            max_domain_size = 1e-6  # Maximum domain size (1 μm)
            separation_progress = min(new_domain_size / max_domain_size, 1.0)

            # Update interfacial area (decreases as domains grow)
            if new_domain_size > 0:
                interfacial_area = (
                    6.0 / new_domain_size
                )  # Spherical domains approximation
            else:
                interfacial_area = float("inf")

            return {
                "domain_size": new_domain_size,
                "phase_fraction": current_morphology["phase_fraction"],
                "interfacial_area": interfacial_area,
                "separation_progress": separation_progress,
                "growth_rate": growth_rate,
            }

        return current_morphology

    def _update_mixture_phases(
        self, mixture: MixtureData, microstructure: Dict[str, Any]
    ) -> MixtureData:
        """Update mixture with phase separation information."""

        separation_progress = microstructure["separation_progress"]

        if separation_progress > 0.1:  # Significant phase separation
            # Create phase data
            phase_data = {
                "n_phases": 2,
                "phase_compositions": [],
                "phase_fractions": microstructure["phase_fraction"],
            }

            # Simplified phase compositions (component 1 enriched in phase 1)
            for i in range(2):
                phase_comp = []
                for j, comp in enumerate(mixture.components):
                    if j == 0:  # First component
                        if i == 0:  # Phase 1 - enriched
                            fraction = comp.mol_fraction * (1 + separation_progress)
                        else:  # Phase 2 - depleted
                            fraction = comp.mol_fraction * (1 - separation_progress)
                    else:  # Other components
                        if i == 0:  # Phase 1 - depleted
                            fraction = comp.mol_fraction * (
                                1 - separation_progress * 0.5
                            )
                        else:  # Phase 2 - enriched
                            fraction = comp.mol_fraction * (
                                1 + separation_progress * 0.5
                            )

                    phase_comp.append(fraction)

                # Normalize
                total = sum(phase_comp)
                phase_comp = [f / total for f in phase_comp]
                phase_data["phase_compositions"].append(phase_comp)

            # Update mixture phase data
            new_mixture = mixture
            new_mixture.phase_data = phase_data
            new_mixture.mixture_properties["phase_morphology"] = microstructure

            return new_mixture

        return mixture

    def _calculate_separation_properties(
        self, mixture: MixtureData, microstructure: Dict[str, Any], T: float
    ) -> Dict[str, float]:
        """Calculate properties during phase separation."""

        domain_size = microstructure["domain_size"]
        separation_progress = microstructure["separation_progress"]
        interfacial_area = microstructure["interfacial_area"]

        return {
            "domain_size": domain_size,
            "separation_progress": separation_progress,
            "interfacial_area": interfacial_area,
            "interfacial_energy": interfacial_area * self.interfacial_tension,
            "optical_properties": self._calculate_optical_properties(domain_size),
            "mechanical_properties": self._calculate_mechanical_properties(
                separation_progress
            ),
        }

    def _calculate_optical_properties(self, domain_size: float) -> float:
        """Calculate optical properties based on domain size."""

        # Light scattering depends on domain size relative to wavelength
        visible_wavelength = 500e-9  # 500 nm

        if domain_size > 0:
            scattering_parameter = domain_size / visible_wavelength
            if scattering_parameter < 0.1:
                opacity = 0.0  # Transparent
            elif scattering_parameter < 1.0:
                opacity = scattering_parameter  # Translucent
            else:
                opacity = 1.0  # Opaque
        else:
            opacity = 0.0

        return opacity

    def _calculate_mechanical_properties(self, separation_progress: float) -> float:
        """Calculate mechanical properties based on separation."""

        # Simplified relationship - phase separation can increase or decrease stiffness
        # depending on phase contrast
        base_modulus = 1e6  # Pa

        # Assume phase separation increases modulus due to reinforcement
        modulus_enhancement = 1.0 + separation_progress * 0.5

        return base_modulus * modulus_enhancement

    def get_model_parameters(self) -> Dict[str, Any]:
        """Get phase separation model parameters."""
        return {
            "interfacial_tension": self.interfacial_tension,
            "mobility": self.mobility,
        }


class DynamicProcessIntegrator:
    """
    Dynamic process integration system for formulations.

    Integrates multiple process models to simulate complex
    formulation processes like crystallization, mixing, and phase separation.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.calculator = MixtureCalculator(backend=backend)

        # Initialize process models
        self.models = {
            "crystallization": CrystallizationModel(backend),
            "mixing": MixingModel(backend),
            "phase_separation": PhaseSeparationModel(backend),
        }

        # Integration settings
        self.default_dt = 0.1  # seconds
        self.max_time = 3600.0  # 1 hour

    def simulate_process(
        self,
        mixture: MixtureData,
        process_conditions: ProcessConditions,
        process_types: List[str],
        simulation_time: float = None,
        dt: float = None,
        **kwargs,
    ) -> FormulationProcess:
        """
        Simulate dynamic formulation process.

        Args:
            mixture: Initial mixture state
            process_conditions: Process conditions over time
            process_types: Types of processes to simulate
            simulation_time: Total simulation time
            dt: Time step size

        Returns:
            FormulationProcess results
        """

        simulation_time = simulation_time or self.max_time
        dt = dt or self.default_dt

        # Initialize simulation
        time_points = np.arange(0, simulation_time + dt, dt)
        mixture_evolution = [mixture]
        property_evolution = []
        microstructure_evolution = []

        current_mixture = mixture

        # Main simulation loop
        for i, t in enumerate(time_points[1:], 1):
            # Get process conditions at current time
            conditions = process_conditions.get_conditions_at_time(t)

            # Apply each process model
            for process_type in process_types:
                if process_type in self.models:
                    model = self.models[process_type]

                    # Simulate process step
                    current_mixture, properties, microstructure = model.simulate_step(
                        current_mixture, conditions, dt, **kwargs
                    )

                    # Store results for this process
                    if i == 1:  # First step
                        property_evolution.append({})
                        microstructure_evolution.append({})

                    # Update stored data
                    property_evolution[-1].update(
                        {
                            f"{process_type}_{key}": value
                            for key, value in properties.items()
                        }
                    )

                    microstructure_evolution[-1].update(
                        {
                            f"{process_type}_{key}": value
                            for key, value in microstructure.items()
                        }
                    )

            # Store mixture state
            mixture_evolution.append(current_mixture)

            # Add overall mixture properties
            overall_properties = self.calculator.calculate_mixture_properties(
                current_mixture
            )
            if property_evolution:
                property_evolution[-1].update(overall_properties)
            else:
                property_evolution.append(overall_properties)

            # Check for simulation termination conditions
            if self._check_termination_conditions(current_mixture, properties):
                break

        return FormulationProcess(
            process_type="_".join(process_types),
            time_points=time_points[: len(mixture_evolution)],
            mixture_evolution=mixture_evolution,
            property_evolution=property_evolution,
            microstructure_evolution=microstructure_evolution,
            process_conditions=process_conditions,
        )

    def _check_termination_conditions(
        self, mixture: MixtureData, properties: Dict[str, float]
    ) -> bool:
        """Check if simulation should terminate early."""

        # Example termination conditions

        # Complete crystallization
        if "crystallization_crystal_fraction" in properties:
            if properties["crystallization_crystal_fraction"] > 0.95:
                return True

        # Complete mixing
        if "mixing_homogeneity" in properties:
            if properties["mixing_homogeneity"] > 0.99:
                return True

        # Complete phase separation
        if "phase_separation_separation_progress" in properties:
            if properties["phase_separation_separation_progress"] > 0.99:
                return True

        return False

    def optimize_process_conditions(
        self,
        mixture: MixtureData,
        target_properties: Dict[str, float],
        process_types: List[str],
        variable_conditions: Dict[str, Tuple[float, float]],
        optimization_method: str = "differential_evolution",
        **kwargs,
    ) -> Tuple[ProcessConditions, FormulationProcess]:
        """
        Optimize process conditions to achieve target properties.

        Args:
            mixture: Initial mixture
            target_properties: Target final properties
            process_types: Process types to simulate
            variable_conditions: Conditions to optimize with bounds
            optimization_method: Optimization algorithm

        Returns:
            Optimal process conditions and simulation results
        """

        try:
            from scipy.optimize import differential_evolution

            def objective(params):
                # Create process conditions from parameters
                conditions = self._create_conditions_from_params(
                    params, variable_conditions
                )

                # Run simulation
                try:
                    result = self.simulate_process(
                        mixture, conditions, process_types, **kwargs
                    )

                    # Calculate objective (difference from targets)
                    final_props = result.final_properties
                    error = 0.0

                    for prop_name, target_value in target_properties.items():
                        if prop_name in final_props:
                            actual_value = final_props[prop_name]
                            relative_error = abs(actual_value - target_value) / abs(
                                target_value
                            )
                            error += relative_error

                    return error

                except Exception as e:
                    return 1e6  # Large penalty for failed simulations

            # Setup bounds
            bounds = list(variable_conditions.values())

            # Run optimization
            result = differential_evolution(
                objective,
                bounds,
                maxiter=kwargs.get("max_iterations", 50),
                popsize=kwargs.get("population_size", 10),
            )

            # Create optimal conditions
            optimal_conditions = self._create_conditions_from_params(
                result.x, variable_conditions
            )

            # Run final simulation with optimal conditions
            final_simulation = self.simulate_process(
                mixture, optimal_conditions, process_types, **kwargs
            )

            return optimal_conditions, final_simulation

        except ImportError:
            raise ChemiAccelError("scipy required for process optimization")

    def _create_conditions_from_params(
        self, params: np.ndarray, variable_conditions: Dict[str, Tuple[float, float]]
    ) -> ProcessConditions:
        """Create ProcessConditions from optimization parameters."""

        conditions = ProcessConditions()
        param_names = list(variable_conditions.keys())

        for i, param_name in enumerate(param_names):
            value = params[i]

            if "temperature" in param_name:
                conditions.temperature_profile = lambda t, v=value: v
            elif "pressure" in param_name:
                conditions.pressure_profile = lambda t, v=value: v
            elif "shear_rate" in param_name:
                conditions.shear_rate_profile = lambda t, v=value: v

        return conditions

    def analyze_process_sensitivity(
        self,
        mixture: MixtureData,
        process_types: List[str],
        base_conditions: ProcessConditions,
        sensitivity_parameters: List[str],
        perturbation_size: float = 0.1,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze sensitivity of process outcomes to conditions.

        Args:
            mixture: Initial mixture
            process_types: Process types to simulate
            base_conditions: Base process conditions
            sensitivity_parameters: Parameters to analyze
            perturbation_size: Relative perturbation size

        Returns:
            Sensitivity analysis results
        """

        # Run base simulation
        base_result = self.simulate_process(
            mixture, base_conditions, process_types, **kwargs
        )
        base_properties = base_result.final_properties

        sensitivity_results = {}

        for param in sensitivity_parameters:
            param_results = {"positive": {}, "negative": {}}

            for direction in ["positive", "negative"]:
                # Create perturbed conditions
                perturbed_conditions = self._perturb_conditions(
                    base_conditions, param, perturbation_size, direction
                )

                # Run perturbed simulation
                perturbed_result = self.simulate_process(
                    mixture, perturbed_conditions, process_types, **kwargs
                )
                perturbed_properties = perturbed_result.final_properties

                # Calculate sensitivities
                for prop_name, base_value in base_properties.items():
                    if prop_name in perturbed_properties:
                        perturbed_value = perturbed_properties[prop_name]
                        if base_value != 0:
                            sensitivity = (perturbed_value - base_value) / (
                                base_value * perturbation_size
                            )
                        else:
                            sensitivity = 0.0

                        param_results[direction][prop_name] = sensitivity

            sensitivity_results[param] = param_results

        return sensitivity_results

    def _perturb_conditions(
        self,
        base_conditions: ProcessConditions,
        parameter: str,
        perturbation_size: float,
        direction: str,
    ) -> ProcessConditions:
        """Create perturbed process conditions."""

        perturbed_conditions = ProcessConditions()

        # Copy base conditions
        perturbed_conditions.temperature_profile = base_conditions.temperature_profile
        perturbed_conditions.pressure_profile = base_conditions.pressure_profile
        perturbed_conditions.shear_rate_profile = base_conditions.shear_rate_profile
        perturbed_conditions.time_steps = base_conditions.time_steps
        perturbed_conditions.boundary_conditions = base_conditions.boundary_conditions
        perturbed_conditions.control_parameters = base_conditions.control_parameters

        # Apply perturbation
        multiplier = (
            1.0 + perturbation_size
            if direction == "positive"
            else 1.0 - perturbation_size
        )

        if "temperature" in parameter and base_conditions.temperature_profile:
            original_profile = base_conditions.temperature_profile
            perturbed_conditions.temperature_profile = (
                lambda t: original_profile(t) * multiplier
            )

        elif "pressure" in parameter and base_conditions.pressure_profile:
            original_profile = base_conditions.pressure_profile
            perturbed_conditions.pressure_profile = (
                lambda t: original_profile(t) * multiplier
            )

        elif "shear_rate" in parameter and base_conditions.shear_rate_profile:
            original_profile = base_conditions.shear_rate_profile
            perturbed_conditions.shear_rate_profile = (
                lambda t: original_profile(t) * multiplier
            )

        return perturbed_conditions

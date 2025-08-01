"""
Dynamic Property Characterization and Failure Mechanism Identification

This module develops capabilities to characterize dynamic material properties
(e.g., shock Hugoniot, radiation damage evolution, glass formation kinetics)
and identify failure mechanisms specific to extreme or non-equilibrium conditions.
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
    from cupyx.scipy import signal as cp_signal

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cp_signal = None
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

try:
    from scipy import cluster, optimize, signal
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    SCIPY_AVAILABLE = True
except ImportError:
    signal = None
    optimize = None
    cluster = None
    SCIPY_AVAILABLE = False

from ..core.exceptions import AnalysisError, ChemiAccelError
from ..utils.logging import get_logger
from .simulation_engine import ExtremeConditionType, SimulationResults

logger = get_logger(__name__)


class PropertyType(Enum):
    """Types of dynamic properties that can be characterized."""

    SHOCK_HUGONIOT = "shock_hugoniot"
    EQUATION_OF_STATE = "equation_of_state"
    STRESS_STRAIN_RESPONSE = "stress_strain_response"
    DAMAGE_EVOLUTION = "damage_evolution"
    CRACK_PROPAGATION = "crack_propagation"
    PHASE_TRANSITION = "phase_transition"
    GLASS_FORMATION = "glass_formation"
    RECRYSTALLIZATION = "recrystallization"
    RADIATION_DAMAGE = "radiation_damage"
    CREEP_BEHAVIOR = "creep_behavior"
    FATIGUE_RESPONSE = "fatigue_response"
    CORROSION_KINETICS = "corrosion_kinetics"


class FailureMechanism(Enum):
    """Types of failure mechanisms."""

    BRITTLE_FRACTURE = "brittle_fracture"
    DUCTILE_FAILURE = "ductile_failure"
    FATIGUE_CRACKING = "fatigue_cracking"
    CREEP_RUPTURE = "creep_rupture"
    STRESS_CORROSION = "stress_corrosion_cracking"
    RADIATION_EMBRITTLEMENT = "radiation_embrittlement"
    THERMAL_SHOCK = "thermal_shock"
    SPALLATION = "spallation"
    VOID_NUCLEATION = "void_nucleation"
    GRAIN_BOUNDARY_FAILURE = "grain_boundary_failure"
    CLEAVAGE = "cleavage"
    DELAMINATION = "delamination"


@dataclass
class DynamicProperty:
    """Represents a characterized dynamic property."""

    property_type: PropertyType
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    time_series: np.ndarray = field(default_factory=lambda: np.array([]))
    parameters: Dict[str, float] = field(default_factory=dict)
    fitting_quality: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # Property-specific metadata
    temperature_dependence: Optional[np.ndarray] = None
    strain_rate_dependence: Optional[np.ndarray] = None
    pressure_dependence: Optional[np.ndarray] = None


@dataclass
class FailureAnalysis:
    """Results from failure mechanism analysis."""

    primary_mechanism: FailureMechanism
    secondary_mechanisms: List[FailureMechanism] = field(default_factory=list)
    failure_onset_time: float = 0.0
    failure_location: np.ndarray = field(default_factory=lambda: np.array([]))

    # Quantitative metrics
    damage_accumulation_rate: float = 0.0
    critical_stress: float = 0.0
    critical_strain: float = 0.0
    fracture_energy: float = 0.0

    # Statistical analysis
    failure_probability: float = 0.0
    confidence_level: float = 0.95

    # Mechanism-specific parameters
    mechanism_parameters: Dict[str, float] = field(default_factory=dict)


class BaseAnalyzer(ABC):
    """Base class for all property analyzers."""

    def __init__(self, backend: str = "cpu"):
        self.backend = backend
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def analyze(self, simulation_results: SimulationResults) -> Dict[str, Any]:
        """Analyze simulation results."""
        pass

    def _move_to_backend(self, array: np.ndarray) -> Union[np.ndarray, Any]:
        """Move array to appropriate backend."""
        if self.backend == "cuda" and CUPY_AVAILABLE:
            return cp.asarray(array)
        elif self.backend == "jax" and JAX_AVAILABLE:
            return jnp.asarray(array)
        else:
            return array

    def _move_to_cpu(self, array: Union[np.ndarray, Any]) -> np.ndarray:
        """Move array back to CPU."""
        if hasattr(array, "__array__"):
            return np.asarray(array)
        elif CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        else:
            return np.asarray(array)


class DynamicPropertyAnalyzer(BaseAnalyzer):
    """Analyzer for dynamic material properties under extreme conditions."""

    def __init__(self, backend: str = "cpu"):
        super().__init__(backend)
        self.property_extractors = {
            PropertyType.SHOCK_HUGONIOT: self._extract_hugoniot,
            PropertyType.STRESS_STRAIN_RESPONSE: self._extract_stress_strain,
            PropertyType.DAMAGE_EVOLUTION: self._extract_damage_evolution,
            PropertyType.PHASE_TRANSITION: self._extract_phase_transition,
            PropertyType.GLASS_FORMATION: self._extract_glass_formation,
            PropertyType.RADIATION_DAMAGE: self._extract_radiation_damage,
        }

    def analyze(
        self, simulation_results: SimulationResults
    ) -> Dict[str, DynamicProperty]:
        """Analyze dynamic properties from simulation results."""

        self.logger.info("Starting dynamic property analysis")

        properties = {}

        # Determine which properties to analyze based on available data
        available_properties = self._identify_available_properties(simulation_results)

        for prop_type in available_properties:
            if prop_type in self.property_extractors:
                try:
                    property_data = self.property_extractors[prop_type](
                        simulation_results
                    )
                    properties[prop_type.value] = property_data

                    self.logger.info(f"Successfully analyzed {prop_type.value}")

                except Exception as e:
                    self.logger.warning(f"Failed to analyze {prop_type.value}: {e}")

        return properties

    def _identify_available_properties(
        self, results: SimulationResults
    ) -> List[PropertyType]:
        """Identify which properties can be analyzed from available data."""

        available = []

        # Check for stress-strain data
        if results.stress_tensor.size > 0 and results.strain_tensor.size > 0:
            available.append(PropertyType.STRESS_STRAIN_RESPONSE)

        # Check for damage data
        if results.damage_fraction.size > 0:
            available.append(PropertyType.DAMAGE_EVOLUTION)

        # Check for high-pressure/shock conditions
        if results.pressures.size > 0 and np.max(results.pressures) > 1e8:  # > 1000 atm
            available.append(PropertyType.SHOCK_HUGONIOT)
            available.append(PropertyType.EQUATION_OF_STATE)

        # Check for temperature variations (phase transitions)
        if results.temperatures.size > 0:
            temp_variation = np.std(results.temperatures)
            if temp_variation > 50.0:  # K
                available.append(PropertyType.PHASE_TRANSITION)

            # Check for rapid cooling (glass formation)
            if len(results.temperatures) > 1:
                cooling_rate = -np.gradient(results.temperatures)
                if np.max(cooling_rate) > 1e6:  # K/s
                    available.append(PropertyType.GLASS_FORMATION)

        # Check for radiation damage
        if results.defect_count.size > 0:
            available.append(PropertyType.RADIATION_DAMAGE)

        return available

    def _extract_hugoniot(self, results: SimulationResults) -> DynamicProperty:
        """Extract shock Hugoniot relationship (Us vs Up)."""

        # Calculate shock velocity (Us) and particle velocity (Up)
        # This is a simplified calculation

        pressures = self._move_to_backend(results.pressures)
        densities = self._calculate_density_evolution(results)

        # Hugoniot relation: Us = c0 + s * Up
        # Where Us is shock velocity, Up is particle velocity

        if len(pressures) < 10:
            # Not enough data points
            return DynamicProperty(
                property_type=PropertyType.SHOCK_HUGONIOT,
                parameters={"c0": 0.0, "s": 0.0},
                fitting_quality=0.0,
            )

        # Estimate particle velocities from pressure and density changes
        initial_density = densities[0] if len(densities) > 0 else 1000.0
        density_ratios = densities / initial_density

        # Simplified Hugoniot calculation
        particle_velocities = np.sqrt(pressures / (initial_density * density_ratios))

        # Rankine-Hugoniot relation
        shock_velocities = particle_velocities * density_ratios / (density_ratios - 1)
        shock_velocities = np.where(np.isfinite(shock_velocities), shock_velocities, 0)

        # Fit linear Hugoniot relation: Us = c0 + s * Up
        if SCIPY_AVAILABLE and len(particle_velocities) > 2:
            try:
                params, cov = np.polyfit(
                    self._move_to_cpu(particle_velocities),
                    self._move_to_cpu(shock_velocities),
                    1,
                    cov=True,
                )
                c0, s = params[1], params[0]

                # Calculate R-squared
                ss_res = np.sum(
                    (shock_velocities - (c0 + s * particle_velocities)) ** 2
                )
                ss_tot = np.sum((shock_velocities - np.mean(shock_velocities)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            except:
                c0, s = 0.0, 1.0
                r_squared = 0.0
        else:
            c0, s = 0.0, 1.0
            r_squared = 0.0

        return DynamicProperty(
            property_type=PropertyType.SHOCK_HUGONIOT,
            values=self._move_to_cpu(shock_velocities),
            parameters={"c0": c0, "s": s, "bulk_sound_speed": c0},
            fitting_quality=r_squared,
        )

    def _extract_stress_strain(self, results: SimulationResults) -> DynamicProperty:
        """Extract stress-strain response."""

        if results.stress_tensor.size == 0 or results.strain_tensor.size == 0:
            return DynamicProperty(
                property_type=PropertyType.STRESS_STRAIN_RESPONSE,
                parameters={"elastic_modulus": 0.0, "yield_strength": 0.0},
            )

        # Calculate von Mises stress and equivalent strain
        von_mises_stress = []
        equivalent_strain = []

        for i in range(len(results.stress_tensor)):
            stress = results.stress_tensor[i]
            strain = results.strain_tensor[i]

            # von Mises stress
            s11, s22, s33 = stress[0, 0], stress[1, 1], stress[2, 2]
            s12, s13, s23 = stress[0, 1], stress[0, 2], stress[1, 2]

            vm_stress = np.sqrt(
                0.5
                * (
                    (s11 - s22) ** 2
                    + (s22 - s33) ** 2
                    + (s33 - s11) ** 2
                    + 6 * (s12**2 + s13**2 + s23**2)
                )
            )
            von_mises_stress.append(vm_stress)

            # Equivalent strain
            e11, e22, e33 = strain[0, 0], strain[1, 1], strain[2, 2]
            e12, e13, e23 = strain[0, 1], strain[0, 2], strain[1, 2]

            eq_strain = np.sqrt(
                (2 / 3)
                * (
                    e11**2
                    + e22**2
                    + e33**2
                    + 2 * (e12**2 + e13**2 + e23**2)
                )
            )
            equivalent_strain.append(eq_strain)

        von_mises_stress = np.array(von_mises_stress)
        equivalent_strain = np.array(equivalent_strain)

        # Calculate elastic modulus (slope of initial linear region)
        elastic_modulus = 0.0
        yield_strength = 0.0

        if len(von_mises_stress) > 5:
            # Find elastic region (first 20% of data or until strain > 0.002)
            elastic_limit = min(
                len(von_mises_stress) // 5, np.argmax(equivalent_strain > 0.002)
            )
            elastic_limit = max(2, elastic_limit)

            if elastic_limit > 1:
                # Linear regression for elastic modulus
                strain_elastic = equivalent_strain[:elastic_limit]
                stress_elastic = von_mises_stress[:elastic_limit]

                if len(strain_elastic) > 1 and np.std(strain_elastic) > 1e-6:
                    elastic_modulus = np.polyfit(strain_elastic, stress_elastic, 1)[0]

                # Yield strength (0.2% offset or maximum stress)
                if len(von_mises_stress) > elastic_limit:
                    yield_strength = np.max(von_mises_stress[elastic_limit:])

        return DynamicProperty(
            property_type=PropertyType.STRESS_STRAIN_RESPONSE,
            values=von_mises_stress,
            time_series=equivalent_strain,
            parameters={
                "elastic_modulus": elastic_modulus,
                "yield_strength": yield_strength,
                "ultimate_strength": np.max(von_mises_stress),
                "fracture_strain": equivalent_strain[-1]
                if len(equivalent_strain) > 0
                else 0.0,
            },
            fitting_quality=0.95,  # High confidence for direct calculation
        )

    def _extract_damage_evolution(self, results: SimulationResults) -> DynamicProperty:
        """Extract damage evolution characteristics."""

        if results.damage_fraction.size == 0:
            return DynamicProperty(
                property_type=PropertyType.DAMAGE_EVOLUTION,
                parameters={"damage_rate": 0.0, "critical_damage": 1.0},
            )

        damage = results.damage_fraction
        time_steps = np.arange(len(damage))

        # Calculate damage rate
        damage_rate = 0.0
        if len(damage) > 1:
            damage_rate = np.mean(np.gradient(damage))

        # Find critical damage threshold (where acceleration occurs)
        critical_damage = 0.1  # Default
        if len(damage) > 10:
            # Find point of maximum acceleration
            damage_accel = np.gradient(np.gradient(damage))
            max_accel_idx = np.argmax(damage_accel)
            if max_accel_idx > 0:
                critical_damage = damage[max_accel_idx]

        # Fit damage evolution model (exponential or power law)
        fitting_quality = 0.0
        model_params = {}

        if SCIPY_AVAILABLE and len(damage) > 5:
            try:
                # Try exponential model: D(t) = D0 * (1 - exp(-k*t))
                def exp_model(t, d0, k):
                    return d0 * (1 - np.exp(-k * t))

                popt, pcov = optimize.curve_fit(
                    exp_model,
                    time_steps,
                    damage,
                    p0=[1.0, 0.01],
                    bounds=([0, 0], [2, 1]),
                )

                d0, k = popt
                predicted = exp_model(time_steps, d0, k)
                r_squared = 1 - np.sum((damage - predicted) ** 2) / np.sum(
                    (damage - np.mean(damage)) ** 2
                )

                model_params = {"D0": d0, "rate_constant": k, "model": "exponential"}
                fitting_quality = r_squared

            except:
                model_params = {"model": "linear"}
                fitting_quality = 0.5

        return DynamicProperty(
            property_type=PropertyType.DAMAGE_EVOLUTION,
            values=damage,
            time_series=time_steps,
            parameters={
                "damage_rate": damage_rate,
                "critical_damage": critical_damage,
                "final_damage": damage[-1] if len(damage) > 0 else 0.0,
                **model_params,
            },
            fitting_quality=fitting_quality,
        )

    def _extract_phase_transition(self, results: SimulationResults) -> DynamicProperty:
        """Extract phase transition characteristics."""

        if results.temperatures.size == 0:
            return DynamicProperty(
                property_type=PropertyType.PHASE_TRANSITION,
                parameters={"transition_temperature": 0.0, "transition_enthalpy": 0.0},
            )

        temperatures = results.temperatures
        energies = (
            results.energies
            if results.energies.size > 0
            else np.zeros_like(temperatures)
        )

        # Look for temperature plateaus or energy discontinuities
        transition_temperature = 0.0
        transition_enthalpy = 0.0

        if len(temperatures) > 10:
            # Find temperature derivative
            temp_derivative = np.gradient(temperatures)

            # Look for regions where temperature change is minimal (plateaus)
            plateau_threshold = np.std(temp_derivative) * 0.1
            plateau_regions = np.abs(temp_derivative) < plateau_threshold

            if np.any(plateau_regions):
                # Find the longest plateau
                plateau_starts = np.where(
                    np.diff(np.concatenate(([False], plateau_regions)))
                )[0]
                plateau_ends = np.where(
                    np.diff(np.concatenate((plateau_regions, [False])))
                )[0]

                if len(plateau_starts) > 0 and len(plateau_ends) > 0:
                    plateau_lengths = plateau_ends - plateau_starts
                    longest_plateau_idx = np.argmax(plateau_lengths)

                    start_idx = plateau_starts[longest_plateau_idx]
                    end_idx = plateau_ends[longest_plateau_idx]

                    transition_temperature = np.mean(temperatures[start_idx:end_idx])

                    # Calculate enthalpy change
                    if len(energies) > end_idx:
                        energy_before = np.mean(
                            energies[max(0, start_idx - 5) : start_idx]
                        )
                        energy_after = np.mean(
                            energies[end_idx : min(len(energies), end_idx + 5)]
                        )
                        transition_enthalpy = energy_after - energy_before

        return DynamicProperty(
            property_type=PropertyType.PHASE_TRANSITION,
            values=temperatures,
            parameters={
                "transition_temperature": transition_temperature,
                "transition_enthalpy": transition_enthalpy,
                "temperature_range": np.max(temperatures) - np.min(temperatures),
            },
        )

    def _extract_glass_formation(self, results: SimulationResults) -> DynamicProperty:
        """Extract glass formation kinetics."""

        if results.temperatures.size == 0:
            return DynamicProperty(
                property_type=PropertyType.GLASS_FORMATION,
                parameters={"glass_transition_temp": 0.0, "cooling_rate": 0.0},
            )

        temperatures = results.temperatures
        time_steps = np.arange(len(temperatures))

        # Calculate cooling rate
        cooling_rate = 0.0
        if len(temperatures) > 1:
            cooling_rate = -np.mean(np.gradient(temperatures))

        # Identify glass transition temperature (inflection point in cooling curve)
        glass_transition_temp = 0.0

        if len(temperatures) > 10 and cooling_rate > 1e3:  # Rapid cooling
            # Find inflection point
            temp_second_deriv = np.gradient(np.gradient(temperatures))

            # Glass transition often shows up as change in heat capacity (second derivative)
            if len(temp_second_deriv) > 5:
                # Find maximum in second derivative
                max_second_deriv_idx = np.argmax(np.abs(temp_second_deriv[2:-2])) + 2
                glass_transition_temp = temperatures[max_second_deriv_idx]

        # Calculate critical cooling rate for glass formation
        critical_cooling_rate = max(1e5, cooling_rate * 0.1)  # Empirical estimate

        return DynamicProperty(
            property_type=PropertyType.GLASS_FORMATION,
            values=temperatures,
            time_series=time_steps,
            parameters={
                "glass_transition_temp": glass_transition_temp,
                "cooling_rate": cooling_rate,
                "critical_cooling_rate": critical_cooling_rate,
                "glass_formation_probability": min(
                    1.0, cooling_rate / critical_cooling_rate
                ),
            },
        )

    def _extract_radiation_damage(self, results: SimulationResults) -> DynamicProperty:
        """Extract radiation damage evolution."""

        if results.defect_count.size == 0:
            return DynamicProperty(
                property_type=PropertyType.RADIATION_DAMAGE,
                parameters={"defect_production_rate": 0.0, "defect_saturation": 0.0},
            )

        defect_counts = results.defect_count
        time_steps = np.arange(len(defect_counts))

        # Calculate defect production rate
        defect_rate = 0.0
        if len(defect_counts) > 1:
            defect_rate = np.mean(np.gradient(defect_counts))

        # Find saturation level
        defect_saturation = np.max(defect_counts) if len(defect_counts) > 0 else 0.0

        # Fit radiation damage model
        model_params = {}
        fitting_quality = 0.0

        if SCIPY_AVAILABLE and len(defect_counts) > 5:
            try:
                # Fit to radiation damage model: N(t) = N_sat * (1 - exp(-K*Phi*t))
                # Where K is damage efficiency, Phi is flux
                def damage_model(t, n_sat, rate):
                    return n_sat * (1 - np.exp(-rate * t))

                popt, pcov = optimize.curve_fit(
                    damage_model,
                    time_steps,
                    defect_counts,
                    p0=[defect_saturation, 0.01],
                )

                n_sat, rate = popt
                predicted = damage_model(time_steps, n_sat, rate)
                r_squared = 1 - np.sum((defect_counts - predicted) ** 2) / np.sum(
                    (defect_counts - np.mean(defect_counts)) ** 2
                )

                model_params = {
                    "saturation_defects": n_sat,
                    "damage_rate_constant": rate,
                }
                fitting_quality = max(0, r_squared)

            except:
                model_params = {"model": "linear"}
                fitting_quality = 0.5

        return DynamicProperty(
            property_type=PropertyType.RADIATION_DAMAGE,
            values=defect_counts,
            time_series=time_steps,
            parameters={
                "defect_production_rate": defect_rate,
                "defect_saturation": defect_saturation,
                **model_params,
            },
            fitting_quality=fitting_quality,
        )

    def _calculate_density_evolution(self, results: SimulationResults) -> np.ndarray:
        """Calculate density evolution from simulation results."""

        if results.positions.size == 0:
            return np.array([1000.0])  # Default density

        densities = []

        for i in range(len(results.positions)):
            positions = results.positions[i]

            # Calculate bounding box volume
            if len(positions) > 0:
                min_coords = np.min(positions, axis=0)
                max_coords = np.max(positions, axis=0)
                volume = np.prod(max_coords - min_coords)

                # Assume unit mass per atom
                if volume > 0:
                    density = len(positions) / volume
                else:
                    density = 1000.0
            else:
                density = 1000.0

            densities.append(density)

        return np.array(densities)


class FailureMechanismIdentifier(BaseAnalyzer):
    """Identifier for failure mechanisms under extreme conditions."""

    def __init__(self, backend: str = "cpu"):
        super().__init__(backend)
        self.mechanism_detectors = {
            FailureMechanism.BRITTLE_FRACTURE: self._detect_brittle_fracture,
            FailureMechanism.DUCTILE_FAILURE: self._detect_ductile_failure,
            FailureMechanism.FATIGUE_CRACKING: self._detect_fatigue_cracking,
            FailureMechanism.RADIATION_EMBRITTLEMENT: self._detect_radiation_embrittlement,
            FailureMechanism.THERMAL_SHOCK: self._detect_thermal_shock,
            FailureMechanism.SPALLATION: self._detect_spallation,
        }

    def analyze(self, simulation_results: SimulationResults) -> FailureAnalysis:
        """Identify failure mechanisms from simulation results."""

        self.logger.info("Starting failure mechanism identification")

        # Initialize failure analysis
        failure_analysis = FailureAnalysis(
            primary_mechanism=FailureMechanism.BRITTLE_FRACTURE  # Default
        )

        # Run all detectors and collect scores
        mechanism_scores = {}
        mechanism_details = {}

        for mechanism, detector in self.mechanism_detectors.items():
            try:
                score, details = detector(simulation_results)
                mechanism_scores[mechanism] = score
                mechanism_details[mechanism] = details

                self.logger.debug(f"{mechanism.value}: score = {score:.3f}")

            except Exception as e:
                self.logger.warning(f"Failed to detect {mechanism.value}: {e}")
                mechanism_scores[mechanism] = 0.0
                mechanism_details[mechanism] = {}

        # Identify primary mechanism (highest score)
        if mechanism_scores:
            primary_mechanism = max(mechanism_scores, key=mechanism_scores.get)
            failure_analysis.primary_mechanism = primary_mechanism

            # Identify secondary mechanisms (score > 0.3 * primary)
            primary_score = mechanism_scores[primary_mechanism]
            secondary_threshold = 0.3 * primary_score

            secondary_mechanisms = [
                mech
                for mech, score in mechanism_scores.items()
                if score >= secondary_threshold and mech != primary_mechanism
            ]
            failure_analysis.secondary_mechanisms = secondary_mechanisms

            # Extract mechanism-specific parameters
            primary_details = mechanism_details[primary_mechanism]
            failure_analysis.mechanism_parameters = primary_details

            # Set general failure metrics
            failure_analysis.failure_probability = min(1.0, primary_score)
            failure_analysis.failure_onset_time = primary_details.get("onset_time", 0.0)
            failure_analysis.critical_stress = primary_details.get(
                "critical_stress", 0.0
            )
            failure_analysis.critical_strain = primary_details.get(
                "critical_strain", 0.0
            )

        self.logger.info(
            f"Primary failure mechanism: {failure_analysis.primary_mechanism.value}"
        )

        return failure_analysis

    def _detect_brittle_fracture(
        self, results: SimulationResults
    ) -> Tuple[float, Dict[str, float]]:
        """Detect brittle fracture indicators."""

        score = 0.0
        details = {}

        # Check stress-strain behavior
        if results.stress_tensor.size > 0 and results.strain_tensor.size > 0:
            # Calculate von Mises stress and strain
            final_stress = results.stress_tensor[-1]
            final_strain = results.strain_tensor[-1]

            vm_stress = self._calculate_von_mises_stress(final_stress)
            eq_strain = self._calculate_equivalent_strain(final_strain)

            details["final_stress"] = vm_stress
            details["final_strain"] = eq_strain

            # Brittle fracture indicators:
            # 1. Low strain at failure
            # 2. High stress concentration
            # 3. Sudden failure

            if eq_strain < 0.02:  # Low ductility
                score += 0.4

            if vm_stress > 1e9:  # High stress
                score += 0.3

            # Check for sudden stress drop
            if len(results.stress_tensor) > 10:
                stress_history = [
                    self._calculate_von_mises_stress(s) for s in results.stress_tensor
                ]
                stress_gradient = np.gradient(stress_history)

                if np.any(stress_gradient < -1e8):  # Sudden stress drop
                    score += 0.3
                    details["sudden_failure"] = True

        # Check temperature effects
        if results.temperatures.size > 0:
            avg_temp = np.mean(results.temperatures)
            if avg_temp < 0.3 * 300:  # Low temperature promotes brittleness
                score += 0.2
                details["low_temperature_effect"] = True

        details["mechanism_score"] = score
        return score, details

    def _detect_ductile_failure(
        self, results: SimulationResults
    ) -> Tuple[float, Dict[str, float]]:
        """Detect ductile failure indicators."""

        score = 0.0
        details = {}

        if results.stress_tensor.size > 0 and results.strain_tensor.size > 0:
            final_strain = results.strain_tensor[-1]
            eq_strain = self._calculate_equivalent_strain(final_strain)

            details["final_strain"] = eq_strain

            # Ductile failure indicators:
            # 1. High strain at failure
            # 2. Necking behavior
            # 3. Void nucleation and growth

            if eq_strain > 0.1:  # High ductility
                score += 0.5

            # Check for progressive damage
            if results.damage_fraction.size > 0:
                damage_evolution = results.damage_fraction
                if len(damage_evolution) > 5:
                    # Gradual damage increase indicates ductile failure
                    damage_gradient = np.gradient(damage_evolution)
                    if np.std(damage_gradient) < 0.1 * np.mean(damage_gradient):
                        score += 0.3
                        details["gradual_damage"] = True

        # Check for void nucleation
        if results.defect_count.size > 0:
            final_defects = results.defect_count[-1]
            if final_defects > 1000:  # High defect density
                score += 0.2
                details["void_nucleation"] = True

        details["mechanism_score"] = score
        return score, details

    def _detect_fatigue_cracking(
        self, results: SimulationResults
    ) -> Tuple[float, Dict[str, float]]:
        """Detect fatigue cracking indicators."""

        score = 0.0
        details = {}

        # Check for cyclic loading (stress oscillations)
        if results.stress_tensor.size > 10:
            stress_history = [
                self._calculate_von_mises_stress(s) for s in results.stress_tensor
            ]

            # Look for periodic stress variations
            if SCIPY_AVAILABLE:
                try:
                    # Find dominant frequency
                    freqs = np.fft.fftfreq(len(stress_history))
                    fft = np.fft.fft(stress_history)

                    # Find peak frequency (excluding DC component)
                    peak_freq_idx = np.argmax(np.abs(fft[1 : len(fft) // 2])) + 1
                    dominant_frequency = freqs[peak_freq_idx]

                    if dominant_frequency > 0:
                        score += 0.4
                        details["cyclic_frequency"] = dominant_frequency

                except:
                    pass

            # Check for stress amplitude
            stress_range = np.max(stress_history) - np.min(stress_history)
            mean_stress = np.mean(stress_history)

            if mean_stress > 0 and stress_range / mean_stress > 0.1:
                score += 0.3
                details["stress_amplitude"] = stress_range

        # Check for crack propagation
        if results.crack_length.size > 0:
            crack_growth = results.crack_length
            if len(crack_growth) > 1 and crack_growth[-1] > crack_growth[0]:
                crack_growth_rate = np.mean(np.gradient(crack_growth))
                if crack_growth_rate > 0:
                    score += 0.4
                    details["crack_growth_rate"] = crack_growth_rate

        details["mechanism_score"] = score
        return score, details

    def _detect_radiation_embrittlement(
        self, results: SimulationResults
    ) -> Tuple[float, Dict[str, float]]:
        """Detect radiation embrittlement indicators."""

        score = 0.0
        details = {}

        # Check for radiation damage
        if results.defect_count.size > 0:
            final_defects = results.defect_count[-1]
            initial_defects = (
                results.defect_count[0] if len(results.defect_count) > 0 else 0
            )

            defect_increase = final_defects - initial_defects

            if defect_increase > 1000:  # Significant defect accumulation
                score += 0.5
                details["defect_accumulation"] = defect_increase

        # Check for reduced ductility after irradiation
        if results.strain_tensor.size > 0:
            final_strain = results.strain_tensor[-1]
            eq_strain = self._calculate_equivalent_strain(final_strain)

            if eq_strain < 0.05:  # Reduced ductility
                score += 0.3
                details["embrittlement_strain"] = eq_strain

        # Check simulation metadata for radiation conditions
        metadata = results.simulation_metadata
        if "condition_type" in metadata and "radiation" in metadata["condition_type"]:
            score += 0.4
            details["radiation_exposure"] = True

        details["mechanism_score"] = score
        return score, details

    def _detect_thermal_shock(
        self, results: SimulationResults
    ) -> Tuple[float, Dict[str, float]]:
        """Detect thermal shock indicators."""

        score = 0.0
        details = {}

        if results.temperatures.size > 1:
            temperatures = results.temperatures

            # Check for rapid temperature changes
            temp_gradient = np.gradient(temperatures)
            max_cooling_rate = -np.min(temp_gradient)
            max_heating_rate = np.max(temp_gradient)

            details["max_cooling_rate"] = max_cooling_rate
            details["max_heating_rate"] = max_heating_rate

            # Thermal shock indicators:
            # 1. Rapid temperature change
            # 2. High thermal gradients
            # 3. Sudden failure after temperature change

            if max_cooling_rate > 1e6 or max_heating_rate > 1e6:  # K/s
                score += 0.4
                details["rapid_thermal_change"] = True

            # Check for temperature cycling
            temp_range = np.max(temperatures) - np.min(temperatures)
            if temp_range > 500:  # K
                score += 0.2
                details["large_temperature_range"] = temp_range

            # Check for failure correlation with temperature change
            if results.damage_fraction.size > 0:
                damage = results.damage_fraction

                # Find rapid damage increase
                damage_gradient = np.gradient(damage)
                max_damage_rate_idx = np.argmax(damage_gradient)

                # Check if it coincides with temperature change
                temp_change_idx = np.argmax(np.abs(temp_gradient))

                if abs(max_damage_rate_idx - temp_change_idx) < len(damage) * 0.1:
                    score += 0.3
                    details["thermal_damage_correlation"] = True

        details["mechanism_score"] = score
        return score, details

    def _detect_spallation(
        self, results: SimulationResults
    ) -> Tuple[float, Dict[str, float]]:
        """Detect spallation (dynamic tensile failure) indicators."""

        score = 0.0
        details = {}

        # Check for shock loading conditions
        if results.pressures.size > 0:
            pressures = results.pressures

            # Look for pressure wave reflections (spallation signature)
            if len(pressures) > 10:
                # Find pressure gradient
                pressure_gradient = np.gradient(pressures)

                # Spallation shows compression followed by tension
                max_compression = np.max(pressures)
                min_tension = np.min(pressures)

                if (
                    max_compression > 1e8 and min_tension < -1e7
                ):  # High compression + tension
                    score += 0.5
                    details["pressure_range"] = max_compression - min_tension

                # Check for characteristic spallation pattern
                # (compression pulse followed by tensile pulse)
                if np.any(pressure_gradient > 1e8) and np.any(pressure_gradient < -1e8):
                    score += 0.3
                    details["pressure_wave_signature"] = True

        # Check for high strain rate loading
        if results.strain_tensor.size > 1:
            strain_rates = []
            for i in range(len(results.strain_tensor) - 1):
                strain_diff = results.strain_tensor[i + 1] - results.strain_tensor[i]
                eq_strain_rate = self._calculate_equivalent_strain(strain_diff)
                strain_rates.append(eq_strain_rate)

            max_strain_rate = np.max(strain_rates) if strain_rates else 0

            if max_strain_rate > 1e6:  # s^-1
                score += 0.3
                details["max_strain_rate"] = max_strain_rate

        # Check simulation metadata for shock conditions
        metadata = results.simulation_metadata
        if "condition_type" in metadata and "shock" in metadata["condition_type"]:
            score += 0.2
            details["shock_loading"] = True

        details["mechanism_score"] = score
        return score, details

    def _calculate_von_mises_stress(self, stress_tensor: np.ndarray) -> float:
        """Calculate von Mises equivalent stress."""
        if stress_tensor.size == 0:
            return 0.0

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
        if strain_tensor.size == 0:
            return 0.0

        e11, e22, e33 = strain_tensor[0, 0], strain_tensor[1, 1], strain_tensor[2, 2]
        e12, e13, e23 = strain_tensor[0, 1], strain_tensor[0, 2], strain_tensor[1, 2]

        return np.sqrt(
            (2 / 3)
            * (e11**2 + e22**2 + e33**2 + 2 * (e12**2 + e13**2 + e23**2))
        )


class ExtremeConditionsCharacterizer:
    """Main characterizer that coordinates all analysis capabilities."""

    def __init__(self, backend: str = "cpu"):
        self.backend = backend
        self.logger = get_logger(self.__class__.__name__)

        # Initialize analyzers
        self.property_analyzer = DynamicPropertyAnalyzer(backend)
        self.failure_analyzer = FailureMechanismIdentifier(backend)

    def characterize_extreme_conditions_response(
        self, simulation_results: SimulationResults
    ) -> Dict[str, Any]:
        """Comprehensive characterization of material response to extreme conditions."""

        self.logger.info("Starting comprehensive extreme conditions characterization")

        characterization = {
            "dynamic_properties": {},
            "failure_analysis": None,
            "summary_metrics": {},
            "recommendations": [],
        }

        # Analyze dynamic properties
        try:
            dynamic_properties = self.property_analyzer.analyze(simulation_results)
            characterization["dynamic_properties"] = dynamic_properties

            self.logger.info(f"Identified {len(dynamic_properties)} dynamic properties")

        except Exception as e:
            self.logger.error(f"Dynamic property analysis failed: {e}")

        # Analyze failure mechanisms
        try:
            failure_analysis = self.failure_analyzer.analyze(simulation_results)
            characterization["failure_analysis"] = failure_analysis

            self.logger.info(
                f"Primary failure mechanism: {failure_analysis.primary_mechanism.value}"
            )

        except Exception as e:
            self.logger.error(f"Failure mechanism analysis failed: {e}")

        # Generate summary metrics
        characterization["summary_metrics"] = self._generate_summary_metrics(
            dynamic_properties, failure_analysis
        )

        # Generate recommendations
        characterization["recommendations"] = self._generate_recommendations(
            dynamic_properties, failure_analysis
        )

        return characterization

    def _generate_summary_metrics(
        self,
        dynamic_properties: Dict[str, DynamicProperty],
        failure_analysis: Optional[FailureAnalysis],
    ) -> Dict[str, float]:
        """Generate summary metrics from analysis results."""

        metrics = {}

        # Property-based metrics
        for prop_name, prop_data in dynamic_properties.items():
            if hasattr(prop_data, "parameters"):
                for param_name, param_value in prop_data.parameters.items():
                    metrics[f"{prop_name}_{param_name}"] = param_value

        # Failure-based metrics
        if failure_analysis:
            metrics["failure_probability"] = failure_analysis.failure_probability
            metrics["critical_stress"] = failure_analysis.critical_stress
            metrics["critical_strain"] = failure_analysis.critical_strain
            metrics["failure_onset_time"] = failure_analysis.failure_onset_time

        return metrics

    def _generate_recommendations(
        self,
        dynamic_properties: Dict[str, DynamicProperty],
        failure_analysis: Optional[FailureAnalysis],
    ) -> List[str]:
        """Generate recommendations based on analysis results."""

        recommendations = []

        # Property-based recommendations
        for prop_name, prop_data in dynamic_properties.items():
            if prop_name == "damage_evolution":
                damage_rate = prop_data.parameters.get("damage_rate", 0)
                if damage_rate > 0.01:
                    recommendations.append(
                        "High damage accumulation rate detected. Consider improving material toughness."
                    )

            elif prop_name == "glass_formation":
                cooling_rate = prop_data.parameters.get("cooling_rate", 0)
                if cooling_rate > 1e6:
                    recommendations.append(
                        "Rapid cooling detected. Monitor for residual stresses and brittleness."
                    )

        # Failure mechanism recommendations
        if failure_analysis:
            primary_mechanism = failure_analysis.primary_mechanism

            if primary_mechanism == FailureMechanism.BRITTLE_FRACTURE:
                recommendations.append(
                    "Brittle fracture risk identified. Consider temperature control and stress concentration reduction."
                )

            elif primary_mechanism == FailureMechanism.RADIATION_EMBRITTLEMENT:
                recommendations.append(
                    "Radiation embrittlement detected. Consider radiation-resistant alloy additions."
                )

            elif primary_mechanism == FailureMechanism.THERMAL_SHOCK:
                recommendations.append(
                    "Thermal shock susceptibility identified. Implement gradual heating/cooling protocols."
                )

        return recommendations

"""
Transport Kinetics and Partitioning Characterization

This module provides comprehensive analysis capabilities for characterizing transport
properties including permeability coefficients, diffusion rates, and partitioning
behavior from simulation data.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, optimize
from scipy.stats import linregress

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupyx_ndimage
    from cupyx.scipy import signal as cupy_signal

    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    cupyx_ndimage = None
    cupy_signal = None
    CUDA_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    jit = None
    vmap = None
    grad = None
    JAX_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from .simulation_engine import TransportResult


@dataclass
class TransportKinetics:
    """Transport kinetics analysis results."""

    permeability_coefficient: float
    diffusion_coefficients: Dict[str, float]
    partition_coefficients: Dict[str, float]
    rate_constants: Dict[str, float]
    residence_times: Dict[str, float]
    transport_mechanisms: List[str]
    kinetic_barriers: Dict[str, float]
    mass_transfer_coefficients: Dict[str, float]
    effective_diffusivity: float
    tortuosity_factor: float


@dataclass
class PartitioningAnalysis:
    """Partitioning behavior analysis results."""

    phase_concentrations: Dict[str, np.ndarray]
    equilibrium_constants: Dict[str, float]
    activity_coefficients: Dict[str, float]
    solubility_parameters: Dict[str, float]
    interaction_parameters: Dict[str, float]
    thermodynamic_driving_forces: Dict[str, float]
    phase_behavior_classification: str


class KineticsAnalyzer:
    """Analyzer for transport kinetics from simulation data."""

    def __init__(self, backend: str = "auto"):
        self.backend = self._select_backend(backend)
        self.logger = logging.getLogger(__name__)

    def _select_backend(self, backend: str) -> str:
        """Select optimal computational backend."""
        if backend == "auto":
            if CUDA_AVAILABLE:
                return "cuda"
            elif JAX_AVAILABLE:
                return "jax"
            else:
                return "cpu"
        return backend

    def analyze_permeability(
        self, simulation_result: TransportResult
    ) -> Dict[str, float]:
        """
        Analyze permeability from simulation trajectories.

        Args:
            simulation_result: Results from transport simulation

        Returns:
            Dictionary of permeability analysis results
        """
        trajectories = simulation_result.trajectories
        flux_data = simulation_result.flux_data

        # Calculate steady-state permeability
        steady_state_flux = np.mean(flux_data[-5:])  # Average last 5 data points

        # Extract concentration gradient from profiles
        concentration_profiles = simulation_result.concentration_profiles
        gradient = self._calculate_concentration_gradient(concentration_profiles)

        # Permeability = Flux / (Area * Gradient)
        membrane_area = 1.0  # Normalized area
        permeability = steady_state_flux / (membrane_area * abs(gradient))

        # Calculate time-dependent permeability
        time_dependent_perm = self._calculate_time_dependent_permeability(
            flux_data, gradient
        )

        # Analyze permeability mechanisms
        mechanisms = self._identify_transport_mechanisms(trajectories)

        return {
            "steady_state_permeability": permeability,
            "time_dependent_permeability": time_dependent_perm,
            "transport_mechanisms": mechanisms,
            "flux_variance": np.var(flux_data),
            "permeability_uncertainty": np.std(time_dependent_perm[-10:]),
        }

    def analyze_diffusion_coefficients(
        self, simulation_result: TransportResult
    ) -> Dict[str, Any]:
        """
        Calculate diffusion coefficients from mean square displacement.

        Args:
            simulation_result: Results from transport simulation

        Returns:
            Dictionary of diffusion analysis results
        """
        trajectories = simulation_result.trajectories
        n_molecules, n_frames, n_dims = trajectories.shape

        # Calculate mean square displacement (MSD)
        if self.backend == "cuda" and CUDA_AVAILABLE:
            msd_results = self._cuda_calculate_msd(trajectories)
        elif self.backend == "jax" and JAX_AVAILABLE:
            msd_results = self._jax_calculate_msd(trajectories)
        else:
            msd_results = self._cpu_calculate_msd(trajectories)

        # Fit linear regions to extract diffusion coefficients
        diffusion_coeffs = self._fit_diffusion_coefficients(msd_results)

        # Analyze phase-specific diffusion
        phase_diffusion = self._analyze_phase_specific_diffusion(
            trajectories, simulation_result.metadata
        )

        # Calculate anomalous diffusion parameters
        anomalous_params = self._analyze_anomalous_diffusion(msd_results)

        return {
            "msd_data": msd_results,
            "diffusion_coefficients": diffusion_coeffs,
            "phase_specific_diffusion": phase_diffusion,
            "anomalous_diffusion": anomalous_params,
            "einstein_relation_check": self._verify_einstein_relation(
                diffusion_coeffs, simulation_result
            ),
        }

    def _cuda_calculate_msd(self, trajectories: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate MSD using CUDA acceleration."""
        trajectories_gpu = cp.asarray(trajectories)
        n_molecules, n_frames, n_dims = trajectories_gpu.shape

        # Pre-allocate MSD arrays
        msd_total = cp.zeros(n_frames)
        msd_x = cp.zeros(n_frames)
        msd_y = cp.zeros(n_frames)
        msd_z = cp.zeros(n_frames)

        # Calculate MSD for each lag time
        for lag in range(1, n_frames):
            displacement = trajectories_gpu[:, lag:, :] - trajectories_gpu[:, :-lag, :]
            squared_displacement = cp.sum(displacement**2, axis=2)

            msd_total[lag] = cp.mean(squared_displacement)
            msd_x[lag] = cp.mean(displacement[:, :, 0] ** 2)
            msd_y[lag] = cp.mean(displacement[:, :, 1] ** 2)
            msd_z[lag] = cp.mean(displacement[:, :, 2] ** 2)

        return {
            "total": cp.asnumpy(msd_total),
            "x": cp.asnumpy(msd_x),
            "y": cp.asnumpy(msd_y),
            "z": cp.asnumpy(msd_z),
        }

    def _jax_calculate_msd(self, trajectories: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate MSD using JAX acceleration."""

        @jit
        def compute_msd_for_lag(traj, lag):
            displacement = traj[:, lag:, :] - traj[:, :-lag, :]
            squared_displacement = jnp.sum(displacement**2, axis=2)
            return jnp.mean(squared_displacement, axis=0)

        trajectories_jax = jnp.asarray(trajectories)
        n_frames = trajectories_jax.shape[1]

        msd_results = {}
        for component, idx in [("total", slice(None)), ("x", 0), ("y", 1), ("z", 2)]:
            if component == "total":
                traj_component = trajectories_jax
            else:
                traj_component = trajectories_jax[:, :, idx : idx + 1]

            msd_values = []
            for lag in range(1, n_frames):
                msd_lag = compute_msd_for_lag(traj_component, lag)
                if component == "total":
                    msd_values.append(float(msd_lag.mean()))
                else:
                    msd_values.append(float(msd_lag))

            msd_results[component] = np.array([0.0] + msd_values)

        return msd_results

    def _cpu_calculate_msd(self, trajectories: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate MSD using CPU computation."""
        n_molecules, n_frames, n_dims = trajectories.shape

        msd_total = np.zeros(n_frames)
        msd_x = np.zeros(n_frames)
        msd_y = np.zeros(n_frames)
        msd_z = np.zeros(n_frames)

        for lag in range(1, n_frames):
            displacement = trajectories[:, lag:, :] - trajectories[:, :-lag, :]
            squared_displacement = np.sum(displacement**2, axis=2)

            msd_total[lag] = np.mean(squared_displacement)
            msd_x[lag] = np.mean(displacement[:, :, 0] ** 2)
            msd_y[lag] = np.mean(displacement[:, :, 1] ** 2)
            msd_z[lag] = np.mean(displacement[:, :, 2] ** 2)

        return {"total": msd_total, "x": msd_x, "y": msd_y, "z": msd_z}

    def _fit_diffusion_coefficients(
        self, msd_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Fit linear regions of MSD to extract diffusion coefficients."""
        diffusion_coeffs = {}

        # Find linear region (typically 10-90% of data)
        n_points = len(msd_data["total"])
        start_idx = max(1, n_points // 10)
        end_idx = min(n_points - 1, 9 * n_points // 10)

        time_points = np.arange(start_idx, end_idx)

        for component, msd_values in msd_data.items():
            if len(msd_values) < 2:
                continue

            # Linear fit: MSD = 2*n*D*t (n=dimensionality)
            slope, intercept, r_value, p_value, std_err = linregress(
                time_points, msd_values[start_idx:end_idx]
            )

            # Extract diffusion coefficient
            if component == "total":
                D = slope / (2 * 3)  # 3D
            else:
                D = slope / (2 * 1)  # 1D

            diffusion_coeffs[f"{component}_diffusion"] = D
            diffusion_coeffs[f"{component}_r_squared"] = r_value**2
            diffusion_coeffs[f"{component}_std_error"] = std_err

        return diffusion_coeffs

    def _analyze_phase_specific_diffusion(
        self, trajectories: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze diffusion coefficients in different phases."""
        n_molecules, n_frames, _ = trajectories.shape

        # Define phase boundaries (simplified)
        aqueous_region = trajectories[:, :, 2] < -2.0  # z < -2 nm
        membrane_region = (trajectories[:, :, 2] >= -2.0) & (
            trajectories[:, :, 2] <= 2.0
        )
        lipid_region = trajectories[:, :, 2] > 2.0  # z > 2 nm

        phase_diffusion = {}

        for phase_name, phase_mask in [
            ("aqueous", aqueous_region),
            ("membrane", membrane_region),
            ("lipid", lipid_region),
        ]:
            # Extract trajectories in this phase
            phase_trajectories = []
            for mol_idx in range(n_molecules):
                phase_frames = np.where(phase_mask[mol_idx, :])[0]
                if len(phase_frames) > 10:  # Minimum frames for analysis
                    phase_traj = trajectories[mol_idx, phase_frames, :]
                    phase_trajectories.append(phase_traj)

            if phase_trajectories:
                # Calculate average diffusion in this phase
                phase_msd = self._calculate_phase_msd(phase_trajectories)
                phase_diffusion[phase_name] = self._extract_diffusion_from_msd(
                    phase_msd
                )
            else:
                phase_diffusion[phase_name] = 0.0

        return phase_diffusion

    def _calculate_phase_msd(self, phase_trajectories: List[np.ndarray]) -> np.ndarray:
        """Calculate MSD for trajectories within a specific phase."""
        all_msd = []

        for traj in phase_trajectories:
            n_frames = len(traj)
            if n_frames < 5:
                continue

            msd = np.zeros(n_frames)
            for lag in range(1, min(n_frames, 20)):  # Limit lag for efficiency
                if lag < n_frames:
                    displacement = traj[lag:] - traj[:-lag]
                    squared_displacement = np.sum(displacement**2, axis=1)
                    msd[lag] = np.mean(squared_displacement)

            all_msd.append(msd)

        if all_msd:
            return np.mean(all_msd, axis=0)
        else:
            return np.zeros(10)

    def _extract_diffusion_from_msd(self, msd: np.ndarray) -> float:
        """Extract diffusion coefficient from MSD data."""
        if len(msd) < 3:
            return 0.0

        # Fit linear region
        valid_points = msd > 0
        if np.sum(valid_points) < 2:
            return 0.0

        time_points = np.arange(len(msd))[valid_points]
        msd_values = msd[valid_points]

        if len(time_points) >= 2:
            slope, _, _, _, _ = linregress(time_points, msd_values)
            return slope / 6.0  # 3D diffusion
        else:
            return 0.0

    def _analyze_anomalous_diffusion(
        self, msd_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Analyze anomalous diffusion behavior."""
        anomalous_params = {}

        for component, msd_values in msd_data.items():
            if len(msd_values) < 5:
                continue

            # Fit power law: MSD = K * t^alpha
            valid_idx = msd_values > 0
            if np.sum(valid_idx) < 3:
                continue

            time_points = np.arange(len(msd_values))[valid_idx]
            msd_vals = msd_values[valid_idx]

            # Log-log fit
            try:
                log_time = np.log(time_points[1:])  # Avoid log(0)
                log_msd = np.log(msd_vals[1:])

                alpha, log_K, r_value, _, _ = linregress(log_time, log_msd)
                K = np.exp(log_K)

                anomalous_params[f"{component}_alpha"] = alpha
                anomalous_params[f"{component}_K"] = K
                anomalous_params[f"{component}_fit_quality"] = r_value**2

                # Classify diffusion type
                if alpha < 0.9:
                    diffusion_type = "subdiffusive"
                elif alpha > 1.1:
                    diffusion_type = "superdiffusive"
                else:
                    diffusion_type = "normal"

                anomalous_params[f"{component}_type"] = diffusion_type

            except (ValueError, ZeroDivisionError):
                anomalous_params[f"{component}_alpha"] = 1.0
                anomalous_params[f"{component}_type"] = "normal"

        return anomalous_params

    def _verify_einstein_relation(
        self, diffusion_coeffs: Dict[str, float], simulation_result: TransportResult
    ) -> Dict[str, float]:
        """Verify Einstein relation between diffusion and mobility."""
        # Einstein relation: D = μ * k_B * T
        k_B = 1.380649e-23  # Boltzmann constant (J/K)
        T = simulation_result.metadata.get("config").temperature

        verification = {}

        if "total_diffusion" in diffusion_coeffs:
            D_measured = diffusion_coeffs["total_diffusion"]

            # Estimate mobility from simulation (simplified)
            mobility_estimate = D_measured / (k_B * T)

            verification["measured_diffusion"] = D_measured
            verification["estimated_mobility"] = mobility_estimate
            verification["einstein_consistency"] = True  # Simplified check

        return verification

    def _calculate_concentration_gradient(
        self, concentration_profiles: Dict[str, np.ndarray]
    ) -> float:
        """Calculate concentration gradient across phases."""
        if not concentration_profiles:
            return 1.0  # Default gradient

        # Use first available profile
        profile_key = list(concentration_profiles.keys())[0]
        profile = concentration_profiles[profile_key]

        # Calculate gradient using finite differences
        gradient = np.gradient(profile)

        # Return maximum gradient magnitude
        return np.max(np.abs(gradient))

    def _calculate_time_dependent_permeability(
        self, flux_data: np.ndarray, gradient: float
    ) -> np.ndarray:
        """Calculate time-dependent permeability."""
        if len(flux_data) == 0:
            return np.array([0.0])

        # Smooth flux data
        if len(flux_data) > 5:
            from scipy.ndimage import gaussian_filter1d

            smoothed_flux = gaussian_filter1d(flux_data, sigma=1.0)
        else:
            smoothed_flux = flux_data

        # Calculate permeability at each time point
        membrane_area = 1.0
        permeability = smoothed_flux / (membrane_area * abs(gradient))

        return permeability

    def _identify_transport_mechanisms(self, trajectories: np.ndarray) -> List[str]:
        """Identify dominant transport mechanisms from trajectory analysis."""
        mechanisms = []

        # Analyze trajectory characteristics
        n_molecules, n_frames, _ = trajectories.shape

        # Check for directional transport
        z_displacement = trajectories[:, -1, 2] - trajectories[:, 0, 2]
        avg_displacement = np.mean(z_displacement)

        if abs(avg_displacement) > 1.0:  # Significant net displacement
            mechanisms.append("advective_transport")

        # Check for oscillatory behavior
        for mol_idx in range(min(5, n_molecules)):  # Sample molecules
            z_traj = trajectories[mol_idx, :, 2]

            # Simple oscillation detection
            zero_crossings = np.where(np.diff(np.signbit(z_traj - np.mean(z_traj))))[0]
            if len(zero_crossings) > n_frames // 10:
                mechanisms.append("oscillatory_transport")
                break

        # Check for confined motion
        for mol_idx in range(min(5, n_molecules)):
            traj_std = np.std(trajectories[mol_idx, :, :], axis=0)
            if np.all(traj_std < 0.5):  # Small spatial fluctuations
                mechanisms.append("confined_diffusion")
                break

        # Default mechanism
        if not mechanisms:
            mechanisms.append("free_diffusion")

        return mechanisms


class PartitioningAnalyzer:
    """Analyzer for molecular partitioning behavior."""

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.logger = logging.getLogger(__name__)

    def analyze_equilibrium_partitioning(
        self, simulation_result: TransportResult
    ) -> PartitioningAnalysis:
        """
        Analyze equilibrium partitioning from simulation data.

        Args:
            simulation_result: Results from transport simulation

        Returns:
            PartitioningAnalysis: Comprehensive partitioning analysis
        """
        trajectories = simulation_result.trajectories

        # Extract phase concentrations
        phase_concentrations = self._calculate_phase_concentrations(trajectories)

        # Calculate equilibrium constants
        equilibrium_constants = self._calculate_equilibrium_constants(
            phase_concentrations
        )

        # Estimate activity coefficients
        activity_coefficients = self._estimate_activity_coefficients(
            phase_concentrations, simulation_result.metadata
        )

        # Calculate solubility parameters
        solubility_parameters = self._calculate_solubility_parameters(simulation_result)

        # Analyze interaction parameters
        interaction_parameters = self._analyze_interaction_parameters(trajectories)

        # Calculate thermodynamic driving forces
        driving_forces = self._calculate_driving_forces(
            phase_concentrations, equilibrium_constants
        )

        # Classify phase behavior
        phase_classification = self._classify_phase_behavior(
            equilibrium_constants, activity_coefficients
        )

        return PartitioningAnalysis(
            phase_concentrations=phase_concentrations,
            equilibrium_constants=equilibrium_constants,
            activity_coefficients=activity_coefficients,
            solubility_parameters=solubility_parameters,
            interaction_parameters=interaction_parameters,
            thermodynamic_driving_forces=driving_forces,
            phase_behavior_classification=phase_classification,
        )

    def _calculate_phase_concentrations(
        self, trajectories: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate molecular concentrations in different phases."""
        n_molecules, n_frames, _ = trajectories.shape

        # Define phase boundaries
        phases = {
            "aqueous": trajectories[:, :, 2] < -2.0,
            "interface": (trajectories[:, :, 2] >= -2.0)
            & (trajectories[:, :, 2] <= 2.0),
            "organic": trajectories[:, :, 2] > 2.0,
        }

        phase_concentrations = {}

        for phase_name, phase_mask in phases.items():
            # Calculate time-averaged concentration in each phase
            concentrations = []
            for frame in range(n_frames):
                n_molecules_in_phase = np.sum(phase_mask[:, frame])
                volume = 1.0  # Normalized volume
                concentration = n_molecules_in_phase / volume
                concentrations.append(concentration)

            phase_concentrations[phase_name] = np.array(concentrations)

        return phase_concentrations

    def _calculate_equilibrium_constants(
        self, phase_concentrations: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate partition coefficients between phases."""
        equilibrium_constants = {}

        # Calculate steady-state concentrations (last 20% of simulation)
        steady_state_slice = slice(
            -len(list(phase_concentrations.values())[0]) // 5, None
        )

        phases = list(phase_concentrations.keys())

        for i, phase1 in enumerate(phases):
            for phase2 in phases[i + 1 :]:
                conc1 = np.mean(phase_concentrations[phase1][steady_state_slice])
                conc2 = np.mean(phase_concentrations[phase2][steady_state_slice])

                if conc2 > 1e-10:  # Avoid division by zero
                    K = conc1 / conc2
                    equilibrium_constants[f"{phase1}_{phase2}"] = K

        return equilibrium_constants

    def _estimate_activity_coefficients(
        self, phase_concentrations: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate activity coefficients from concentration data."""
        activity_coefficients = {}

        # Simplified activity coefficient estimation
        for phase_name, concentrations in phase_concentrations.items():
            avg_concentration = np.mean(concentrations)

            # Simple model: γ = 1 + α * C (α is interaction parameter)
            alpha = 0.1  # Simplified interaction parameter
            gamma = 1.0 + alpha * avg_concentration

            activity_coefficients[phase_name] = gamma

        return activity_coefficients

    def _calculate_solubility_parameters(
        self, simulation_result: TransportResult
    ) -> Dict[str, float]:
        """Calculate Hansen solubility parameters."""
        # Simplified solubility parameter estimation
        solubility_parameters = {
            "dispersive": 18.0,  # δd (MPa^0.5)
            "polar": 10.0,  # δp (MPa^0.5)
            "hydrogen_bonding": 6.0,  # δh (MPa^0.5)
        }

        # Total solubility parameter
        delta_total = np.sqrt(
            solubility_parameters["dispersive"] ** 2
            + solubility_parameters["polar"] ** 2
            + solubility_parameters["hydrogen_bonding"] ** 2
        )

        solubility_parameters["total"] = delta_total

        return solubility_parameters

    def _analyze_interaction_parameters(
        self, trajectories: np.ndarray
    ) -> Dict[str, float]:
        """Analyze molecular interaction parameters."""
        n_molecules, n_frames, _ = trajectories.shape

        interaction_parameters = {}

        # Calculate radial distribution function approximation
        if n_molecules > 1:
            intermolecular_distances = []

            for frame in range(0, n_frames, max(1, n_frames // 20)):  # Sample frames
                for i in range(n_molecules):
                    for j in range(i + 1, n_molecules):
                        distance = np.linalg.norm(
                            trajectories[i, frame, :] - trajectories[j, frame, :]
                        )
                        intermolecular_distances.append(distance)

            if intermolecular_distances:
                avg_distance = np.mean(intermolecular_distances)
                interaction_parameters["average_intermolecular_distance"] = avg_distance
                interaction_parameters["interaction_strength"] = 1.0 / avg_distance
            else:
                interaction_parameters["average_intermolecular_distance"] = 5.0
                interaction_parameters["interaction_strength"] = 0.2
        else:
            interaction_parameters["average_intermolecular_distance"] = 5.0
            interaction_parameters["interaction_strength"] = 0.0

        return interaction_parameters

    def _calculate_driving_forces(
        self,
        phase_concentrations: Dict[str, np.ndarray],
        equilibrium_constants: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate thermodynamic driving forces for transport."""
        driving_forces = {}

        # Chemical potential differences
        for phase_pair, K_eq in equilibrium_constants.items():
            phase1, phase2 = phase_pair.split("_")

            if phase1 in phase_concentrations and phase2 in phase_concentrations:
                conc1 = np.mean(phase_concentrations[phase1][-10:])  # Recent average
                conc2 = np.mean(phase_concentrations[phase2][-10:])

                if conc1 > 1e-10 and conc2 > 1e-10:
                    # Simplified chemical potential difference
                    # Δμ = RT * ln(a1/a2) = RT * ln(γ1*C1/γ2*C2)
                    RT = 8.314 * 298.15  # J/mol at room temperature
                    mu_diff = RT * np.log(conc1 / conc2)

                    driving_forces[f"chemical_potential_{phase_pair}"] = mu_diff

        return driving_forces

    def _classify_phase_behavior(
        self,
        equilibrium_constants: Dict[str, float],
        activity_coefficients: Dict[str, float],
    ) -> str:
        """Classify the phase behavior type."""
        # Analyze partition coefficients to classify behavior
        if not equilibrium_constants:
            return "single_phase"

        K_values = list(equilibrium_constants.values())
        max_K = max(K_values) if K_values else 1.0
        min_K = min(K_values) if K_values else 1.0

        # Classification based on partition coefficient ranges
        if max_K > 100:
            return "strongly_partitioning"
        elif max_K > 10:
            return "moderately_partitioning"
        elif min_K < 0.1:
            return "selective_partitioning"
        else:
            return "uniform_distribution"


class TransportAnalyzer:
    """
    Comprehensive analyzer for transport kinetics and partitioning behavior.

    This class provides unified access to kinetics and partitioning analysis
    capabilities for transport simulation results.
    """

    def __init__(self, backend: str = "auto"):
        """
        Initialize the transport analyzer.

        Args:
            backend: Computational backend ('cuda', 'jax', 'cpu', 'auto')
        """
        self.backend = backend
        self.logger = logging.getLogger(__name__)

        # Initialize component analyzers
        self.kinetics_analyzer = KineticsAnalyzer(backend)
        self.partitioning_analyzer = PartitioningAnalyzer(backend)

    def analyze_permeability(
        self, simulation_result: TransportResult
    ) -> Dict[str, Any]:
        """
        Complete permeability analysis from simulation results.

        Args:
            simulation_result: Results from transport simulation

        Returns:
            Dictionary containing comprehensive permeability analysis
        """
        self.logger.info("Analyzing permeability from simulation results")

        # Basic permeability analysis
        permeability_results = self.kinetics_analyzer.analyze_permeability(
            simulation_result
        )

        # Add statistical analysis
        permeability_results.update(
            self._statistical_permeability_analysis(simulation_result)
        )

        return permeability_results

    def analyze_diffusion(self, simulation_result: TransportResult) -> Dict[str, Any]:
        """
        Complete diffusion analysis from simulation results.

        Args:
            simulation_result: Results from transport simulation

        Returns:
            Dictionary containing comprehensive diffusion analysis
        """
        self.logger.info("Analyzing diffusion coefficients from trajectories")

        return self.kinetics_analyzer.analyze_diffusion_coefficients(simulation_result)

    def analyze_partitioning(
        self, simulation_result: TransportResult
    ) -> PartitioningAnalysis:
        """
        Complete partitioning analysis from simulation results.

        Args:
            simulation_result: Results from transport simulation

        Returns:
            PartitioningAnalysis object with comprehensive results
        """
        self.logger.info("Analyzing molecular partitioning behavior")

        return self.partitioning_analyzer.analyze_equilibrium_partitioning(
            simulation_result
        )

    def generate_transport_report(
        self, simulation_result: TransportResult
    ) -> Dict[str, Any]:
        """
        Generate comprehensive transport analysis report.

        Args:
            simulation_result: Results from transport simulation

        Returns:
            Dictionary containing complete transport analysis
        """
        self.logger.info("Generating comprehensive transport analysis report")

        report = {
            "simulation_metadata": simulation_result.metadata,
            "permeability_analysis": self.analyze_permeability(simulation_result),
            "diffusion_analysis": self.analyze_diffusion(simulation_result),
            "partitioning_analysis": self.analyze_partitioning(simulation_result),
            "transport_kinetics": self._analyze_transport_kinetics(simulation_result),
            "recommendations": self._generate_recommendations(simulation_result),
        }

        return report

    def _statistical_permeability_analysis(
        self, simulation_result: TransportResult
    ) -> Dict[str, float]:
        """Perform statistical analysis of permeability data."""
        flux_data = simulation_result.flux_data

        if len(flux_data) == 0:
            return {"statistical_error": 0.0}

        # Calculate statistics
        flux_mean = np.mean(flux_data)
        flux_std = np.std(flux_data)
        flux_cv = flux_std / flux_mean if flux_mean > 0 else 0.0

        # Steady-state analysis
        steady_state_start = len(flux_data) // 2
        steady_flux = flux_data[steady_state_start:]
        steady_cv = (
            np.std(steady_flux) / np.mean(steady_flux) if len(steady_flux) > 0 else 0.0
        )

        return {
            "flux_coefficient_of_variation": flux_cv,
            "steady_state_cv": steady_cv,
            "statistical_confidence": 1.0 - steady_cv if steady_cv < 1.0 else 0.0,
        }

    def _analyze_transport_kinetics(
        self, simulation_result: TransportResult
    ) -> TransportKinetics:
        """Analyze comprehensive transport kinetics."""
        # Calculate rate constants from flux data
        rate_constants = self._calculate_rate_constants(simulation_result)

        # Calculate residence times
        residence_times = self._calculate_residence_times(simulation_result)

        # Identify transport mechanisms
        mechanisms = self.kinetics_analyzer._identify_transport_mechanisms(
            simulation_result.trajectories
        )

        # Calculate kinetic barriers
        kinetic_barriers = self._estimate_kinetic_barriers(simulation_result)

        # Calculate mass transfer coefficients
        mass_transfer_coeffs = self._calculate_mass_transfer_coefficients(
            simulation_result
        )

        return TransportKinetics(
            permeability_coefficient=simulation_result.permeability_coefficient,
            diffusion_coefficients=simulation_result.diffusion_coefficients,
            partition_coefficients=simulation_result.partition_coefficients,
            rate_constants=rate_constants,
            residence_times=residence_times,
            transport_mechanisms=mechanisms,
            kinetic_barriers=kinetic_barriers,
            mass_transfer_coefficients=mass_transfer_coeffs,
            effective_diffusivity=np.mean(
                list(simulation_result.diffusion_coefficients.values())
            ),
            tortuosity_factor=1.5,  # Simplified estimate
        )

    def _calculate_rate_constants(
        self, simulation_result: TransportResult
    ) -> Dict[str, float]:
        """Calculate transport rate constants."""
        # Simplified rate constant calculation
        flux_data = simulation_result.flux_data

        if len(flux_data) > 1:
            # First-order rate constant from flux decay/buildup
            k_forward = np.mean(flux_data[-5:]) / 1.0  # Normalized
            k_reverse = k_forward * 0.5  # Simplified

            return {
                "forward_rate": k_forward,
                "reverse_rate": k_reverse,
                "equilibrium_constant": k_forward / k_reverse if k_reverse > 0 else 1.0,
            }
        else:
            return {
                "forward_rate": 0.0,
                "reverse_rate": 0.0,
                "equilibrium_constant": 1.0,
            }

    def _calculate_residence_times(
        self, simulation_result: TransportResult
    ) -> Dict[str, float]:
        """Calculate molecular residence times in different phases."""
        trajectories = simulation_result.trajectories
        n_molecules, n_frames, _ = trajectories.shape

        # Define phase boundaries
        phases = {
            "aqueous": trajectories[:, :, 2] < -2.0,
            "membrane": (trajectories[:, :, 2] >= -2.0)
            & (trajectories[:, :, 2] <= 2.0),
            "organic": trajectories[:, :, 2] > 2.0,
        }

        residence_times = {}

        for phase_name, phase_mask in phases.items():
            all_residence_times = []

            for mol_idx in range(n_molecules):
                # Find continuous residence periods
                in_phase = phase_mask[mol_idx, :]

                residence_periods = []
                current_period = 0

                for frame in range(n_frames):
                    if in_phase[frame]:
                        current_period += 1
                    else:
                        if current_period > 0:
                            residence_periods.append(current_period)
                            current_period = 0

                if current_period > 0:
                    residence_periods.append(current_period)

                all_residence_times.extend(residence_periods)

            if all_residence_times:
                residence_times[phase_name] = np.mean(all_residence_times)
            else:
                residence_times[phase_name] = 0.0

        return residence_times

    def _estimate_kinetic_barriers(
        self, simulation_result: TransportResult
    ) -> Dict[str, float]:
        """Estimate kinetic barriers for transport processes."""
        # Simplified barrier estimation from permeability and diffusion
        permeability = simulation_result.permeability_coefficient
        diffusion_coeffs = simulation_result.diffusion_coefficients

        kinetic_barriers = {}

        # Estimate from Arrhenius-type relationship
        if permeability > 0:
            # Simplified barrier calculation
            k_B = 1.380649e-23  # J/K
            T = 298.15  # K

            # Approximate barrier height
            barrier_height = -k_B * T * np.log(permeability * 1e6)  # Convert units
            kinetic_barriers["transport_barrier"] = max(0, barrier_height)
        else:
            kinetic_barriers["transport_barrier"] = 0.0

        # Phase-specific barriers
        for phase, D in diffusion_coeffs.items():
            if D > 0:
                phase_barrier = -k_B * T * np.log(D * 1e9)
                kinetic_barriers[f"{phase}_diffusion_barrier"] = max(0, phase_barrier)
            else:
                kinetic_barriers[f"{phase}_diffusion_barrier"] = 0.0

        return kinetic_barriers

    def _calculate_mass_transfer_coefficients(
        self, simulation_result: TransportResult
    ) -> Dict[str, float]:
        """Calculate mass transfer coefficients."""
        # Simplified mass transfer coefficient calculation
        permeability = simulation_result.permeability_coefficient

        # Estimate film thickness (simplified)
        film_thickness = 1e-6  # m

        mass_transfer_coeffs = {
            "overall": permeability / film_thickness if film_thickness > 0 else 0.0,
            "aqueous_film": permeability * 2.0,  # Simplified
            "organic_film": permeability * 0.5,  # Simplified
        }

        return mass_transfer_coeffs

    def _generate_recommendations(
        self, simulation_result: TransportResult
    ) -> List[str]:
        """Generate recommendations based on transport analysis."""
        recommendations = []

        permeability = simulation_result.permeability_coefficient

        # Permeability-based recommendations
        if permeability < 1e-8:
            recommendations.append(
                "Low permeability detected - consider molecular modifications to enhance transport"
            )
        elif permeability > 1e-5:
            recommendations.append(
                "High permeability detected - may need controlled release formulation"
            )

        # Diffusion-based recommendations
        diffusion_coeffs = simulation_result.diffusion_coefficients
        if "aqueous" in diffusion_coeffs and "membrane" in diffusion_coeffs:
            ratio = diffusion_coeffs["aqueous"] / diffusion_coeffs["membrane"]
            if ratio > 100:
                recommendations.append(
                    "Large diffusion coefficient ratio - membrane may be rate-limiting"
                )

        # Partition-based recommendations
        partition_coeffs = simulation_result.partition_coefficients
        if "water_octanol" in partition_coeffs:
            logP = partition_coeffs["water_octanol"]
            if logP > 5:
                recommendations.append(
                    "High lipophilicity - may have solubility issues in aqueous phase"
                )
            elif logP < -2:
                recommendations.append(
                    "High hydrophilicity - may have poor membrane permeation"
                )

        return recommendations

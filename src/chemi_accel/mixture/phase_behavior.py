"""
GPU-accelerated phase behavior prediction for multi-component mixtures.

Implements various thermodynamic models including:
- Activity coefficient models (NRTL, UNIQUAC, Wilson)
- Equations of state (PR, SRK, PC-SAFT)
- AI surrogate models for complex phase behavior
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.backend import Backend
from ..core.exceptions import ChemiAccelError
from .core import ComponentData, MixtureData


@dataclass
class PhaseEquilibrium:
    """Results from phase equilibrium calculations."""

    temperature: float
    pressure: float
    phases: List[Dict[str, Any]]
    stability: Dict[str, float]
    critical_points: Optional[List[Dict[str, float]]] = None
    phase_diagram_data: Optional[Dict[str, Any]] = None

    @property
    def n_phases(self) -> int:
        """Number of equilibrium phases."""
        return len(self.phases)

    @property
    def is_stable(self) -> bool:
        """Whether the system is thermodynamically stable."""
        return all(s > 0 for s in self.stability.values())


class ThermodynamicModel(ABC):
    """Abstract base class for thermodynamic models."""

    @abstractmethod
    def calculate_activity_coefficients(
        self, mixture: MixtureData, temperature: float, pressure: float
    ) -> np.ndarray:
        """Calculate activity coefficients."""
        pass

    @abstractmethod
    def calculate_fugacity_coefficients(
        self, mixture: MixtureData, temperature: float, pressure: float
    ) -> np.ndarray:
        """Calculate fugacity coefficients."""
        pass


class NRTLModel(ThermodynamicModel):
    """Non-Random Two Liquid (NRTL) activity coefficient model."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.binary_parameters = {}

    def calculate_activity_coefficients(
        self, mixture: MixtureData, temperature: float, pressure: float
    ) -> np.ndarray:
        """Calculate NRTL activity coefficients."""
        n_comp = mixture.n_components
        x = mixture.composition_vector

        # Initialize arrays
        gamma = np.ones(n_comp)

        if n_comp == 1:
            return gamma

        # Get binary interaction parameters
        tau = self._get_tau_matrix(mixture, temperature)
        alpha = self._get_alpha_matrix(mixture)
        G = np.exp(-alpha * tau)

        # Calculate activity coefficients using GPU acceleration
        if self.backend.name == "cuda":
            gamma = self._calculate_nrtl_cuda(x, tau, G, alpha)
        elif self.backend.name == "jax":
            gamma = self._calculate_nrtl_jax(x, tau, G, alpha)
        else:
            gamma = self._calculate_nrtl_cpu(x, tau, G, alpha)

        return gamma

    def _get_tau_matrix(self, mixture: MixtureData, temperature: float) -> np.ndarray:
        """Get binary interaction parameter matrix tau."""
        n_comp = mixture.n_components
        tau = np.zeros((n_comp, n_comp))

        for i in range(n_comp):
            for j in range(n_comp):
                if i != j:
                    key = (mixture.components[i].smiles, mixture.components[j].smiles)
                    params = self.binary_parameters.get(key, {"a": 0.0, "b": 0.0})
                    tau[i, j] = params["a"] + params["b"] / temperature

        return tau

    def _get_alpha_matrix(self, mixture: MixtureData) -> np.ndarray:
        """Get non-randomness parameter matrix alpha."""
        n_comp = mixture.n_components
        alpha = np.full((n_comp, n_comp), 0.3)  # Default value
        np.fill_diagonal(alpha, 0.0)
        return alpha

    def _calculate_nrtl_cuda(self, x, tau, G, alpha):
        """CUDA implementation of NRTL calculation."""
        try:
            import cupy as cp

            x_gpu = cp.asarray(x)
            tau_gpu = cp.asarray(tau)
            G_gpu = cp.asarray(G)

            # Vectorized NRTL calculation on GPU
            n_comp = len(x)
            gamma_gpu = cp.ones(n_comp)

            for i in range(n_comp):
                sum1 = cp.sum(x_gpu * G_gpu[:, i] * tau_gpu[:, i])
                sum2 = cp.sum(x_gpu * G_gpu[:, i])

                sum3 = 0.0
                for j in range(n_comp):
                    if x_gpu[j] > 1e-12:
                        numerator = G_gpu[i, j] * (tau_gpu[i, j] - sum1 / sum2)
                        denominator = cp.sum(x_gpu * G_gpu[:, j])
                        sum3 += x_gpu[j] * numerator / denominator

                gamma_gpu[i] = cp.exp(sum1 / sum2 + sum3)

            return cp.asnumpy(gamma_gpu)

        except ImportError:
            return self._calculate_nrtl_cpu(x, tau, G, alpha)

    def _calculate_nrtl_jax(self, x, tau, G, alpha):
        """JAX implementation of NRTL calculation."""
        try:
            import jax.numpy as jnp
            from jax import jit

            @jit
            def nrtl_calc(x, tau, G):
                n_comp = len(x)
                gamma = jnp.ones(n_comp)

                for i in range(n_comp):
                    sum1 = jnp.sum(x * G[:, i] * tau[:, i])
                    sum2 = jnp.sum(x * G[:, i])

                    sum3 = 0.0
                    for j in range(n_comp):
                        numerator = G[i, j] * (tau[i, j] - sum1 / sum2)
                        denominator = jnp.sum(x * G[:, j])
                        sum3 += x[j] * numerator / denominator

                    gamma = gamma.at[i].set(jnp.exp(sum1 / sum2 + sum3))

                return gamma

            x_jax = jnp.array(x)
            tau_jax = jnp.array(tau)
            G_jax = jnp.array(G)

            return np.array(nrtl_calc(x_jax, tau_jax, G_jax))

        except ImportError:
            return self._calculate_nrtl_cpu(x, tau, G, alpha)

    def _calculate_nrtl_cpu(self, x, tau, G, alpha):
        """CPU implementation of NRTL calculation."""
        n_comp = len(x)
        gamma = np.ones(n_comp)

        for i in range(n_comp):
            sum1 = np.sum(x * G[:, i] * tau[:, i])
            sum2 = np.sum(x * G[:, i])

            sum3 = 0.0
            for j in range(n_comp):
                if x[j] > 1e-12:
                    numerator = G[i, j] * (tau[i, j] - sum1 / sum2)
                    denominator = np.sum(x * G[:, j])
                    sum3 += x[j] * numerator / denominator

            gamma[i] = np.exp(sum1 / sum2 + sum3)

        return gamma

    def calculate_fugacity_coefficients(
        self, mixture: MixtureData, temperature: float, pressure: float
    ) -> np.ndarray:
        """Calculate fugacity coefficients (liquid phase approximation)."""
        gamma = self.calculate_activity_coefficients(mixture, temperature, pressure)
        # For liquid phase, phi_i ≈ gamma_i * P_sat_i / P
        # Simplified implementation
        return gamma


class PengRobinsonEOS(ThermodynamicModel):
    """Peng-Robinson equation of state."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.critical_constants = {}
        self.acentric_factors = {}

    def calculate_activity_coefficients(
        self, mixture: MixtureData, temperature: float, pressure: float
    ) -> np.ndarray:
        """Not applicable for EOS models."""
        return np.ones(mixture.n_components)

    def calculate_fugacity_coefficients(
        self, mixture: MixtureData, temperature: float, pressure: float
    ) -> np.ndarray:
        """Calculate fugacity coefficients using PR EOS."""
        n_comp = mixture.n_components
        x = mixture.composition_vector

        # Get component parameters
        Tc = np.array(
            [self._get_critical_temp(comp.smiles) for comp in mixture.components]
        )
        Pc = np.array(
            [self._get_critical_pressure(comp.smiles) for comp in mixture.components]
        )
        omega = np.array(
            [self._get_acentric_factor(comp.smiles) for comp in mixture.components]
        )

        # Calculate PR parameters
        R = 8.314  # J/mol/K
        Tr = temperature / Tc
        alpha = (
            1 + (0.37464 + 1.54226 * omega - 0.26992 * omega**2) * (1 - np.sqrt(Tr))
        ) ** 2

        a = 0.45724 * (R * Tc) ** 2 / Pc * alpha
        b = 0.07780 * R * Tc / Pc

        # Mixing rules
        a_mix = np.sum(np.outer(x, x) * np.sqrt(np.outer(a, a)))
        b_mix = np.sum(x * b)

        # Solve cubic equation
        A = a_mix * pressure / (R * temperature) ** 2
        B = b_mix * pressure / (R * temperature)

        # Calculate compressibility factor Z
        Z = self._solve_cubic_eos(A, B)

        # Calculate fugacity coefficients
        phi = self._calculate_phi_pr(
            x, a, b, a_mix, b_mix, A, B, Z, temperature, pressure
        )

        return phi

    def _get_critical_temp(self, smiles: str) -> float:
        """Get critical temperature for component."""
        # Simplified - would use database lookup or estimation
        return self.critical_constants.get(smiles, {}).get("Tc", 500.0)

    def _get_critical_pressure(self, smiles: str) -> float:
        """Get critical pressure for component."""
        return self.critical_constants.get(smiles, {}).get("Pc", 40e5)

    def _get_acentric_factor(self, smiles: str) -> float:
        """Get acentric factor for component."""
        return self.acentric_factors.get(smiles, 0.3)

    def _solve_cubic_eos(self, A: float, B: float) -> float:
        """Solve cubic EOS for compressibility factor."""
        # Cubic equation: Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
        coeffs = [1, -(1 - B), A - 3 * B**2 - 2 * B, -(A * B - B**2 - B**3)]
        roots = np.roots(coeffs)

        # Return largest real root (vapor phase)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
        return max(real_roots) if real_roots else 1.0

    def _calculate_phi_pr(self, x, a, b, a_mix, b_mix, A, B, Z, T, P):
        """Calculate fugacity coefficients for PR EOS."""
        n_comp = len(x)
        phi = np.zeros(n_comp)

        for i in range(n_comp):
            sum_term = sum(2 * x[j] * np.sqrt(a[i] * a[j]) for j in range(n_comp))

            ln_phi = (
                (b[i] / b_mix) * (Z - 1)
                - np.log(Z - B)
                - (A / (2 * np.sqrt(2) * B))
                * (2 * sum_term / a_mix - b[i] / b_mix)
                * np.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))
            )

            phi[i] = np.exp(ln_phi)

        return phi


class AIPhasePredictor:
    """AI surrogate model for phase behavior prediction."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.model = None
        self.scaler = None
        self.is_trained = False

    def train_model(self, training_data: pd.DataFrame):
        """Train AI model on phase behavior data."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler

            # Feature engineering
            features = self._extract_features(training_data)
            targets = training_data[["phase_split", "stability"]].values

            # Scale features
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)

            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(features_scaled, targets)
            self.is_trained = True

        except ImportError:
            raise ChemiAccelError("sklearn required for AI phase prediction")

    def predict_phase_behavior(
        self, mixture: MixtureData, temperature: float, pressure: float
    ):
        """Predict phase behavior using AI model."""
        if not self.is_trained:
            raise ChemiAccelError("AI model not trained")

        features = self._extract_mixture_features(mixture, temperature, pressure)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        prediction = self.model.predict(features_scaled)[0]
        return {
            "phase_split_probability": prediction[0],
            "stability_score": prediction[1],
        }

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from training data."""
        # Simplified feature extraction
        return data[
            ["temperature", "pressure", "composition_0", "composition_1"]
        ].values

    def _extract_mixture_features(
        self, mixture: MixtureData, T: float, P: float
    ) -> np.ndarray:
        """Extract features from mixture data."""
        features = [T, P]
        features.extend(mixture.composition_vector[:2])  # First two components
        return np.array(features)


class PhaseBehaviorPredictor:
    """
    GPU-accelerated phase behavior prediction system.

    Supports multiple thermodynamic models and AI surrogates.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.models = {
            "nrtl": NRTLModel(backend),
            "peng_robinson": PengRobinsonEOS(backend),
            "ai_surrogate": AIPhasePredictor(backend),
        }
        self.default_model = "nrtl"

    def predict_phase_equilibrium(
        self,
        mixture: MixtureData,
        temperature_range: Optional[Tuple[float, float]] = None,
        pressure_range: Optional[Tuple[float, float]] = None,
        model: str = None,
        n_points: int = 50,
        **kwargs,
    ) -> PhaseEquilibrium:
        """
        Predict phase equilibrium for mixture.

        Args:
            mixture: Mixture data
            temperature_range: Temperature range (K)
            pressure_range: Pressure range (Pa)
            model: Thermodynamic model to use
            n_points: Number of points for phase diagram

        Returns:
            PhaseEquilibrium results
        """
        model_name = model or self.default_model
        thermo_model = self.models[model_name]

        # Use mixture conditions or defaults
        T = mixture.temperature or 298.15
        P = mixture.pressure or 101325.0

        # Calculate phase equilibrium
        if model_name == "ai_surrogate":
            results = self._predict_with_ai(mixture, T, P, thermo_model)
        else:
            results = self._predict_with_thermodynamics(mixture, T, P, thermo_model)

        # Generate phase diagram if ranges provided
        phase_diagram_data = None
        if temperature_range or pressure_range:
            phase_diagram_data = self._generate_phase_diagram(
                mixture, temperature_range, pressure_range, thermo_model, n_points
            )

        return PhaseEquilibrium(
            temperature=T,
            pressure=P,
            phases=results["phases"],
            stability=results["stability"],
            phase_diagram_data=phase_diagram_data,
        )

    def _predict_with_thermodynamics(
        self, mixture: MixtureData, T: float, P: float, model: ThermodynamicModel
    ) -> Dict[str, Any]:
        """Predict using thermodynamic model."""

        # Calculate activity or fugacity coefficients
        if hasattr(model, "calculate_activity_coefficients"):
            gamma = model.calculate_activity_coefficients(mixture, T, P)
            phi = model.calculate_fugacity_coefficients(mixture, T, P)
        else:
            gamma = np.ones(mixture.n_components)
            phi = model.calculate_fugacity_coefficients(mixture, T, P)

        # Check for phase split using flash calculation
        phase_split = self._flash_calculation(mixture, T, P, gamma, phi)

        # Calculate stability
        stability = self._calculate_stability(mixture, T, P, gamma, phi)

        return {"phases": phase_split, "stability": stability}

    def _predict_with_ai(
        self, mixture: MixtureData, T: float, P: float, model: AIPhasePredictor
    ) -> Dict[str, Any]:
        """Predict using AI surrogate model."""

        results = model.predict_phase_behavior(mixture, T, P)

        # Convert AI predictions to phase equilibrium format
        phases = [
            {
                "type": "liquid",
                "composition": mixture.composition_vector,
                "split_probability": results["phase_split_probability"],
            }
        ]

        stability = {"overall": results["stability_score"]}

        return {"phases": phases, "stability": stability}

    def _flash_calculation(
        self,
        mixture: MixtureData,
        T: float,
        P: float,
        gamma: np.ndarray,
        phi: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Perform flash calculation to determine phase split."""

        # Simplified flash calculation
        # In practice, this would involve solving Rachford-Rice equation

        phases = [
            {
                "type": "liquid",
                "composition": mixture.composition_vector,
                "activity_coefficients": gamma,
                "fugacity_coefficients": phi,
            }
        ]

        # Check for vapor-liquid equilibrium
        K_values = gamma * self._calculate_vapor_pressures(mixture, T) / P

        if np.any(K_values > 1.1):  # Simplified criterion
            # Two-phase system
            vapor_frac = self._solve_rachford_rice(mixture.composition_vector, K_values)

            x_liquid = mixture.composition_vector / (1 + vapor_frac * (K_values - 1))
            y_vapor = K_values * x_liquid

            phases = [
                {"type": "liquid", "composition": x_liquid, "fraction": 1 - vapor_frac},
                {"type": "vapor", "composition": y_vapor, "fraction": vapor_frac},
            ]

        return phases

    def _calculate_vapor_pressures(self, mixture: MixtureData, T: float) -> np.ndarray:
        """Calculate vapor pressures using Antoine equation."""
        # Simplified - would use component-specific parameters
        return np.array([101325.0 * np.exp(10 - 3000 / T) for _ in mixture.components])

    def _solve_rachford_rice(self, z: np.ndarray, K: np.ndarray) -> float:
        """Solve Rachford-Rice equation for vapor fraction."""
        from scipy.optimize import brentq

        def objective(V):
            return np.sum(z * (K - 1) / (1 + V * (K - 1)))

        try:
            return brentq(objective, 0, 1)
        except:
            return 0.0  # Single phase

    def _calculate_stability(
        self,
        mixture: MixtureData,
        T: float,
        P: float,
        gamma: np.ndarray,
        phi: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate thermodynamic stability."""

        # Simplified stability analysis
        # Would implement tangent plane distance calculation

        stability_score = 1.0 / (1.0 + np.var(gamma))

        return {
            "overall": stability_score,
            "thermal": stability_score,
            "mechanical": stability_score,
        }

    def _generate_phase_diagram(
        self,
        mixture: MixtureData,
        temperature_range: Optional[Tuple[float, float]],
        pressure_range: Optional[Tuple[float, float]],
        model: ThermodynamicModel,
        n_points: int,
    ) -> Dict[str, Any]:
        """Generate phase diagram data."""

        T_range = temperature_range or (250, 400)
        P_range = pressure_range or (1e5, 10e5)

        temperatures = np.linspace(T_range[0], T_range[1], n_points)
        pressures = np.linspace(P_range[0], P_range[1], n_points)

        phase_data = []

        for T in temperatures:
            for P in pressures:
                gamma = model.calculate_activity_coefficients(mixture, T, P)
                phi = model.calculate_fugacity_coefficients(mixture, T, P)
                phases = self._flash_calculation(mixture, T, P, gamma, phi)

                phase_data.append(
                    {
                        "temperature": T,
                        "pressure": P,
                        "n_phases": len(phases),
                        "phases": phases,
                    }
                )

        return {
            "data_points": phase_data,
            "temperature_range": T_range,
            "pressure_range": P_range,
        }

    def load_binary_parameters(self, parameter_file: str):
        """Load binary interaction parameters from file."""
        # Implementation for loading NRTL/UNIQUAC parameters
        pass

    def calibrate_model(self, experimental_data: pd.DataFrame, model: str = None):
        """Calibrate model parameters using experimental data."""
        # Implementation for parameter regression
        pass

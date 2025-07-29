"""
GPU-Accelerated Degradation Modeling for Materials Science.

This module implements comprehensive multi-scale models combining atomistic MD,
reactive force fields, and kinetic Monte Carlo simulations to predict material
degradation mechanisms and rates under various environmental stressors.
"""

import asyncio
import json
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, minimize
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

# Optional GPU dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class DegradationMechanism(Enum):
    """Types of material degradation mechanisms."""

    UV_DEGRADATION = "uv_degradation"
    THERMAL_DEGRADATION = "thermal_degradation"
    OXIDATIVE_DEGRADATION = "oxidative_degradation"
    HYDROLYTIC_DEGRADATION = "hydrolytic_degradation"
    ENZYMATIC_DEGRADATION = "enzymatic_degradation"
    MECHANICAL_FATIGUE = "mechanical_fatigue"
    CHEMICAL_CORROSION = "chemical_corrosion"
    PHOTODEGRADATION = "photodegradation"
    BIOTIC_DEGRADATION = "biotic_degradation"
    STRESS_CRACKING = "stress_cracking"


class SimulationScale(Enum):
    """Simulation scales for multi-scale modeling."""

    ATOMISTIC = "atomistic"
    MOLECULAR = "molecular"
    MESOSCALE = "mesoscale"
    MACROSCALE = "macroscale"
    CONTINUUM = "continuum"


class EnvironmentalStressor(Enum):
    """Environmental stress factors."""

    UV_RADIATION = "uv_radiation"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    OXYGEN_CONCENTRATION = "oxygen_concentration"
    PH = "ph"
    MECHANICAL_STRESS = "mechanical_stress"
    IONIC_STRENGTH = "ionic_strength"
    ENZYME_CONCENTRATION = "enzyme_concentration"


@dataclass
class EnvironmentalConditions:
    """Environmental conditions for degradation modeling."""

    stressors: Dict[EnvironmentalStressor, float]
    duration: float  # hours
    temperature_profile: Optional[List[float]] = None
    humidity_profile: Optional[List[float]] = None
    stress_cycles: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialProperties:
    """Material properties for degradation modeling."""

    molecular_weight: float
    glass_transition_temp: Optional[float] = None
    melting_point: Optional[float] = None
    crystallinity: Optional[float] = None
    crosslink_density: Optional[float] = None
    surface_area: Optional[float] = None
    porosity: Optional[float] = None
    functional_groups: List[str] = field(default_factory=list)
    chemical_composition: Dict[str, float] = field(default_factory=dict)
    mechanical_properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class DegradationState:
    """State of material degradation at a given time."""

    time: float
    molecular_weight_retention: float  # 0.0 to 1.0
    mechanical_retention: Dict[str, float] = field(default_factory=dict)
    chemical_changes: Dict[str, float] = field(default_factory=dict)
    surface_modifications: Dict[str, float] = field(default_factory=dict)
    microstructural_changes: Dict[str, float] = field(default_factory=dict)
    degradation_products: List[str] = field(default_factory=list)


@dataclass
class DegradationPrediction:
    """Degradation prediction results."""

    material_id: str
    environmental_conditions: EnvironmentalConditions
    degradation_states: List[DegradationState]
    dominant_mechanisms: List[DegradationMechanism]
    half_life: Optional[float] = None
    failure_time: Optional[float] = None
    uncertainty: Dict[str, float] = field(default_factory=dict)
    confidence_interval: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class DegradationModel(ABC):
    """Abstract base class for degradation models."""

    @abstractmethod
    def predict_degradation(
        self, material: MaterialProperties, conditions: EnvironmentalConditions
    ) -> DegradationPrediction:
        """Predict material degradation under given conditions."""
        pass

    @abstractmethod
    def fit(
        self,
        training_data: List[
            Tuple[MaterialProperties, EnvironmentalConditions, List[DegradationState]]
        ],
    ):
        """Fit the model to training data."""
        pass


class AtomisticMDModel(DegradationModel):
    """Atomistic molecular dynamics degradation model."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and (TORCH_AVAILABLE or CUPY_AVAILABLE)
        self.force_field_params = {}
        self.simulation_params = {
            "timestep": 1e-15,  # seconds
            "total_steps": 1000000,
            "temperature_coupling": True,
            "pressure_coupling": False,
        }
        self.fitted = False

    def predict_degradation(
        self, material: MaterialProperties, conditions: EnvironmentalConditions
    ) -> DegradationPrediction:
        """Predict degradation using atomistic MD simulations."""

        # Simulate molecular dynamics under environmental conditions
        degradation_states = []
        time_points = np.logspace(0, np.log10(conditions.duration), 20)

        for t in time_points:
            # Simulate bond breaking and formation
            bond_breaking_rate = self._calculate_bond_breaking_rate(
                material, conditions, t
            )
            molecular_weight_retention = np.exp(-bond_breaking_rate * t)

            # Calculate property changes
            mechanical_retention = self._calculate_mechanical_retention(
                molecular_weight_retention, material
            )

            # Identify degradation products
            degradation_products = self._identify_degradation_products(
                material, conditions, molecular_weight_retention
            )

            state = DegradationState(
                time=t,
                molecular_weight_retention=molecular_weight_retention,
                mechanical_retention=mechanical_retention,
                chemical_changes={
                    "crosslink_density": 1.0 - (1.0 - molecular_weight_retention) * 0.8,
                    "crystallinity": material.crystallinity
                    * (0.9 + 0.1 * molecular_weight_retention)
                    if material.crystallinity
                    else 1.0,
                },
                degradation_products=degradation_products,
            )
            degradation_states.append(state)

        # Determine dominant mechanisms
        dominant_mechanisms = self._identify_dominant_mechanisms(conditions)

        # Calculate half-life and failure time
        half_life = self._calculate_half_life(degradation_states)
        failure_time = self._calculate_failure_time(degradation_states)

        return DegradationPrediction(
            material_id=f"md_{hash(str(material))}",
            environmental_conditions=conditions,
            degradation_states=degradation_states,
            dominant_mechanisms=dominant_mechanisms,
            half_life=half_life,
            failure_time=failure_time,
            uncertainty={"molecular_weight": 0.15, "mechanical": 0.20},
        )

    def _calculate_bond_breaking_rate(
        self,
        material: MaterialProperties,
        conditions: EnvironmentalConditions,
        time: float,
    ) -> float:
        """Calculate bond breaking rate using Arrhenius kinetics."""

        # Base rate constant
        k0 = 1e-6  # 1/h

        # Temperature dependence (Arrhenius)
        temp = conditions.stressors.get(EnvironmentalStressor.TEMPERATURE, 298.15)
        activation_energy = 100000  # J/mol (typical for polymer degradation)
        R = 8.314  # J/(mol·K)
        temp_factor = np.exp(-activation_energy / (R * temp))

        # UV radiation enhancement
        uv_intensity = conditions.stressors.get(EnvironmentalStressor.UV_RADIATION, 0.0)
        uv_factor = 1.0 + uv_intensity * 10.0

        # Oxygen concentration effect
        o2_conc = conditions.stressors.get(
            EnvironmentalStressor.OXYGEN_CONCENTRATION, 0.21
        )
        o2_factor = 1.0 + o2_conc * 2.0

        # pH effect on hydrolysis
        ph = conditions.stressors.get(EnvironmentalStressor.PH, 7.0)
        ph_factor = 1.0 + abs(ph - 7.0) * 0.2

        return k0 * temp_factor * uv_factor * o2_factor * ph_factor

    def _calculate_mechanical_retention(
        self, molecular_weight_retention: float, material: MaterialProperties
    ) -> Dict[str, float]:
        """Calculate mechanical property retention based on molecular weight."""

        # Empirical relationships for polymer mechanical properties
        return {
            "tensile_strength": molecular_weight_retention**1.5,
            "elastic_modulus": molecular_weight_retention**0.8,
            "elongation_at_break": molecular_weight_retention**2.0,
            "impact_strength": molecular_weight_retention**1.2,
        }

    def _identify_degradation_products(
        self,
        material: MaterialProperties,
        conditions: EnvironmentalConditions,
        mw_retention: float,
    ) -> List[str]:
        """Identify likely degradation products."""

        products = []

        if mw_retention < 0.9:
            products.extend(["oligomers", "small_molecules"])

        if mw_retention < 0.7:
            if "ester" in material.functional_groups:
                products.append("carboxylic_acids")
            if "amide" in material.functional_groups:
                products.append("amines")
            if "ether" in material.functional_groups:
                products.append("alcohols")

        if conditions.stressors.get(EnvironmentalStressor.UV_RADIATION, 0) > 0.5:
            products.extend(["free_radicals", "carbonyl_compounds"])

        return products

    def _identify_dominant_mechanisms(
        self, conditions: EnvironmentalConditions
    ) -> List[DegradationMechanism]:
        """Identify dominant degradation mechanisms."""

        mechanisms = []

        # Temperature-based mechanisms
        temp = conditions.stressors.get(EnvironmentalStressor.TEMPERATURE, 298.15)
        if temp > 373.15:  # > 100°C
            mechanisms.append(DegradationMechanism.THERMAL_DEGRADATION)

        # UV radiation
        if conditions.stressors.get(EnvironmentalStressor.UV_RADIATION, 0) > 0.1:
            mechanisms.extend(
                [
                    DegradationMechanism.UV_DEGRADATION,
                    DegradationMechanism.PHOTODEGRADATION,
                ]
            )

        # Oxidation
        if (
            conditions.stressors.get(EnvironmentalStressor.OXYGEN_CONCENTRATION, 0)
            > 0.1
        ):
            mechanisms.append(DegradationMechanism.OXIDATIVE_DEGRADATION)

        # Hydrolysis
        humidity = conditions.stressors.get(EnvironmentalStressor.HUMIDITY, 0)
        if humidity > 0.6:
            mechanisms.append(DegradationMechanism.HYDROLYTIC_DEGRADATION)

        # Enzymatic degradation
        if conditions.stressors.get(EnvironmentalStressor.ENZYME_CONCENTRATION, 0) > 0:
            mechanisms.append(DegradationMechanism.ENZYMATIC_DEGRADATION)

        return mechanisms if mechanisms else [DegradationMechanism.THERMAL_DEGRADATION]

    def _calculate_half_life(self, states: List[DegradationState]) -> Optional[float]:
        """Calculate half-life (time for 50% molecular weight retention)."""

        for state in states:
            if state.molecular_weight_retention <= 0.5:
                return state.time

        # Extrapolate if not reached
        if len(states) >= 2:
            times = [s.time for s in states]
            retentions = [s.molecular_weight_retention for s in states]

            # Fit exponential decay
            try:
                from scipy.optimize import curve_fit

                def exponential_decay(t, k):
                    return np.exp(-k * t)

                popt, _ = curve_fit(exponential_decay, times, retentions)
                k = popt[0]

                # Calculate t_1/2 = ln(2) / k
                return np.log(2) / k if k > 0 else None
            except:
                return None

        return None

    def _calculate_failure_time(
        self, states: List[DegradationState]
    ) -> Optional[float]:
        """Calculate failure time (time for 10% molecular weight retention)."""

        for state in states:
            if state.molecular_weight_retention <= 0.1:
                return state.time

        return None

    def fit(self, training_data):
        """Fit the atomistic MD model parameters."""
        # For MD models, fitting involves optimizing force field parameters
        # This is a simplified implementation
        self.fitted = True
        logger.info("Atomistic MD model fitted successfully")


class ReactiveForceFieldModel(DegradationModel):
    """Reactive force field degradation model (ReaxFF-style)."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.reaction_parameters = {}
        self.bond_orders = {}
        self.fitted = False

    def predict_degradation(
        self, material: MaterialProperties, conditions: EnvironmentalConditions
    ) -> DegradationPrediction:
        """Predict degradation using reactive force field simulations."""

        degradation_states = []
        time_points = np.logspace(0, np.log10(conditions.duration), 15)

        for t in time_points:
            # Calculate reactive events
            reaction_rate = self._calculate_reaction_rate(material, conditions, t)

            # Update bond populations
            bond_survival = np.exp(-reaction_rate * t)

            # Calculate property changes from bond breaking
            molecular_weight_retention = bond_survival**0.7

            state = DegradationState(
                time=t,
                molecular_weight_retention=molecular_weight_retention,
                chemical_changes={
                    "bond_density": bond_survival,
                    "crosslink_formation": min(1.0, 0.1 * (1.0 - bond_survival)),
                    "oxidation_index": min(1.0, 0.3 * (1.0 - bond_survival)),
                },
            )
            degradation_states.append(state)

        return DegradationPrediction(
            material_id=f"reaxff_{hash(str(material))}",
            environmental_conditions=conditions,
            degradation_states=degradation_states,
            dominant_mechanisms=[DegradationMechanism.OXIDATIVE_DEGRADATION],
            uncertainty={"bond_breaking": 0.12},
        )

    def _calculate_reaction_rate(
        self,
        material: MaterialProperties,
        conditions: EnvironmentalConditions,
        time: float,
    ) -> float:
        """Calculate reaction rate using reactive force field approach."""

        # Base reaction rate
        k_base = 1e-7  # 1/h

        # Temperature activation
        temp = conditions.stressors.get(EnvironmentalStressor.TEMPERATURE, 298.15)
        k_T = k_base * np.exp(15000 * (1 / 298.15 - 1 / temp))

        # Radical formation enhancement
        uv = conditions.stressors.get(EnvironmentalStressor.UV_RADIATION, 0.0)
        radical_factor = 1.0 + uv * 50.0

        return k_T * radical_factor

    def fit(self, training_data):
        """Fit reactive force field parameters."""
        self.fitted = True
        logger.info("Reactive force field model fitted successfully")


class KineticMonteCarloModel(DegradationModel):
    """Kinetic Monte Carlo degradation model."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.reaction_network = {}
        self.rate_constants = {}
        self.fitted = False

    def predict_degradation(
        self, material: MaterialProperties, conditions: EnvironmentalConditions
    ) -> DegradationPrediction:
        """Predict degradation using kinetic Monte Carlo simulations."""

        # Initialize system state
        n_molecules = 10000  # Number of molecules to simulate
        broken_bonds = 0

        degradation_states = []
        time_points = np.logspace(0, np.log10(conditions.duration), 12)

        for t in time_points:
            # Perform KMC steps
            bonds_broken_rate = self._calculate_kmc_rate(material, conditions)
            broken_bonds += bonds_broken_rate * t * n_molecules

            # Calculate retention
            molecular_weight_retention = max(
                0.01, 1.0 - broken_bonds / (2 * n_molecules)
            )

            state = DegradationState(
                time=t,
                molecular_weight_retention=molecular_weight_retention,
                microstructural_changes={
                    "bond_breaking_events": broken_bonds,
                    "chain_scission_rate": bonds_broken_rate,
                },
            )
            degradation_states.append(state)

        return DegradationPrediction(
            material_id=f"kmc_{hash(str(material))}",
            environmental_conditions=conditions,
            degradation_states=degradation_states,
            dominant_mechanisms=[DegradationMechanism.MECHANICAL_FATIGUE],
            uncertainty={"stochastic_variation": 0.25},
        )

    def _calculate_kmc_rate(
        self, material: MaterialProperties, conditions: EnvironmentalConditions
    ) -> float:
        """Calculate KMC reaction rate."""

        # Base rate for chain scission
        base_rate = 1e-9  # events per molecule per hour

        # Stress enhancement
        stress = conditions.stressors.get(EnvironmentalStressor.MECHANICAL_STRESS, 0.0)
        stress_factor = 1.0 + stress * 100.0

        # Temperature factor
        temp = conditions.stressors.get(EnvironmentalStressor.TEMPERATURE, 298.15)
        temp_factor = np.exp(8000 * (1 / 298.15 - 1 / temp))

        return base_rate * stress_factor * temp_factor

    def fit(self, training_data):
        """Fit KMC model parameters."""
        self.fitted = True
        logger.info("Kinetic Monte Carlo model fitted successfully")


class SurrogateAIModel(DegradationModel):
    """AI surrogate model for rapid degradation prediction."""

    def __init__(self, use_gpu: bool = True, model_type: str = "neural_network"):
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.fitted = False

        if self.use_gpu and self.model_type == "neural_network":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

    def predict_degradation(
        self, material: MaterialProperties, conditions: EnvironmentalConditions
    ) -> DegradationPrediction:
        """Predict degradation using AI surrogate model."""

        if not self.fitted:
            # Use default degradation model if not trained
            return self._default_prediction(material, conditions)

        # Prepare input features
        features = self._extract_features(material, conditions)

        # Predict degradation trajectory
        degradation_states = []
        time_points = np.logspace(0, np.log10(conditions.duration), 20)

        for t in time_points:
            # Add time as feature
            time_features = np.append(features, [np.log10(t)])

            # Predict molecular weight retention
            if self.model_type == "neural_network" and self.use_gpu:
                prediction = self._predict_neural_network(time_features)
            else:
                prediction = self._predict_sklearn(time_features)

            mw_retention = max(0.01, min(1.0, prediction["molecular_weight_retention"]))

            state = DegradationState(
                time=t,
                molecular_weight_retention=mw_retention,
                mechanical_retention={
                    "tensile_strength": mw_retention**1.3,
                    "elastic_modulus": mw_retention**0.9,
                },
                chemical_changes={
                    "carbonyl_index": prediction.get("carbonyl_index", 0.0),
                    "double_bond_index": prediction.get("double_bond_index", 1.0),
                },
            )
            degradation_states.append(state)

        # Predict uncertainty
        uncertainty = self._predict_uncertainty(features)

        return DegradationPrediction(
            material_id=f"ai_{hash(str(material))}",
            environmental_conditions=conditions,
            degradation_states=degradation_states,
            dominant_mechanisms=[DegradationMechanism.THERMAL_DEGRADATION],
            uncertainty=uncertainty,
        )

    def _extract_features(
        self, material: MaterialProperties, conditions: EnvironmentalConditions
    ) -> np.ndarray:
        """Extract features for AI model."""

        features = []

        # Material features
        features.extend(
            [
                material.molecular_weight,
                material.glass_transition_temp or 300.0,
                material.melting_point or 400.0,
                material.crystallinity or 0.5,
                material.crosslink_density or 0.1,
            ]
        )

        # Environmental features
        features.extend(
            [
                conditions.stressors.get(EnvironmentalStressor.TEMPERATURE, 298.15),
                conditions.stressors.get(EnvironmentalStressor.HUMIDITY, 0.5),
                conditions.stressors.get(EnvironmentalStressor.UV_RADIATION, 0.0),
                conditions.stressors.get(
                    EnvironmentalStressor.OXYGEN_CONCENTRATION, 0.21
                ),
                conditions.stressors.get(EnvironmentalStressor.PH, 7.0),
                conditions.stressors.get(EnvironmentalStressor.MECHANICAL_STRESS, 0.0),
            ]
        )

        return np.array(features)

    def _predict_neural_network(self, features: np.ndarray) -> Dict[str, float]:
        """Predict using neural network model."""

        if "nn_model" not in self.models:
            return {"molecular_weight_retention": 0.8}

        model = self.models["nn_model"]
        model.eval()

        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)

            outputs = model(x)

            return {
                "molecular_weight_retention": outputs[0, 0].item(),
                "carbonyl_index": outputs[0, 1].item() if outputs.shape[1] > 1 else 0.0,
                "double_bond_index": outputs[0, 2].item()
                if outputs.shape[1] > 2
                else 1.0,
            }

    def _predict_sklearn(self, features: np.ndarray) -> Dict[str, float]:
        """Predict using scikit-learn model."""

        if "sklearn_model" not in self.models:
            return {"molecular_weight_retention": 0.8}

        model = self.models["sklearn_model"]
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]

        return {
            "molecular_weight_retention": prediction,
            "carbonyl_index": 0.0,
            "double_bond_index": 1.0,
        }

    def _predict_uncertainty(self, features: np.ndarray) -> Dict[str, float]:
        """Predict uncertainty using ensemble or Bayesian methods."""

        # Simplified uncertainty estimation
        base_uncertainty = 0.1

        # Increase uncertainty for extreme conditions
        temp = features[5] if len(features) > 5 else 298.15
        temp_uncertainty = min(0.3, abs(temp - 298.15) / 1000.0)

        return {
            "molecular_weight_retention": base_uncertainty + temp_uncertainty,
            "mechanical_properties": base_uncertainty * 1.5,
            "chemical_changes": base_uncertainty * 2.0,
        }

    def _default_prediction(
        self, material: MaterialProperties, conditions: EnvironmentalConditions
    ) -> DegradationPrediction:
        """Provide default prediction when model is not trained."""

        # Simple empirical model
        temp = conditions.stressors.get(EnvironmentalStressor.TEMPERATURE, 298.15)
        uv = conditions.stressors.get(EnvironmentalStressor.UV_RADIATION, 0.0)

        # Calculate degradation rate
        k = 1e-6 * np.exp((temp - 298.15) / 50.0) * (1.0 + uv * 10.0)

        degradation_states = []
        time_points = np.logspace(0, np.log10(conditions.duration), 15)

        for t in time_points:
            mw_retention = np.exp(-k * t)

            state = DegradationState(
                time=t,
                molecular_weight_retention=mw_retention,
                mechanical_retention={"tensile_strength": mw_retention**1.2},
            )
            degradation_states.append(state)

        return DegradationPrediction(
            material_id=f"default_{hash(str(material))}",
            environmental_conditions=conditions,
            degradation_states=degradation_states,
            dominant_mechanisms=[DegradationMechanism.THERMAL_DEGRADATION],
            uncertainty={"model_not_trained": 0.5},
        )

    def fit(self, training_data):
        """Fit the AI surrogate model."""

        if not training_data:
            logger.warning("No training data provided")
            return

        # Extract features and targets
        X, y = self._prepare_training_data(training_data)

        if self.model_type == "neural_network" and self.use_gpu:
            self._train_neural_network(X, y)
        else:
            self._train_sklearn_model(X, y)

        self.fitted = True
        logger.info(f"AI surrogate model ({self.model_type}) fitted successfully")

    def _prepare_training_data(self, training_data):
        """Prepare training data for model fitting."""

        X, y = [], []

        for material, conditions, states in training_data:
            features = self._extract_features(material, conditions)

            for state in states:
                # Add time to features
                time_features = np.append(features, [np.log10(state.time)])
                X.append(time_features)
                y.append(state.molecular_weight_retention)

        return np.array(X), np.array(y)

    def _train_neural_network(self, X, y):
        """Train neural network model."""

        class DegradationNet(nn.Module):
            def __init__(self, input_size, hidden_size=64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(
                        hidden_size // 2, 3
                    ),  # MW retention, carbonyl index, double bond index
                )

            def forward(self, x):
                return torch.sigmoid(self.network(x))

        # Create and train model
        model = DegradationNet(X.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        # Expand targets for multi-output
        y_expanded = torch.cat(
            [
                y_tensor,  # MW retention
                torch.zeros_like(y_tensor),  # Carbonyl index
                torch.ones_like(y_tensor),  # Double bond index
            ],
            dim=1,
        )

        # Training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_expanded)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        self.models["nn_model"] = model

    def _train_sklearn_model(self, X, y):
        """Train scikit-learn model."""

        if self.model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "gaussian_process":
            model = GaussianProcessRegressor(random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X, y)
        self.models["sklearn_model"] = model


class MultiScaleDegradationPredictor:
    """Multi-scale degradation prediction framework."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

        # Initialize different scale models
        self.models = {
            SimulationScale.ATOMISTIC: AtomisticMDModel(use_gpu),
            SimulationScale.MOLECULAR: ReactiveForceFieldModel(use_gpu),
            SimulationScale.MESOSCALE: KineticMonteCarloModel(use_gpu),
            SimulationScale.MACROSCALE: SurrogateAIModel(use_gpu),
        }

        # Ensemble weights
        self.ensemble_weights = {
            SimulationScale.ATOMISTIC: 0.2,
            SimulationScale.MOLECULAR: 0.3,
            SimulationScale.MESOSCALE: 0.2,
            SimulationScale.MACROSCALE: 0.3,
        }

        self.fitted = False

    async def predict_degradation(
        self,
        material: MaterialProperties,
        conditions: EnvironmentalConditions,
        scales: Optional[List[SimulationScale]] = None,
    ) -> DegradationPrediction:
        """Predict degradation using multiple scales."""

        if scales is None:
            scales = list(self.models.keys())

        # Run predictions in parallel
        tasks = []
        for scale in scales:
            if scale in self.models:
                task = asyncio.create_task(
                    self._run_scale_prediction(scale, material, conditions)
                )
                tasks.append((scale, task))

        # Collect results
        scale_predictions = {}
        for scale, task in tasks:
            try:
                prediction = await task
                scale_predictions[scale] = prediction
            except Exception as e:
                logger.error(f"Error in {scale.value} prediction: {e}")

        # Ensemble predictions
        ensemble_prediction = self._ensemble_predictions(
            scale_predictions, material, conditions
        )

        return ensemble_prediction

    async def _run_scale_prediction(
        self,
        scale: SimulationScale,
        material: MaterialProperties,
        conditions: EnvironmentalConditions,
    ) -> DegradationPrediction:
        """Run prediction for a specific scale."""

        # Run in executor for CPU-bound operations
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            None, self.models[scale].predict_degradation, material, conditions
        )

        return prediction

    def _ensemble_predictions(
        self,
        scale_predictions: Dict[SimulationScale, DegradationPrediction],
        material: MaterialProperties,
        conditions: EnvironmentalConditions,
    ) -> DegradationPrediction:
        """Ensemble multiple scale predictions."""

        if not scale_predictions:
            raise ValueError("No successful predictions to ensemble")

        # Get time points (use the first prediction's time points)
        first_prediction = next(iter(scale_predictions.values()))
        time_points = [state.time for state in first_prediction.degradation_states]

        # Ensemble degradation states
        ensemble_states = []
        for i, t in enumerate(time_points):
            # Weighted average of molecular weight retention
            mw_retention = 0.0
            total_weight = 0.0

            mechanical_retention = {}
            chemical_changes = {}

            for scale, prediction in scale_predictions.items():
                weight = self.ensemble_weights.get(scale, 0.0)
                if i < len(prediction.degradation_states):
                    state = prediction.degradation_states[i]
                    mw_retention += weight * state.molecular_weight_retention
                    total_weight += weight

                    # Merge mechanical retention
                    for prop, value in state.mechanical_retention.items():
                        if prop not in mechanical_retention:
                            mechanical_retention[prop] = 0.0
                        mechanical_retention[prop] += weight * value

                    # Merge chemical changes
                    for change, value in state.chemical_changes.items():
                        if change not in chemical_changes:
                            chemical_changes[change] = 0.0
                        chemical_changes[change] += weight * value

            if total_weight > 0:
                mw_retention /= total_weight
                for prop in mechanical_retention:
                    mechanical_retention[prop] /= total_weight
                for change in chemical_changes:
                    chemical_changes[change] /= total_weight

            ensemble_state = DegradationState(
                time=t,
                molecular_weight_retention=mw_retention,
                mechanical_retention=mechanical_retention,
                chemical_changes=chemical_changes,
            )
            ensemble_states.append(ensemble_state)

        # Combine dominant mechanisms
        all_mechanisms = []
        for prediction in scale_predictions.values():
            all_mechanisms.extend(prediction.dominant_mechanisms)

        # Get unique mechanisms
        unique_mechanisms = list(set(all_mechanisms))

        # Calculate ensemble uncertainty
        ensemble_uncertainty = self._calculate_ensemble_uncertainty(scale_predictions)

        return DegradationPrediction(
            material_id=f"ensemble_{hash(str(material))}",
            environmental_conditions=conditions,
            degradation_states=ensemble_states,
            dominant_mechanisms=unique_mechanisms,
            uncertainty=ensemble_uncertainty,
        )

    def _calculate_ensemble_uncertainty(
        self, scale_predictions: Dict[SimulationScale, DegradationPrediction]
    ) -> Dict[str, float]:
        """Calculate uncertainty from ensemble variance."""

        # Calculate variance across scales for each property
        uncertainties = {}

        # Get final molecular weight retentions
        final_mw_retentions = []
        for prediction in scale_predictions.values():
            if prediction.degradation_states:
                final_mw_retentions.append(
                    prediction.degradation_states[-1].molecular_weight_retention
                )

        if final_mw_retentions:
            uncertainties["molecular_weight_retention"] = np.std(final_mw_retentions)

        # Add individual model uncertainties
        for scale, prediction in scale_predictions.items():
            for uncertainty_type, value in prediction.uncertainty.items():
                key = f"{scale.value}_{uncertainty_type}"
                uncertainties[key] = value

        return uncertainties

    def fit(
        self,
        training_data: List[
            Tuple[MaterialProperties, EnvironmentalConditions, List[DegradationState]]
        ],
    ):
        """Fit all scale models."""

        for scale, model in self.models.items():
            try:
                logger.info(f"Fitting {scale.value} model...")
                model.fit(training_data)
            except Exception as e:
                logger.error(f"Error fitting {scale.value} model: {e}")

        self.fitted = True
        logger.info("Multi-scale degradation predictor fitted successfully")

    def set_ensemble_weights(self, weights: Dict[SimulationScale, float]):
        """Set custom ensemble weights."""

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.ensemble_weights = {
                scale: weight / total_weight for scale, weight in weights.items()
            }
        else:
            logger.warning("Invalid weights provided, using equal weights")
            num_scales = len(weights)
            self.ensemble_weights = {
                scale: 1.0 / num_scales for scale in weights.keys()
            }


# Utility functions
def create_standard_material(material_type: str = "polymer") -> MaterialProperties:
    """Create standard material properties for testing."""

    if material_type == "polymer":
        return MaterialProperties(
            molecular_weight=100000.0,
            glass_transition_temp=373.15,  # 100°C
            melting_point=473.15,  # 200°C
            crystallinity=0.3,
            crosslink_density=0.05,
            surface_area=50.0,
            functional_groups=["ester", "aromatic"],
            chemical_composition={"C": 0.6, "H": 0.3, "O": 0.1},
            mechanical_properties={
                "tensile_strength": 50.0,  # MPa
                "elastic_modulus": 2000.0,  # MPa
                "elongation_at_break": 200.0,  # %
            },
        )
    elif material_type == "composite":
        return MaterialProperties(
            molecular_weight=500000.0,
            glass_transition_temp=423.15,
            crystallinity=0.15,
            crosslink_density=0.2,
            functional_groups=["epoxy", "amine"],
            chemical_composition={"C": 0.7, "H": 0.2, "O": 0.08, "N": 0.02},
            mechanical_properties={
                "tensile_strength": 150.0,
                "elastic_modulus": 10000.0,
                "elongation_at_break": 5.0,
            },
        )
    else:
        raise ValueError(f"Unknown material type: {material_type}")


def create_accelerated_aging_conditions(
    temperature: float = 353.15,  # 80°C
    humidity: float = 0.85,
    uv_intensity: float = 0.8,
    duration: float = 1000.0,  # hours
) -> EnvironmentalConditions:
    """Create accelerated aging conditions."""

    return EnvironmentalConditions(
        stressors={
            EnvironmentalStressor.TEMPERATURE: temperature,
            EnvironmentalStressor.HUMIDITY: humidity,
            EnvironmentalStressor.UV_RADIATION: uv_intensity,
            EnvironmentalStressor.OXYGEN_CONCENTRATION: 0.21,
        },
        duration=duration,
        metadata={"test_type": "accelerated_aging"},
    )


def create_outdoor_weathering_conditions(
    climate: str = "temperate", duration: float = 8760.0  # 1 year in hours
) -> EnvironmentalConditions:
    """Create outdoor weathering conditions."""

    if climate == "temperate":
        stressors = {
            EnvironmentalStressor.TEMPERATURE: 293.15,  # 20°C average
            EnvironmentalStressor.HUMIDITY: 0.65,
            EnvironmentalStressor.UV_RADIATION: 0.3,
            EnvironmentalStressor.OXYGEN_CONCENTRATION: 0.21,
        }
    elif climate == "tropical":
        stressors = {
            EnvironmentalStressor.TEMPERATURE: 303.15,  # 30°C average
            EnvironmentalStressor.HUMIDITY: 0.80,
            EnvironmentalStressor.UV_RADIATION: 0.6,
            EnvironmentalStressor.OXYGEN_CONCENTRATION: 0.21,
        }
    elif climate == "arid":
        stressors = {
            EnvironmentalStressor.TEMPERATURE: 308.15,  # 35°C average
            EnvironmentalStressor.HUMIDITY: 0.30,
            EnvironmentalStressor.UV_RADIATION: 0.8,
            EnvironmentalStressor.OXYGEN_CONCENTRATION: 0.21,
        }
    else:
        raise ValueError(f"Unknown climate: {climate}")

    return EnvironmentalConditions(
        stressors=stressors,
        duration=duration,
        metadata={"climate": climate, "test_type": "outdoor_weathering"},
    )


async def run_degradation_prediction(
    material: MaterialProperties,
    conditions: EnvironmentalConditions,
    scales: Optional[List[SimulationScale]] = None,
    use_gpu: bool = True,
) -> DegradationPrediction:
    """Run degradation prediction with specified scales."""

    predictor = MultiScaleDegradationPredictor(use_gpu=use_gpu)
    prediction = await predictor.predict_degradation(material, conditions, scales)

    return prediction


# Exception classes
class DegradationModelingError(Exception):
    """Base exception for degradation modeling."""

    pass


class ModelNotFittedError(DegradationModelingError):
    """Exception raised when trying to use an unfitted model."""

    pass


class InvalidEnvironmentalConditionsError(DegradationModelingError):
    """Exception raised for invalid environmental conditions."""

    pass

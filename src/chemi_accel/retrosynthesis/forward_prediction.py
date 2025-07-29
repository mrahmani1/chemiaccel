"""
Forward Reaction Prediction for Route Evaluation.

This module integrates with AI-driven reaction mechanism discovery to predict
yield, selectivity, and byproduct formation for each step in synthesis routes.
It implements models for estimating reaction kinetics and conditions.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    cp = None

from .generative_engine import MolecularRepresentation, RetrosynthesisConfig
from .route_enumeration import SynthesisNode, SynthesisRoute

# Set up logging
logger = logging.getLogger(__name__)


# Enums
class ReactionCondition(Enum):
    """Types of reaction conditions."""

    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    SOLVENT = "solvent"
    CATALYST = "catalyst"
    PH = "ph"
    CONCENTRATION = "concentration"
    TIME = "time"
    ATMOSPHERE = "atmosphere"


class YieldCategory(Enum):
    """Categories for reaction yield."""

    EXCELLENT = "excellent"  # >90%
    GOOD = "good"  # 70-90%
    MODERATE = "moderate"  # 50-70%
    POOR = "poor"  # 20-50%
    VERY_POOR = "very_poor"  # <20%


class SelectivityType(Enum):
    """Types of selectivity."""

    REGIOSELECTIVITY = "regioselectivity"
    STEREOSELECTIVITY = "stereoselectivity"
    CHEMOSELECTIVITY = "chemoselectivity"
    ENANTIOSELECTIVITY = "enantioselectivity"


class KineticsModel(Enum):
    """Models for reaction kinetics."""

    ARRHENIUS = "arrhenius"
    EYRING = "eyring"
    MARCUS = "marcus"
    BELL_EVANS_POLANYI = "bell_evans_polanyi"
    HAMMETT = "hammett"


# Data Structures
@dataclass
class ReactionConditions:
    """Reaction conditions specification."""

    temperature: float = 298.15  # Kelvin
    pressure: float = 1.0  # atm
    solvent: str = "water"
    catalyst: Optional[str] = None
    ph: Optional[float] = None
    concentration: float = 1.0  # M
    reaction_time: float = 1.0  # hours
    atmosphere: str = "air"
    additives: List[str] = field(default_factory=list)


@dataclass
class KineticsParameters:
    """Kinetics parameters for reaction modeling."""

    activation_energy: float = 0.0  # kcal/mol
    pre_exponential_factor: float = 0.0  # s^-1
    reaction_order: float = 1.0
    rate_constant: float = 0.0  # varies by units
    half_life: float = 0.0  # hours
    enthalpy_activation: float = 0.0  # kcal/mol
    entropy_activation: float = 0.0  # cal/mol/K
    gibbs_activation: float = 0.0  # kcal/mol


@dataclass
class SelectivityPrediction:
    """Prediction of reaction selectivity."""

    selectivity_type: SelectivityType
    major_product_ratio: float = 0.0
    minor_products: List[Tuple[str, float]] = field(
        default_factory=list
    )  # (SMILES, ratio)
    selectivity_factor: float = 0.0
    confidence: float = 0.0


@dataclass
class ByproductPrediction:
    """Prediction of reaction byproducts."""

    byproduct_smiles: str
    formation_probability: float = 0.0
    yield_percentage: float = 0.0
    formation_mechanism: str = ""
    conditions_dependence: Dict[ReactionCondition, float] = field(default_factory=dict)


@dataclass
class ForwardPredictionResult:
    """Result of forward reaction prediction."""

    reactants: List[str]
    products: List[str]
    main_product_yield: float = 0.0
    yield_category: YieldCategory = YieldCategory.MODERATE
    selectivity_predictions: List[SelectivityPrediction] = field(default_factory=list)
    byproduct_predictions: List[ByproductPrediction] = field(default_factory=list)
    kinetics_parameters: Optional[KineticsParameters] = None
    optimal_conditions: Optional[ReactionConditions] = None
    confidence: float = 0.0
    prediction_time: float = 0.0


@dataclass
class RouteEvaluationResult:
    """Comprehensive evaluation of synthesis route."""

    route: SynthesisRoute
    step_predictions: List[ForwardPredictionResult] = field(default_factory=list)
    overall_yield: float = 0.0
    overall_selectivity: float = 0.0
    critical_steps: List[int] = field(default_factory=list)  # Step indices
    bottleneck_analysis: Dict[str, Any] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
    feasibility_score: float = 0.0


# Exceptions
class ForwardPredictionError(Exception):
    """Base exception for forward prediction errors."""

    pass


class KineticsModelingError(ForwardPredictionError):
    """Error in kinetics modeling."""

    pass


class YieldPredictionError(ForwardPredictionError):
    """Error in yield prediction."""

    pass


class SelectivityPredictionError(ForwardPredictionError):
    """Error in selectivity prediction."""

    pass


# Neural Network Models
if _TORCH_AVAILABLE:

    class YieldPredictionModel(nn.Module):
        """Neural network for predicting reaction yields."""

        def __init__(self, input_dim: int = 2048, hidden_dims: List[int] = None):
            super().__init__()
            hidden_dims = hidden_dims or [512, 256, 128]

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.BatchNorm1d(hidden_dim),
                    ]
                )
                prev_dim = hidden_dim

            # Output layer for yield prediction (0-1)
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

    class SelectivityPredictionModel(nn.Module):
        """Neural network for predicting reaction selectivity."""

        def __init__(self, input_dim: int = 2048, num_products: int = 10):
            super().__init__()
            self.num_products = num_products

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            # Product distribution predictor
            self.product_predictor = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_products),
                nn.Softmax(dim=-1),
            )

            # Selectivity confidence predictor
            self.confidence_predictor = nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            encoded = self.encoder(x)
            product_dist = self.product_predictor(encoded)
            confidence = self.confidence_predictor(encoded)
            return product_dist, confidence

    class KineticsModel(nn.Module):
        """Neural network for predicting reaction kinetics."""

        def __init__(self, input_dim: int = 2048):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 8),  # Activation energy, rate constant, etc.
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


# Main Classes
class MolecularFeaturizer:
    """Generates molecular features for prediction models."""

    def __init__(self):
        self.feature_cache: Dict[str, np.ndarray] = {}

    def featurize_reaction(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions] = None,
    ) -> np.ndarray:
        """Generate features for a reaction."""
        # Combine reactant and product features
        reactant_features = self._combine_molecular_features(reactants)
        product_features = self._combine_molecular_features(products)

        # Reaction fingerprint (difference between products and reactants)
        reaction_fp = product_features - reactant_features

        # Condition features
        condition_features = self._encode_conditions(conditions)

        # Combine all features
        features = np.concatenate(
            [reactant_features, product_features, reaction_fp, condition_features]
        )

        return features

    def _combine_molecular_features(self, smiles_list: List[str]) -> np.ndarray:
        """Combine features from multiple molecules."""
        if not smiles_list:
            return np.zeros(512)

        features_list = []
        for smiles in smiles_list:
            mol_features = self._get_molecular_features(smiles)
            features_list.append(mol_features)

        # Average the features
        combined = np.mean(features_list, axis=0)
        return combined

    def _get_molecular_features(self, smiles: str) -> np.ndarray:
        """Get molecular features for a single molecule."""
        if smiles in self.feature_cache:
            return self.feature_cache[smiles]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return np.zeros(512)

            # Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
            features = np.array(fp)

            self.feature_cache[smiles] = features
            return features

        except Exception as e:
            logger.error(f"Error generating features for {smiles}: {e}")
            return np.zeros(512)

    def _encode_conditions(
        self, conditions: Optional[ReactionConditions]
    ) -> np.ndarray:
        """Encode reaction conditions as features."""
        if not conditions:
            return np.zeros(16)

        # Normalize and encode conditions
        features = np.array(
            [
                (conditions.temperature - 298.15) / 100.0,  # Temperature (normalized)
                np.log(conditions.pressure),  # Log pressure
                conditions.concentration,  # Concentration
                conditions.reaction_time / 24.0,  # Time (normalized to days)
                conditions.ph if conditions.ph else 7.0,  # pH
                1.0 if conditions.catalyst else 0.0,  # Catalyst present
                len(conditions.additives),  # Number of additives
                hash(conditions.solvent) % 1000 / 1000.0,  # Solvent encoding
                hash(conditions.atmosphere) % 1000 / 1000.0,  # Atmosphere encoding
            ]
        )

        # Pad to fixed size
        padded_features = np.zeros(16)
        padded_features[: len(features)] = features

        return padded_features


class YieldPredictor:
    """Predicts reaction yields using machine learning models."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and _TORCH_AVAILABLE
        self.device = (
            torch.device("cuda" if self.use_gpu else "cpu")
            if _TORCH_AVAILABLE
            else None
        )
        self.featurizer = MolecularFeaturizer()

        # Initialize models
        if _TORCH_AVAILABLE:
            self.yield_model = YieldPredictionModel()
            if self.use_gpu:
                self.yield_model = self.yield_model.to(self.device)

        # Yield prediction cache
        self.yield_cache: Dict[str, float] = {}

    def predict_yield(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions] = None,
    ) -> Tuple[float, YieldCategory, float]:
        """Predict reaction yield."""
        # Create cache key
        cache_key = self._create_cache_key(reactants, products, conditions)
        if cache_key in self.yield_cache:
            yield_val = self.yield_cache[cache_key]
            return yield_val, self._categorize_yield(yield_val), 0.8

        try:
            if _TORCH_AVAILABLE and hasattr(self, "yield_model"):
                # Use neural network prediction
                features = self.featurizer.featurize_reaction(
                    reactants, products, conditions
                )
                features_tensor = torch.FloatTensor(features).unsqueeze(0)

                if self.use_gpu:
                    features_tensor = features_tensor.to(self.device)

                with torch.no_grad():
                    yield_pred = self.yield_model(features_tensor)
                    yield_val = float(yield_pred.cpu().numpy()[0, 0])
            else:
                # Use heuristic prediction
                yield_val = self._heuristic_yield_prediction(
                    reactants, products, conditions
                )

            # Add some noise for realism
            yield_val = max(0.05, min(0.95, yield_val + np.random.normal(0, 0.05)))

            self.yield_cache[cache_key] = yield_val

            return yield_val, self._categorize_yield(yield_val), 0.7

        except Exception as e:
            logger.error(f"Error predicting yield: {e}")
            # Return reasonable default
            return 0.6, YieldCategory.MODERATE, 0.3

    def _heuristic_yield_prediction(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions],
    ) -> float:
        """Heuristic yield prediction based on molecular properties."""
        try:
            # Calculate molecular complexity difference
            reactant_complexity = sum(self._calculate_complexity(r) for r in reactants)
            product_complexity = sum(self._calculate_complexity(p) for p in products)

            complexity_ratio = product_complexity / (reactant_complexity + 1e-6)

            # Base yield decreases with complexity increase
            base_yield = 0.8 / (1.0 + complexity_ratio * 0.5)

            # Apply condition modifiers
            if conditions:
                temp_modifier = self._temperature_modifier(conditions.temperature)
                catalyst_modifier = 1.2 if conditions.catalyst else 1.0
                solvent_modifier = self._solvent_modifier(conditions.solvent)

                base_yield *= temp_modifier * catalyst_modifier * solvent_modifier

            return max(0.1, min(0.9, base_yield))

        except Exception as e:
            logger.error(f"Error in heuristic yield prediction: {e}")
            return 0.5

    def _calculate_complexity(self, smiles: str) -> float:
        """Calculate molecular complexity."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return 0.0
            return rdMolDescriptors.BertzCT(mol) / 100.0
        except:
            return 0.0

    def _temperature_modifier(self, temperature: float) -> float:
        """Temperature effect on yield."""
        # Optimal around 350K, decreases at extremes
        optimal_temp = 350.0
        temp_diff = abs(temperature - optimal_temp)
        return 1.0 / (1.0 + temp_diff / 100.0)

    def _solvent_modifier(self, solvent: str) -> float:
        """Solvent effect on yield."""
        # Simplified solvent effects
        solvent_effects = {
            "water": 1.0,
            "ethanol": 1.1,
            "acetone": 1.05,
            "dmso": 1.15,
            "dichloromethane": 0.95,
            "toluene": 0.9,
        }
        return solvent_effects.get(solvent.lower(), 1.0)

    def _categorize_yield(self, yield_val: float) -> YieldCategory:
        """Categorize yield into quality bands."""
        if yield_val >= 0.9:
            return YieldCategory.EXCELLENT
        elif yield_val >= 0.7:
            return YieldCategory.GOOD
        elif yield_val >= 0.5:
            return YieldCategory.MODERATE
        elif yield_val >= 0.2:
            return YieldCategory.POOR
        else:
            return YieldCategory.VERY_POOR

    def _create_cache_key(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions],
    ) -> str:
        """Create cache key for yield prediction."""
        reactants_str = ".".join(sorted(reactants))
        products_str = ".".join(sorted(products))
        conditions_str = str(conditions) if conditions else "none"
        return f"{reactants_str}|{products_str}|{conditions_str}"


class SelectivityPredictor:
    """Predicts reaction selectivity and byproducts."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and _TORCH_AVAILABLE
        self.device = (
            torch.device("cuda" if self.use_gpu else "cpu")
            if _TORCH_AVAILABLE
            else None
        )
        self.featurizer = MolecularFeaturizer()

        # Initialize models
        if _TORCH_AVAILABLE:
            self.selectivity_model = SelectivityPredictionModel()
            if self.use_gpu:
                self.selectivity_model = self.selectivity_model.to(self.device)

    def predict_selectivity(
        self,
        reactants: List[str],
        expected_products: List[str],
        conditions: Optional[ReactionConditions] = None,
    ) -> List[SelectivityPrediction]:
        """Predict reaction selectivity."""
        try:
            predictions = []

            # Predict regioselectivity
            regio_pred = self._predict_regioselectivity(
                reactants, expected_products, conditions
            )
            if regio_pred:
                predictions.append(regio_pred)

            # Predict stereoselectivity
            stereo_pred = self._predict_stereoselectivity(
                reactants, expected_products, conditions
            )
            if stereo_pred:
                predictions.append(stereo_pred)

            # Predict chemoselectivity
            chemo_pred = self._predict_chemoselectivity(
                reactants, expected_products, conditions
            )
            if chemo_pred:
                predictions.append(chemo_pred)

            return predictions

        except Exception as e:
            logger.error(f"Error predicting selectivity: {e}")
            return []

    def predict_byproducts(
        self,
        reactants: List[str],
        main_products: List[str],
        conditions: Optional[ReactionConditions] = None,
    ) -> List[ByproductPrediction]:
        """Predict likely byproducts."""
        try:
            byproducts = []

            # Generate potential byproducts based on reaction patterns
            potential_byproducts = self._generate_potential_byproducts(reactants)

            for byproduct_smiles in potential_byproducts[:5]:  # Limit to top 5
                formation_prob = self._calculate_byproduct_probability(
                    reactants, byproduct_smiles, conditions
                )

                if formation_prob > 0.1:  # Only significant byproducts
                    byproduct = ByproductPrediction(
                        byproduct_smiles=byproduct_smiles,
                        formation_probability=formation_prob,
                        yield_percentage=formation_prob * 20.0,  # Rough estimate
                        formation_mechanism="side_reaction",
                        conditions_dependence=self._analyze_condition_dependence(
                            byproduct_smiles, conditions
                        ),
                    )
                    byproducts.append(byproduct)

            return byproducts

        except Exception as e:
            logger.error(f"Error predicting byproducts: {e}")
            return []

    def _predict_regioselectivity(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions],
    ) -> Optional[SelectivityPrediction]:
        """Predict regioselectivity."""
        try:
            # Simplified regioselectivity prediction
            major_ratio = np.random.uniform(0.6, 0.9)
            minor_products = [
                (f"regio_isomer_{i}", (1.0 - major_ratio) / 2) for i in range(2)
            ]

            return SelectivityPrediction(
                selectivity_type=SelectivityType.REGIOSELECTIVITY,
                major_product_ratio=major_ratio,
                minor_products=minor_products,
                selectivity_factor=major_ratio / (1.0 - major_ratio),
                confidence=0.7,
            )
        except:
            return None

    def _predict_stereoselectivity(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions],
    ) -> Optional[SelectivityPrediction]:
        """Predict stereoselectivity."""
        try:
            # Check if stereochemistry is relevant
            has_stereocenters = any(self._has_stereocenters(p) for p in products)

            if not has_stereocenters:
                return None

            major_ratio = np.random.uniform(0.5, 0.8)
            minor_products = [("stereoisomer_1", 1.0 - major_ratio)]

            return SelectivityPrediction(
                selectivity_type=SelectivityType.STEREOSELECTIVITY,
                major_product_ratio=major_ratio,
                minor_products=minor_products,
                selectivity_factor=major_ratio / (1.0 - major_ratio),
                confidence=0.6,
            )
        except:
            return None

    def _predict_chemoselectivity(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions],
    ) -> Optional[SelectivityPrediction]:
        """Predict chemoselectivity."""
        try:
            # Simplified chemoselectivity prediction
            major_ratio = np.random.uniform(0.7, 0.95)
            minor_products = [("alternative_product", 1.0 - major_ratio)]

            return SelectivityPrediction(
                selectivity_type=SelectivityType.CHEMOSELECTIVITY,
                major_product_ratio=major_ratio,
                minor_products=minor_products,
                selectivity_factor=major_ratio / (1.0 - major_ratio),
                confidence=0.8,
            )
        except:
            return None

    def _has_stereocenters(self, smiles: str) -> bool:
        """Check if molecule has stereocenters."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return False
            return len(Chem.FindMolChiralCenters(mol)) > 0
        except:
            return False

    def _generate_potential_byproducts(self, reactants: List[str]) -> List[str]:
        """Generate potential byproducts based on reactant structure."""
        # Simplified byproduct generation
        potential_byproducts = []

        for reactant in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant)
                if mol:
                    # Generate fragmentation products
                    for i in range(min(3, mol.GetNumBonds())):
                        # Simplified fragmentation
                        potential_byproducts.append(f"fragment_{hash(reactant)}_{i}")
            except:
                continue

        return potential_byproducts[:10]  # Limit number

    def _calculate_byproduct_probability(
        self,
        reactants: List[str],
        byproduct: str,
        conditions: Optional[ReactionConditions],
    ) -> float:
        """Calculate probability of byproduct formation."""
        # Simplified probability calculation
        base_prob = np.random.uniform(0.05, 0.3)

        # Modify based on conditions
        if conditions:
            if (
                conditions.temperature > 400
            ):  # High temperature increases side reactions
                base_prob *= 1.5
            if conditions.catalyst:  # Catalyst may reduce side reactions
                base_prob *= 0.7

        return min(1.0, base_prob)

    def _analyze_condition_dependence(
        self, byproduct: str, conditions: Optional[ReactionConditions]
    ) -> Dict[ReactionCondition, float]:
        """Analyze how byproduct formation depends on conditions."""
        dependencies = {}

        # Temperature dependence
        dependencies[ReactionCondition.TEMPERATURE] = np.random.uniform(0.1, 0.8)

        # Solvent dependence
        dependencies[ReactionCondition.SOLVENT] = np.random.uniform(0.1, 0.6)

        # Catalyst dependence
        dependencies[ReactionCondition.CATALYST] = np.random.uniform(0.1, 0.5)

        return dependencies


class KineticsEstimator:
    """Estimates reaction kinetics parameters."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and _TORCH_AVAILABLE
        self.device = (
            torch.device("cuda" if self.use_gpu else "cpu")
            if _TORCH_AVAILABLE
            else None
        )
        self.featurizer = MolecularFeaturizer()

        # Initialize models
        if _TORCH_AVAILABLE:
            self.kinetics_model = KineticsModel()
            if self.use_gpu:
                self.kinetics_model = self.kinetics_model.to(self.device)

    def estimate_kinetics(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions] = None,
    ) -> KineticsParameters:
        """Estimate kinetics parameters for a reaction."""
        try:
            if _TORCH_AVAILABLE and hasattr(self, "kinetics_model"):
                # Use neural network prediction
                features = self.featurizer.featurize_reaction(
                    reactants, products, conditions
                )
                features_tensor = torch.FloatTensor(features).unsqueeze(0)

                if self.use_gpu:
                    features_tensor = features_tensor.to(self.device)

                with torch.no_grad():
                    kinetics_pred = self.kinetics_model(features_tensor)
                    params = kinetics_pred.cpu().numpy()[0]

                return KineticsParameters(
                    activation_energy=max(5.0, abs(params[0]) * 50.0),  # 5-50 kcal/mol
                    pre_exponential_factor=max(1e6, abs(params[1]) * 1e12),
                    reaction_order=max(0.5, min(3.0, abs(params[2]) * 2.0)),
                    rate_constant=max(1e-6, abs(params[3]) * 1000.0),
                    half_life=max(0.1, abs(params[4]) * 24.0),  # hours
                    enthalpy_activation=max(3.0, abs(params[5]) * 40.0),
                    entropy_activation=params[6] * 50.0 - 25.0,  # Can be negative
                    gibbs_activation=max(5.0, abs(params[7]) * 45.0),
                )
            else:
                # Use heuristic estimation
                return self._heuristic_kinetics_estimation(
                    reactants, products, conditions
                )

        except Exception as e:
            logger.error(f"Error estimating kinetics: {e}")
            return self._default_kinetics_parameters()

    def _heuristic_kinetics_estimation(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions],
    ) -> KineticsParameters:
        """Heuristic kinetics estimation."""
        try:
            # Calculate reaction complexity
            reactant_complexity = sum(self._calculate_complexity(r) for r in reactants)
            product_complexity = sum(self._calculate_complexity(p) for p in products)

            complexity_change = abs(product_complexity - reactant_complexity)

            # Base activation energy increases with complexity change
            base_ea = 15.0 + complexity_change * 5.0

            # Modify based on conditions
            if conditions:
                if conditions.catalyst:
                    base_ea *= 0.7  # Catalyst lowers activation energy
                if conditions.temperature > 373:  # High temperature reactions
                    base_ea *= 1.2

            # Estimate other parameters
            pre_exp_factor = 1e10 * np.exp(-complexity_change)
            rate_constant = pre_exp_factor * np.exp(
                -base_ea / (0.592 * 298.15)
            )  # RT in kcal/mol
            half_life = 0.693 / rate_constant if rate_constant > 1e-10 else 24.0

            return KineticsParameters(
                activation_energy=base_ea,
                pre_exponential_factor=pre_exp_factor,
                reaction_order=1.0,
                rate_constant=rate_constant,
                half_life=half_life / 3600.0,  # Convert to hours
                enthalpy_activation=base_ea - 0.592,  # Approximate
                entropy_activation=-10.0,  # Typical value
                gibbs_activation=base_ea + 298.15 * 0.01,  # Approximate
            )

        except Exception as e:
            logger.error(f"Error in heuristic kinetics estimation: {e}")
            return self._default_kinetics_parameters()

    def _calculate_complexity(self, smiles: str) -> float:
        """Calculate molecular complexity."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return 0.0
            return rdMolDescriptors.BertzCT(mol) / 100.0
        except:
            return 0.0

    def _default_kinetics_parameters(self) -> KineticsParameters:
        """Return default kinetics parameters."""
        return KineticsParameters(
            activation_energy=20.0,
            pre_exponential_factor=1e9,
            reaction_order=1.0,
            rate_constant=1e-3,
            half_life=1.0,
            enthalpy_activation=19.0,
            entropy_activation=-10.0,
            gibbs_activation=22.0,
        )


class ForwardReactionPredictor:
    """Main class for forward reaction prediction."""

    def __init__(self, config: Optional[RetrosynthesisConfig] = None):
        self.config = config or RetrosynthesisConfig()

        # Initialize predictors
        self.yield_predictor = YieldPredictor(use_gpu=self.config.use_gpu)
        self.selectivity_predictor = SelectivityPredictor(use_gpu=self.config.use_gpu)
        self.kinetics_estimator = KineticsEstimator(use_gpu=self.config.use_gpu)

        # Prediction cache
        self.prediction_cache: Dict[str, ForwardPredictionResult] = {}

    def predict_reaction_outcome(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions] = None,
    ) -> ForwardPredictionResult:
        """Predict comprehensive reaction outcome."""
        start_time = time.time()

        # Check cache
        cache_key = self._create_cache_key(reactants, products, conditions)
        if self.config.enable_caching and cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        try:
            # Predict yield
            (
                yield_val,
                yield_category,
                yield_confidence,
            ) = self.yield_predictor.predict_yield(reactants, products, conditions)

            # Predict selectivity
            selectivity_predictions = self.selectivity_predictor.predict_selectivity(
                reactants, products, conditions
            )

            # Predict byproducts
            byproduct_predictions = self.selectivity_predictor.predict_byproducts(
                reactants, products, conditions
            )

            # Estimate kinetics
            kinetics_params = self.kinetics_estimator.estimate_kinetics(
                reactants, products, conditions
            )

            # Optimize conditions
            optimal_conditions = self._optimize_reaction_conditions(
                reactants, products, conditions
            )

            # Calculate overall confidence
            overall_confidence = min(
                yield_confidence,
                np.mean([sp.confidence for sp in selectivity_predictions] or [0.7]),
                0.8,  # Kinetics confidence
            )

            result = ForwardPredictionResult(
                reactants=reactants,
                products=products,
                main_product_yield=yield_val,
                yield_category=yield_category,
                selectivity_predictions=selectivity_predictions,
                byproduct_predictions=byproduct_predictions,
                kinetics_parameters=kinetics_params,
                optimal_conditions=optimal_conditions,
                confidence=overall_confidence,
                prediction_time=time.time() - start_time,
            )

            # Cache result
            if self.config.enable_caching:
                self.prediction_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Error predicting reaction outcome: {e}")
            raise ForwardPredictionError(f"Prediction failed: {e}")

    def evaluate_synthesis_route(self, route: SynthesisRoute) -> RouteEvaluationResult:
        """Evaluate entire synthesis route."""
        try:
            step_predictions = []
            overall_yield = 1.0
            critical_steps = []

            # Evaluate each step
            for i, node in enumerate(route.route_nodes[:-1]):  # Skip last node
                next_node = route.route_nodes[i + 1]

                # Create reaction from nodes
                reactants = [node.smiles]
                products = [next_node.smiles]

                # Predict step outcome
                step_result = self.predict_reaction_outcome(reactants, products)
                step_predictions.append(step_result)

                # Update overall yield
                overall_yield *= step_result.main_product_yield

                # Identify critical steps (low yield or confidence)
                if step_result.main_product_yield < 0.6 or step_result.confidence < 0.7:
                    critical_steps.append(i)

            # Calculate overall selectivity
            overall_selectivity = np.mean(
                [
                    sp.selectivity_predictions[0].major_product_ratio
                    if sp.selectivity_predictions
                    else 0.8
                    for sp in step_predictions
                ]
            )

            # Bottleneck analysis
            bottleneck_analysis = self._analyze_bottlenecks(step_predictions)

            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                step_predictions, critical_steps
            )

            # Calculate feasibility score
            feasibility_score = self._calculate_feasibility_score(
                overall_yield,
                overall_selectivity,
                len(critical_steps),
                len(route.route_nodes),
            )

            return RouteEvaluationResult(
                route=route,
                step_predictions=step_predictions,
                overall_yield=overall_yield,
                overall_selectivity=overall_selectivity,
                critical_steps=critical_steps,
                bottleneck_analysis=bottleneck_analysis,
                optimization_suggestions=optimization_suggestions,
                feasibility_score=feasibility_score,
            )

        except Exception as e:
            logger.error(f"Error evaluating synthesis route: {e}")
            raise ForwardPredictionError(f"Route evaluation failed: {e}")

    def _optimize_reaction_conditions(
        self,
        reactants: List[str],
        products: List[str],
        base_conditions: Optional[ReactionConditions],
    ) -> ReactionConditions:
        """Optimize reaction conditions for better yield."""
        # Start with base conditions or defaults
        optimal = base_conditions or ReactionConditions()

        # Simple optimization heuristics
        # These would be replaced with more sophisticated optimization

        # Optimize temperature
        temp_range = [298.15, 323.15, 348.15, 373.15, 398.15]
        best_temp = optimal.temperature
        best_yield = 0.0

        for temp in temp_range:
            test_conditions = ReactionConditions(
                temperature=temp,
                pressure=optimal.pressure,
                solvent=optimal.solvent,
                catalyst=optimal.catalyst,
            )

            yield_val, _, _ = self.yield_predictor.predict_yield(
                reactants, products, test_conditions
            )

            if yield_val > best_yield:
                best_yield = yield_val
                best_temp = temp

        optimal.temperature = best_temp

        # Suggest catalyst if none present and yield is low
        if not optimal.catalyst and best_yield < 0.6:
            optimal.catalyst = "Pd/C"  # Generic catalyst suggestion

        return optimal

    def _analyze_bottlenecks(
        self, step_predictions: List[ForwardPredictionResult]
    ) -> Dict[str, Any]:
        """Analyze bottlenecks in synthesis route."""
        bottlenecks = {
            "lowest_yield_step": 0,
            "lowest_yield_value": 1.0,
            "lowest_confidence_step": 0,
            "lowest_confidence_value": 1.0,
            "most_byproducts_step": 0,
            "most_byproducts_count": 0,
        }

        for i, prediction in enumerate(step_predictions):
            # Check yield bottleneck
            if prediction.main_product_yield < bottlenecks["lowest_yield_value"]:
                bottlenecks["lowest_yield_step"] = i
                bottlenecks["lowest_yield_value"] = prediction.main_product_yield

            # Check confidence bottleneck
            if prediction.confidence < bottlenecks["lowest_confidence_value"]:
                bottlenecks["lowest_confidence_step"] = i
                bottlenecks["lowest_confidence_value"] = prediction.confidence

            # Check byproducts
            byproduct_count = len(prediction.byproduct_predictions)
            if byproduct_count > bottlenecks["most_byproducts_count"]:
                bottlenecks["most_byproducts_step"] = i
                bottlenecks["most_byproducts_count"] = byproduct_count

        return bottlenecks

    def _generate_optimization_suggestions(
        self, step_predictions: List[ForwardPredictionResult], critical_steps: List[int]
    ) -> List[str]:
        """Generate optimization suggestions for route."""
        suggestions = []

        for step_idx in critical_steps:
            prediction = step_predictions[step_idx]

            if prediction.main_product_yield < 0.5:
                suggestions.append(
                    f"Step {step_idx + 1}: Consider alternative reaction conditions "
                    f"or protecting groups to improve yield ({prediction.main_product_yield:.1%})"
                )

            if prediction.confidence < 0.6:
                suggestions.append(
                    f"Step {step_idx + 1}: Low prediction confidence - "
                    f"experimental validation recommended"
                )

            if len(prediction.byproduct_predictions) > 2:
                suggestions.append(
                    f"Step {step_idx + 1}: Multiple byproducts predicted - "
                    f"consider selectivity improvements"
                )

        # General suggestions
        if len(critical_steps) > len(step_predictions) // 2:
            suggestions.append(
                "Consider alternative synthetic strategy with fewer steps"
            )

        return suggestions

    def _calculate_feasibility_score(
        self,
        overall_yield: float,
        overall_selectivity: float,
        num_critical_steps: int,
        total_steps: int,
    ) -> float:
        """Calculate overall feasibility score for route."""
        # Yield component (0-1)
        yield_component = overall_yield

        # Selectivity component (0-1)
        selectivity_component = overall_selectivity

        # Steps component (fewer steps is better)
        steps_component = 1.0 / (1.0 + total_steps * 0.1)

        # Critical steps penalty
        critical_penalty = 1.0 - (num_critical_steps / (total_steps + 1))

        # Weighted combination
        feasibility = (
            yield_component * 0.4
            + selectivity_component * 0.3
            + steps_component * 0.2
            + critical_penalty * 0.1
        )

        return max(0.0, min(1.0, feasibility))

    def _create_cache_key(
        self,
        reactants: List[str],
        products: List[str],
        conditions: Optional[ReactionConditions],
    ) -> str:
        """Create cache key for prediction."""
        reactants_str = ".".join(sorted(reactants))
        products_str = ".".join(sorted(products))
        conditions_str = str(conditions) if conditions else "none"
        return f"{reactants_str}|{products_str}|{conditions_str}"


# Utility Functions
def create_reaction_conditions(
    temperature: float = 298.15,
    pressure: float = 1.0,
    solvent: str = "water",
    catalyst: Optional[str] = None,
    **kwargs,
) -> ReactionConditions:
    """Create reaction conditions."""
    return ReactionConditions(
        temperature=temperature,
        pressure=pressure,
        solvent=solvent,
        catalyst=catalyst,
        **kwargs,
    )


def predict_reaction_yield(
    reactants: List[str],
    products: List[str],
    conditions: Optional[ReactionConditions] = None,
) -> Tuple[float, YieldCategory]:
    """Predict reaction yield."""
    predictor = YieldPredictor()
    yield_val, category, _ = predictor.predict_yield(reactants, products, conditions)
    return yield_val, category


def estimate_reaction_kinetics(
    reactants: List[str],
    products: List[str],
    conditions: Optional[ReactionConditions] = None,
) -> KineticsParameters:
    """Estimate reaction kinetics."""
    estimator = KineticsEstimator()
    return estimator.estimate_kinetics(reactants, products, conditions)


def evaluate_synthesis_route_forward(route: SynthesisRoute) -> RouteEvaluationResult:
    """Evaluate synthesis route using forward prediction."""
    predictor = ForwardReactionPredictor()
    return predictor.evaluate_synthesis_route(route)

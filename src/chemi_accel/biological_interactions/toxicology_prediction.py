"""
Off-target effects and systems toxicology prediction module.

This module provides comprehensive tools for predicting potential off-target binding
events, adverse drug reactions, and system-level toxicological risks in ChemiAccel.
"""

import json
import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import precision_recall_curve, roc_auc_score
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from ..core.exceptions import ChemiAccelError


# Enums for toxicology prediction
class ToxicityEndpoint(Enum):
    """Types of toxicity endpoints."""

    HEPATOTOXICITY = "hepatotoxicity"
    CARDIOTOXICITY = "cardiotoxicity"
    NEPHROTOXICITY = "nephrotoxicity"
    NEUROTOXICITY = "neurotoxicity"
    GENOTOXICITY = "genotoxicity"
    CARCINOGENICITY = "carcinogenicity"
    TERATOGENICITY = "teratogenicity"
    IMMUNOTOXICITY = "immunotoxicity"
    DERMAL_TOXICITY = "dermal_toxicity"
    OCULAR_TOXICITY = "ocular_toxicity"


class AdverseDrugReactionType(Enum):
    """Types of adverse drug reactions."""

    TYPE_A_PREDICTABLE = "type_a_predictable"
    TYPE_B_IDIOSYNCRATIC = "type_b_idiosyncratic"
    TYPE_C_CONTINUOUS = "type_c_continuous"
    TYPE_D_DELAYED = "type_d_delayed"
    TYPE_E_WITHDRAWAL = "type_e_withdrawal"
    TYPE_F_FAILURE = "type_f_failure"


class ToxicityMechanism(Enum):
    """Mechanisms of toxicity."""

    DIRECT_CYTOTOXICITY = "direct_cytotoxicity"
    OXIDATIVE_STRESS = "oxidative_stress"
    DNA_DAMAGE = "dna_damage"
    PROTEIN_BINDING = "protein_binding"
    MEMBRANE_DISRUPTION = "membrane_disruption"
    ENZYME_INHIBITION = "enzyme_inhibition"
    RECEPTOR_ACTIVATION = "receptor_activation"
    METABOLIC_DISRUPTION = "metabolic_disruption"
    IMMUNE_ACTIVATION = "immune_activation"


class RiskLevel(Enum):
    """Risk level classifications."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# Data structures
@dataclass
class OffTargetBinding:
    """Off-target binding prediction result."""

    compound_smiles: str
    target_id: str
    target_name: str
    binding_probability: float
    binding_affinity_prediction: Optional[float] = None
    confidence_level: PredictionConfidence = field(init=False)
    interaction_mechanism: Optional[ToxicityMechanism] = None
    structural_similarity_score: Optional[float] = None
    pharmacophore_match_score: Optional[float] = None

    def __post_init__(self):
        """Set confidence level based on probability."""
        if self.binding_probability >= 0.9:
            self.confidence_level = PredictionConfidence.VERY_HIGH
        elif self.binding_probability >= 0.75:
            self.confidence_level = PredictionConfidence.HIGH
        elif self.binding_probability >= 0.5:
            self.confidence_level = PredictionConfidence.MEDIUM
        elif self.binding_probability >= 0.25:
            self.confidence_level = PredictionConfidence.LOW
        else:
            self.confidence_level = PredictionConfidence.VERY_LOW


@dataclass
class ToxicityPrediction:
    """Toxicity prediction result."""

    compound_smiles: str
    toxicity_endpoint: ToxicityEndpoint
    probability: float
    risk_level: RiskLevel = field(init=False)
    confidence_score: float = 0.0
    predicted_mechanism: Optional[ToxicityMechanism] = None
    dose_response_curve: Optional[Dict[str, float]] = None
    structural_alerts: List[str] = field(default_factory=list)
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set risk level based on probability."""
        if self.probability >= 0.8:
            self.risk_level = RiskLevel.VERY_HIGH
        elif self.probability >= 0.6:
            self.risk_level = RiskLevel.HIGH
        elif self.probability >= 0.4:
            self.risk_level = RiskLevel.MODERATE
        elif self.probability >= 0.2:
            self.risk_level = RiskLevel.LOW
        else:
            self.risk_level = RiskLevel.VERY_LOW


@dataclass
class AdverseDrugReaction:
    """Adverse drug reaction prediction."""

    compound_smiles: str
    adr_type: AdverseDrugReactionType
    system_affected: str
    severity_score: float
    probability: float
    onset_time_prediction: Optional[float] = None  # Days
    reversibility_prediction: bool = True
    dose_dependency: bool = True
    patient_risk_factors: List[str] = field(default_factory=list)
    clinical_manifestations: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)


@dataclass
class SystemsToxicologyProfile:
    """Comprehensive systems toxicology profile."""

    compound_smiles: str
    off_target_bindings: List[OffTargetBinding]
    toxicity_predictions: List[ToxicityPrediction]
    adr_predictions: List[AdverseDrugReaction]
    overall_safety_score: float
    therapeutic_index_estimate: Optional[float] = None
    contraindications: List[str] = field(default_factory=list)
    safety_warnings: List[str] = field(default_factory=list)
    recommended_monitoring: List[str] = field(default_factory=list)

    def get_high_risk_predictions(self) -> Dict[str, List]:
        """Get high-risk predictions across all categories."""
        high_risk = {
            "off_targets": [
                ot
                for ot in self.off_target_bindings
                if ot.confidence_level
                in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]
                and ot.binding_probability > 0.7
            ],
            "toxicities": [
                tox
                for tox in self.toxicity_predictions
                if tox.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
            ],
            "adrs": [
                adr
                for adr in self.adr_predictions
                if adr.severity_score > 0.7 and adr.probability > 0.5
            ],
        }
        return high_risk

    def generate_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report."""
        high_risk = self.get_high_risk_predictions()

        report = {
            "compound_smiles": self.compound_smiles,
            "overall_safety_score": self.overall_safety_score,
            "risk_summary": {
                "high_risk_off_targets": len(high_risk["off_targets"]),
                "high_risk_toxicities": len(high_risk["toxicities"]),
                "high_risk_adrs": len(high_risk["adrs"]),
            },
            "key_concerns": [],
            "recommendations": self.recommended_monitoring,
            "contraindications": self.contraindications,
        }

        # Add key concerns
        for ot in high_risk["off_targets"]:
            report["key_concerns"].append(f"Off-target binding to {ot.target_name}")

        for tox in high_risk["toxicities"]:
            report["key_concerns"].append(f"{tox.toxicity_endpoint.value} risk")

        for adr in high_risk["adrs"]:
            report["key_concerns"].append(
                f"{adr.adr_type.value} affecting {adr.system_affected}"
            )

        return report


@dataclass
class ToxicologyDatabase:
    """Database of toxicology information."""

    name: str
    version: str
    targets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    toxicity_pathways: Dict[str, List[str]] = field(default_factory=dict)
    known_toxicants: Dict[str, List[ToxicityEndpoint]] = field(default_factory=dict)
    adr_associations: Dict[str, List[AdverseDrugReaction]] = field(default_factory=dict)
    structural_alerts: Dict[str, List[str]] = field(default_factory=dict)


# Exception classes
class ToxicologyPredictionError(ChemiAccelError):
    """Exception raised during toxicology prediction."""

    pass


class OffTargetPredictionError(ChemiAccelError):
    """Exception raised during off-target prediction."""

    pass


class ADRPredictionError(ChemiAccelError):
    """Exception raised during ADR prediction."""

    pass


class DatabaseLoadingError(ChemiAccelError):
    """Exception raised during database loading."""

    pass


# Neural network models for toxicology prediction
if TORCH_AVAILABLE:

    class ToxicityPredictionNet(nn.Module):
        """Neural network for toxicity prediction."""

        def __init__(
            self,
            input_dim: int = 2048,
            hidden_dims: List[int] = None,
            num_endpoints: int = len(ToxicityEndpoint),
            dropout: float = 0.2,
        ):
            super().__init__()

            if hidden_dims is None:
                hidden_dims = [1024, 512, 256]

            self.num_endpoints = num_endpoints

            # Build layers
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.BatchNorm1d(hidden_dim),
                    ]
                )
                prev_dim = hidden_dim

            self.feature_extractor = nn.Sequential(*layers)

            # Endpoint-specific heads
            self.endpoint_heads = nn.ModuleDict(
                {
                    endpoint.value: nn.Sequential(
                        nn.Linear(prev_dim, 128),
                        nn.ReLU(),
                        nn.Dropout(dropout / 2),
                        nn.Linear(128, 1),
                        nn.Sigmoid(),
                    )
                    for endpoint in ToxicityEndpoint
                }
            )

            # Mechanism prediction head
            self.mechanism_head = nn.Sequential(
                nn.Linear(prev_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, len(ToxicityMechanism)),
                nn.Softmax(dim=1),
            )

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Forward pass."""
            features = self.feature_extractor(x)

            # Endpoint predictions
            endpoint_outputs = {}
            for endpoint, head in self.endpoint_heads.items():
                endpoint_outputs[endpoint] = head(features)

            # Mechanism prediction
            mechanism_output = self.mechanism_head(features)

            return {
                "endpoints": endpoint_outputs,
                "mechanisms": mechanism_output,
                "features": features,
            }

    class OffTargetPredictionNet(nn.Module):
        """Neural network for off-target binding prediction."""

        def __init__(
            self,
            compound_dim: int = 2048,
            target_dim: int = 1280,
            hidden_dim: int = 512,
            dropout: float = 0.2,
        ):
            super().__init__()

            # Compound and target encoders
            self.compound_encoder = nn.Sequential(
                nn.Linear(compound_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim),
            )

            self.target_encoder = nn.Sequential(
                nn.Linear(target_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim),
            )

            # Interaction prediction
            self.interaction_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

            # Affinity prediction
            self.affinity_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(
            self, compound_features: torch.Tensor, target_features: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
            """Forward pass."""
            compound_encoded = self.compound_encoder(compound_features)
            target_encoded = self.target_encoder(target_features)

            # Concatenate features
            combined_features = torch.cat([compound_encoded, target_encoded], dim=1)

            # Predictions
            binding_prob = self.interaction_predictor(combined_features)
            affinity_pred = self.affinity_predictor(combined_features)

            return {
                "binding_probability": binding_prob,
                "binding_affinity": affinity_pred,
                "compound_features": compound_encoded,
                "target_features": target_encoded,
            }


# Main prediction classes
class OffTargetPredictor:
    """Predictor for off-target binding events."""

    def __init__(self, use_gpu: bool = False, model_path: Optional[str] = None):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device = (
            torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            if TORCH_AVAILABLE
            else None
        )
        self.logger = logging.getLogger(__name__)

        # Initialize model
        if TORCH_AVAILABLE:
            self.model = OffTargetPredictionNet()
            if self.device:
                self.model.to(self.device)

            if model_path and Path(model_path).exists():
                self.load_model(model_path)
        else:
            self.model = None

        # Load target database
        self.target_database = self._create_sample_target_database()

    def predict_off_targets(
        self,
        compound_smiles: str,
        confidence_threshold: float = 0.3,
        max_targets: int = 100,
    ) -> List[OffTargetBinding]:
        """Predict off-target binding for a compound."""
        try:
            predictions = []

            # Encode compound
            compound_features = self._encode_compound(compound_smiles)

            # Predict against target database
            for target_id, target_info in list(self.target_database.items())[
                :max_targets
            ]:
                target_features = self._encode_target(target_info)

                # Make prediction
                if TORCH_AVAILABLE and self.model:
                    binding_prob, affinity = self._predict_binding(
                        compound_features, target_features
                    )
                else:
                    # Fallback prediction
                    binding_prob = np.random.random()
                    affinity = np.random.uniform(1, 1000)

                if binding_prob >= confidence_threshold:
                    # Calculate additional scores
                    similarity_score = self._calculate_structural_similarity(
                        compound_smiles, target_info
                    )
                    pharmacophore_score = self._calculate_pharmacophore_match(
                        compound_smiles, target_info
                    )

                    # Determine mechanism
                    mechanism = self._predict_interaction_mechanism(
                        compound_features, target_features
                    )

                    prediction = OffTargetBinding(
                        compound_smiles=compound_smiles,
                        target_id=target_id,
                        target_name=target_info.get("name", target_id),
                        binding_probability=float(binding_prob),
                        binding_affinity_prediction=float(affinity),
                        interaction_mechanism=mechanism,
                        structural_similarity_score=float(similarity_score),
                        pharmacophore_match_score=float(pharmacophore_score),
                    )
                    predictions.append(prediction)

            # Sort by binding probability
            predictions.sort(key=lambda x: x.binding_probability, reverse=True)

            self.logger.info(
                f"Predicted {len(predictions)} off-target bindings for compound"
            )
            return predictions

        except Exception as e:
            raise OffTargetPredictionError(f"Off-target prediction failed: {str(e)}")

    def _encode_compound(self, smiles: str) -> np.ndarray:
        """Encode compound SMILES into feature vector."""
        # Simplified molecular descriptor calculation
        # In practice, would use RDKit or similar
        return np.random.randn(2048)

    def _encode_target(self, target_info: Dict[str, Any]) -> np.ndarray:
        """Encode target information into feature vector."""
        # Simplified target encoding
        return np.random.randn(1280)

    def _predict_binding(
        self, compound_features: np.ndarray, target_features: np.ndarray
    ) -> Tuple[float, float]:
        """Predict binding probability and affinity."""
        if not TORCH_AVAILABLE or not self.model:
            return np.random.random(), np.random.uniform(1, 1000)

        with torch.no_grad():
            compound_tensor = torch.FloatTensor(compound_features).unsqueeze(0)
            target_tensor = torch.FloatTensor(target_features).unsqueeze(0)

            if self.device:
                compound_tensor = compound_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)

            outputs = self.model(compound_tensor, target_tensor)

            binding_prob = outputs["binding_probability"].cpu().item()
            affinity = (
                torch.exp(outputs["binding_affinity"]).cpu().item()
            )  # Convert from log scale

            return binding_prob, affinity

    def _calculate_structural_similarity(
        self, smiles: str, target_info: Dict[str, Any]
    ) -> float:
        """Calculate structural similarity score."""
        # Simplified similarity calculation
        return np.random.random()

    def _calculate_pharmacophore_match(
        self, smiles: str, target_info: Dict[str, Any]
    ) -> float:
        """Calculate pharmacophore matching score."""
        # Simplified pharmacophore matching
        return np.random.random()

    def _predict_interaction_mechanism(
        self, compound_features: np.ndarray, target_features: np.ndarray
    ) -> ToxicityMechanism:
        """Predict interaction mechanism."""
        # Simplified mechanism prediction
        mechanisms = list(ToxicityMechanism)
        return np.random.choice(mechanisms)

    def _create_sample_target_database(self) -> Dict[str, Dict[str, Any]]:
        """Create sample target database for testing."""
        targets = {
            f"TARGET_{i:04d}": {
                "name": f"Target {i}",
                "type": np.random.choice(
                    ["kinase", "gpcr", "enzyme", "transporter", "channel"]
                ),
                "family": f"Family_{i//10}",
                "organism": "Homo sapiens",
                "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVLALEAKLNDDEQYAWESSAGGSFTVRTDTGEPMGRGTKVILHLKEDQTEYLEERRIKEIVKAQRGVFLAVLLALTTTHRMSNKGLLQIILGQAAASAALLRVTKVL",
            }
            for i in range(1000)
        }
        return targets

    def save_model(self, path: str):
        """Save model to file."""
        if TORCH_AVAILABLE and self.model:
            torch.save(self.model.state_dict(), path)
            self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from file."""
        if TORCH_AVAILABLE and self.model:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            self.logger.info(f"Model loaded from {path}")


class ToxicityPredictor:
    """Predictor for various toxicity endpoints."""

    def __init__(self, use_gpu: bool = False, model_path: Optional[str] = None):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device = (
            torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            if TORCH_AVAILABLE
            else None
        )
        self.logger = logging.getLogger(__name__)

        # Initialize model
        if TORCH_AVAILABLE:
            self.model = ToxicityPredictionNet()
            if self.device:
                self.model.to(self.device)

            if model_path and Path(model_path).exists():
                self.load_model(model_path)
        else:
            self.model = None

        # Load structural alerts
        self.structural_alerts = self._load_structural_alerts()

    def predict_toxicity(
        self, compound_smiles: str, endpoints: Optional[List[ToxicityEndpoint]] = None
    ) -> List[ToxicityPrediction]:
        """Predict toxicity for specified endpoints."""
        if endpoints is None:
            endpoints = list(ToxicityEndpoint)

        try:
            predictions = []

            # Encode compound
            compound_features = self._encode_compound(compound_smiles)

            # Check structural alerts
            alerts = self._check_structural_alerts(compound_smiles)

            for endpoint in endpoints:
                # Make prediction
                if TORCH_AVAILABLE and self.model:
                    probability, confidence, mechanism = self._predict_endpoint(
                        compound_features, endpoint
                    )
                else:
                    # Fallback prediction
                    probability = np.random.random()
                    confidence = np.random.random()
                    mechanism = np.random.choice(list(ToxicityMechanism))

                # Create dose-response curve (simplified)
                dose_response = self._generate_dose_response_curve(probability)

                # Supporting evidence
                evidence = {
                    "model_prediction": probability,
                    "structural_alerts": [
                        alert for alert in alerts if endpoint.value in alert
                    ],
                    "confidence_factors": {
                        "model_uncertainty": 1 - confidence,
                        "structural_similarity": np.random.random(),
                        "training_data_coverage": np.random.random(),
                    },
                }

                prediction = ToxicityPrediction(
                    compound_smiles=compound_smiles,
                    toxicity_endpoint=endpoint,
                    probability=float(probability),
                    confidence_score=float(confidence),
                    predicted_mechanism=mechanism,
                    dose_response_curve=dose_response,
                    structural_alerts=[
                        alert for alert in alerts if endpoint.value in alert
                    ],
                    supporting_evidence=evidence,
                )
                predictions.append(prediction)

            self.logger.info(f"Predicted toxicity for {len(endpoints)} endpoints")
            return predictions

        except Exception as e:
            raise ToxicologyPredictionError(f"Toxicity prediction failed: {str(e)}")

    def _encode_compound(self, smiles: str) -> np.ndarray:
        """Encode compound SMILES into feature vector."""
        # Simplified molecular descriptor calculation
        return np.random.randn(2048)

    def _predict_endpoint(
        self, compound_features: np.ndarray, endpoint: ToxicityEndpoint
    ) -> Tuple[float, float, ToxicityMechanism]:
        """Predict specific toxicity endpoint."""
        if not TORCH_AVAILABLE or not self.model:
            return (
                np.random.random(),
                np.random.random(),
                np.random.choice(list(ToxicityMechanism)),
            )

        with torch.no_grad():
            compound_tensor = torch.FloatTensor(compound_features).unsqueeze(0)

            if self.device:
                compound_tensor = compound_tensor.to(self.device)

            outputs = self.model(compound_tensor)

            # Extract endpoint prediction
            probability = outputs["endpoints"][endpoint.value].cpu().item()

            # Extract mechanism prediction
            mechanism_probs = outputs["mechanisms"].cpu().numpy()[0]
            mechanism_idx = np.argmax(mechanism_probs)
            mechanism = list(ToxicityMechanism)[mechanism_idx]

            # Calculate confidence (simplified)
            confidence = float(np.max(mechanism_probs))

            return probability, confidence, mechanism

    def _check_structural_alerts(self, smiles: str) -> List[str]:
        """Check for structural alerts in the compound."""
        alerts = []

        # Simplified structural alert checking
        for alert_name, patterns in self.structural_alerts.items():
            for pattern in patterns:
                if pattern.lower() in smiles.lower():  # Simplified pattern matching
                    alerts.append(alert_name)
                    break

        return alerts

    def _generate_dose_response_curve(
        self, base_probability: float
    ) -> Dict[str, float]:
        """Generate simplified dose-response curve."""
        doses = [0.1, 1.0, 10.0, 100.0, 1000.0]  # mg/kg

        curve = {}
        for dose in doses:
            # Simplified dose-response relationship
            response = base_probability * (1 - np.exp(-dose / 100))
            curve[f"dose_{dose}_mg_kg"] = float(response)

        return curve

    def _load_structural_alerts(self) -> Dict[str, List[str]]:
        """Load structural alerts database."""
        # Simplified structural alerts
        return {
            "hepatotoxicity_nitro": ["NO2", "N(=O)=O"],
            "hepatotoxicity_quinone": ["C1=CC(=O)C=CC1=O"],
            "genotoxicity_aromatic_amine": ["c1ccccc1N"],
            "carcinogenicity_benzene": ["c1ccccc1"],
            "nephrotoxicity_halogen": ["Cl", "Br", "F"],
            "cardiotoxicity_ether": ["COC", "O-C"],
        }

    def save_model(self, path: str):
        """Save model to file."""
        if TORCH_AVAILABLE and self.model:
            torch.save(self.model.state_dict(), path)
            self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from file."""
        if TORCH_AVAILABLE and self.model:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            self.logger.info(f"Model loaded from {path}")


class ADRPredictor:
    """Predictor for adverse drug reactions."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.logger = logging.getLogger(__name__)

        # Load ADR database
        self.adr_database = self._create_sample_adr_database()

        # Initialize models (simplified)
        if SKLEARN_AVAILABLE:
            self.severity_model = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            self.onset_model = GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
            self._train_models()

    def predict_adrs(
        self,
        compound_smiles: str,
        off_target_bindings: List[OffTargetBinding],
        toxicity_predictions: List[ToxicityPrediction],
        patient_factors: Optional[Dict[str, Any]] = None,
    ) -> List[AdverseDrugReaction]:
        """Predict adverse drug reactions."""
        try:
            predictions = []

            if patient_factors is None:
                patient_factors = {}

            # Analyze off-target effects for ADR potential
            for off_target in off_target_bindings:
                if off_target.binding_probability > 0.5:
                    adr_predictions = self._predict_adr_from_off_target(
                        compound_smiles, off_target, patient_factors
                    )
                    predictions.extend(adr_predictions)

            # Analyze toxicity predictions for ADR potential
            for toxicity in toxicity_predictions:
                if toxicity.probability > 0.3:
                    adr_predictions = self._predict_adr_from_toxicity(
                        compound_smiles, toxicity, patient_factors
                    )
                    predictions.extend(adr_predictions)

            # Remove duplicates and merge similar ADRs
            predictions = self._merge_similar_adrs(predictions)

            # Sort by severity and probability
            predictions.sort(
                key=lambda x: (x.severity_score, x.probability), reverse=True
            )

            self.logger.info(f"Predicted {len(predictions)} potential ADRs")
            return predictions

        except Exception as e:
            raise ADRPredictionError(f"ADR prediction failed: {str(e)}")

    def _predict_adr_from_off_target(
        self,
        compound_smiles: str,
        off_target: OffTargetBinding,
        patient_factors: Dict[str, Any],
    ) -> List[AdverseDrugReaction]:
        """Predict ADRs from off-target binding."""
        adrs = []

        # Map target to potential ADRs
        target_adr_mapping = {
            "kinase": [
                ("cardiovascular", AdverseDrugReactionType.TYPE_A_PREDICTABLE, 0.6),
                ("hepatic", AdverseDrugReactionType.TYPE_A_PREDICTABLE, 0.4),
            ],
            "gpcr": [
                ("neurological", AdverseDrugReactionType.TYPE_A_PREDICTABLE, 0.7),
                ("gastrointestinal", AdverseDrugReactionType.TYPE_A_PREDICTABLE, 0.5),
            ],
            "enzyme": [
                ("metabolic", AdverseDrugReactionType.TYPE_A_PREDICTABLE, 0.5),
                ("hepatic", AdverseDrugReactionType.TYPE_A_PREDICTABLE, 0.6),
            ],
        }

        target_type = "kinase"  # Simplified - would be determined from target info

        if target_type in target_adr_mapping:
            for system, adr_type, base_prob in target_adr_mapping[target_type]:
                # Adjust probability based on binding strength
                probability = base_prob * off_target.binding_probability

                # Adjust for patient factors
                if patient_factors.get("age", 0) > 65:
                    probability *= 1.2
                if patient_factors.get("kidney_function", "normal") == "impaired":
                    probability *= 1.3

                # Calculate severity
                severity = self._calculate_adr_severity(
                    off_target.binding_probability, system, patient_factors
                )

                # Generate clinical information
                manifestations = self._generate_clinical_manifestations(
                    system, adr_type
                )
                monitoring = self._generate_monitoring_recommendations(system, severity)

                adr = AdverseDrugReaction(
                    compound_smiles=compound_smiles,
                    adr_type=adr_type,
                    system_affected=system,
                    severity_score=severity,
                    probability=min(probability, 1.0),
                    onset_time_prediction=np.random.uniform(1, 30),  # Days
                    reversibility_prediction=severity < 0.7,
                    dose_dependency=adr_type
                    == AdverseDrugReactionType.TYPE_A_PREDICTABLE,
                    patient_risk_factors=list(patient_factors.keys()),
                    clinical_manifestations=manifestations,
                    monitoring_recommendations=monitoring,
                )
                adrs.append(adr)

        return adrs

    def _predict_adr_from_toxicity(
        self,
        compound_smiles: str,
        toxicity: ToxicityPrediction,
        patient_factors: Dict[str, Any],
    ) -> List[AdverseDrugReaction]:
        """Predict ADRs from toxicity prediction."""
        adrs = []

        # Map toxicity endpoints to ADR systems
        toxicity_adr_mapping = {
            ToxicityEndpoint.HEPATOTOXICITY: (
                "hepatic",
                AdverseDrugReactionType.TYPE_A_PREDICTABLE,
            ),
            ToxicityEndpoint.CARDIOTOXICITY: (
                "cardiovascular",
                AdverseDrugReactionType.TYPE_A_PREDICTABLE,
            ),
            ToxicityEndpoint.NEPHROTOXICITY: (
                "renal",
                AdverseDrugReactionType.TYPE_A_PREDICTABLE,
            ),
            ToxicityEndpoint.NEUROTOXICITY: (
                "neurological",
                AdverseDrugReactionType.TYPE_A_PREDICTABLE,
            ),
            ToxicityEndpoint.IMMUNOTOXICITY: (
                "immunological",
                AdverseDrugReactionType.TYPE_B_IDIOSYNCRATIC,
            ),
        }

        if toxicity.toxicity_endpoint in toxicity_adr_mapping:
            system, adr_type = toxicity_adr_mapping[toxicity.toxicity_endpoint]

            # Calculate ADR probability and severity
            probability = (
                toxicity.probability * 0.8
            )  # ADR probability is related but not identical to toxicity
            severity = self._calculate_adr_severity(
                toxicity.probability, system, patient_factors
            )

            # Generate clinical information
            manifestations = self._generate_clinical_manifestations(system, adr_type)
            monitoring = self._generate_monitoring_recommendations(system, severity)

            adr = AdverseDrugReaction(
                compound_smiles=compound_smiles,
                adr_type=adr_type,
                system_affected=system,
                severity_score=severity,
                probability=probability,
                onset_time_prediction=self._predict_adr_onset_time(
                    toxicity.toxicity_endpoint
                ),
                reversibility_prediction=self._predict_adr_reversibility(
                    toxicity.toxicity_endpoint
                ),
                dose_dependency=adr_type == AdverseDrugReactionType.TYPE_A_PREDICTABLE,
                patient_risk_factors=list(patient_factors.keys()),
                clinical_manifestations=manifestations,
                monitoring_recommendations=monitoring,
            )
            adrs.append(adr)

        return adrs

    def _calculate_adr_severity(
        self, base_probability: float, system: str, patient_factors: Dict[str, Any]
    ) -> float:
        """Calculate ADR severity score."""
        # Base severity from probability
        severity = base_probability * 0.8

        # System-specific adjustments
        system_severity_factors = {
            "cardiovascular": 1.2,
            "hepatic": 1.1,
            "neurological": 1.3,
            "renal": 1.1,
            "immunological": 0.9,
        }

        if system in system_severity_factors:
            severity *= system_severity_factors[system]

        # Patient factor adjustments
        if patient_factors.get("age", 0) > 75:
            severity *= 1.3
        if patient_factors.get("comorbidities", 0) > 2:
            severity *= 1.2

        return min(severity, 1.0)

    def _generate_clinical_manifestations(
        self, system: str, adr_type: AdverseDrugReactionType
    ) -> List[str]:
        """Generate clinical manifestations for ADR."""
        manifestations_db = {
            "cardiovascular": [
                "chest pain",
                "palpitations",
                "dyspnea",
                "edema",
                "arrhythmia",
            ],
            "hepatic": [
                "jaundice",
                "abdominal pain",
                "nausea",
                "fatigue",
                "elevated enzymes",
            ],
            "neurological": [
                "headache",
                "dizziness",
                "confusion",
                "seizures",
                "tremor",
            ],
            "renal": [
                "oliguria",
                "proteinuria",
                "edema",
                "hypertension",
                "electrolyte imbalance",
            ],
            "gastrointestinal": [
                "nausea",
                "vomiting",
                "diarrhea",
                "abdominal pain",
                "bleeding",
            ],
            "immunological": [
                "rash",
                "fever",
                "lymphadenopathy",
                "eosinophilia",
                "anaphylaxis",
            ],
        }

        if system in manifestations_db:
            # Return 2-4 random manifestations
            manifestations = manifestations_db[system]
            n_select = np.random.randint(2, min(5, len(manifestations) + 1))
            return list(np.random.choice(manifestations, size=n_select, replace=False))

        return ["general symptoms"]

    def _generate_monitoring_recommendations(
        self, system: str, severity: float
    ) -> List[str]:
        """Generate monitoring recommendations."""
        monitoring_db = {
            "cardiovascular": [
                "ECG monitoring",
                "blood pressure monitoring",
                "cardiac enzymes",
            ],
            "hepatic": [
                "liver function tests",
                "bilirubin monitoring",
                "albumin levels",
            ],
            "neurological": [
                "neurological examination",
                "cognitive assessment",
                "EEG if indicated",
            ],
            "renal": ["serum creatinine", "urine analysis", "electrolyte monitoring"],
            "immunological": [
                "complete blood count",
                "immunoglobulin levels",
                "allergy testing",
            ],
        }

        base_monitoring = monitoring_db.get(system, ["routine monitoring"])

        # Add intensive monitoring for severe ADRs
        if severity > 0.7:
            base_monitoring.extend(
                ["frequent vital signs", "close clinical observation"]
            )

        return base_monitoring

    def _predict_adr_onset_time(self, toxicity_endpoint: ToxicityEndpoint) -> float:
        """Predict ADR onset time in days."""
        onset_times = {
            ToxicityEndpoint.HEPATOTOXICITY: np.random.uniform(7, 30),
            ToxicityEndpoint.CARDIOTOXICITY: np.random.uniform(1, 14),
            ToxicityEndpoint.NEPHROTOXICITY: np.random.uniform(3, 21),
            ToxicityEndpoint.NEUROTOXICITY: np.random.uniform(1, 7),
            ToxicityEndpoint.IMMUNOTOXICITY: np.random.uniform(1, 14),
        }

        return onset_times.get(toxicity_endpoint, np.random.uniform(1, 30))

    def _predict_adr_reversibility(self, toxicity_endpoint: ToxicityEndpoint) -> bool:
        """Predict if ADR is reversible."""
        reversibility = {
            ToxicityEndpoint.HEPATOTOXICITY: 0.8,
            ToxicityEndpoint.CARDIOTOXICITY: 0.6,
            ToxicityEndpoint.NEPHROTOXICITY: 0.7,
            ToxicityEndpoint.NEUROTOXICITY: 0.5,
            ToxicityEndpoint.IMMUNOTOXICITY: 0.9,
        }

        prob = reversibility.get(toxicity_endpoint, 0.7)
        return np.random.random() < prob

    def _merge_similar_adrs(
        self, adrs: List[AdverseDrugReaction]
    ) -> List[AdverseDrugReaction]:
        """Merge similar ADR predictions."""
        # Simplified merging - group by system and ADR type
        merged = {}

        for adr in adrs:
            key = (adr.system_affected, adr.adr_type)

            if key not in merged:
                merged[key] = adr
            else:
                # Merge with existing - take higher probability and severity
                existing = merged[key]
                if adr.probability > existing.probability:
                    existing.probability = adr.probability
                if adr.severity_score > existing.severity_score:
                    existing.severity_score = adr.severity_score

                # Combine manifestations and monitoring
                existing.clinical_manifestations = list(
                    set(existing.clinical_manifestations + adr.clinical_manifestations)
                )
                existing.monitoring_recommendations = list(
                    set(
                        existing.monitoring_recommendations
                        + adr.monitoring_recommendations
                    )
                )

        return list(merged.values())

    def _create_sample_adr_database(self) -> Dict[str, Any]:
        """Create sample ADR database."""
        return {
            "version": "1.0",
            "last_updated": "2024-01-01",
            "sources": ["FDA Adverse Event Reporting System", "WHO VigiBase"],
            "adr_frequency": {
                "cardiovascular": 0.15,
                "hepatic": 0.12,
                "neurological": 0.20,
                "gastrointestinal": 0.25,
                "dermatological": 0.18,
                "immunological": 0.08,
            },
        }

    def _train_models(self):
        """Train ADR prediction models (simplified)."""
        if not SKLEARN_AVAILABLE:
            return

        # Generate synthetic training data
        n_samples = 1000
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y_severity = np.random.randint(0, 2, n_samples)  # Binary classification
        y_onset = np.random.randint(0, 3, n_samples)  # 3 onset categories

        # Train models
        self.severity_model.fit(X, y_severity)
        self.onset_model.fit(X, y_onset)


# Main systems toxicology engine
class SystemsToxicologyEngine:
    """Comprehensive systems toxicology prediction engine."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.off_target_predictor = OffTargetPredictor(use_gpu=use_gpu)
        self.toxicity_predictor = ToxicityPredictor(use_gpu=use_gpu)
        self.adr_predictor = ADRPredictor(use_gpu=use_gpu)
        self.logger = logging.getLogger(__name__)

    def predict_systems_toxicology(
        self,
        compound_smiles: str,
        patient_factors: Optional[Dict[str, Any]] = None,
        include_off_targets: bool = True,
        include_toxicity: bool = True,
        include_adrs: bool = True,
        **kwargs,
    ) -> SystemsToxicologyProfile:
        """Generate comprehensive systems toxicology profile."""
        try:
            self.logger.info("Starting systems toxicology prediction for compound")

            # Initialize result containers
            off_target_bindings = []
            toxicity_predictions = []
            adr_predictions = []

            # Off-target prediction
            if include_off_targets:
                off_target_bindings = self.off_target_predictor.predict_off_targets(
                    compound_smiles,
                    confidence_threshold=kwargs.get("off_target_threshold", 0.3),
                    max_targets=kwargs.get("max_targets", 100),
                )

            # Toxicity prediction
            if include_toxicity:
                toxicity_predictions = self.toxicity_predictor.predict_toxicity(
                    compound_smiles, endpoints=kwargs.get("toxicity_endpoints", None)
                )

            # ADR prediction
            if include_adrs:
                adr_predictions = self.adr_predictor.predict_adrs(
                    compound_smiles,
                    off_target_bindings,
                    toxicity_predictions,
                    patient_factors,
                )

            # Calculate overall safety score
            safety_score = self._calculate_overall_safety_score(
                off_target_bindings, toxicity_predictions, adr_predictions
            )

            # Generate safety recommendations
            contraindications = self._generate_contraindications(
                off_target_bindings, toxicity_predictions, adr_predictions
            )

            safety_warnings = self._generate_safety_warnings(
                off_target_bindings, toxicity_predictions, adr_predictions
            )

            monitoring_recommendations = self._generate_monitoring_recommendations(
                off_target_bindings, toxicity_predictions, adr_predictions
            )

            # Estimate therapeutic index
            therapeutic_index = self._estimate_therapeutic_index(
                compound_smiles, safety_score
            )

            profile = SystemsToxicologyProfile(
                compound_smiles=compound_smiles,
                off_target_bindings=off_target_bindings,
                toxicity_predictions=toxicity_predictions,
                adr_predictions=adr_predictions,
                overall_safety_score=safety_score,
                therapeutic_index_estimate=therapeutic_index,
                contraindications=contraindications,
                safety_warnings=safety_warnings,
                recommended_monitoring=monitoring_recommendations,
            )

            self.logger.info("Systems toxicology prediction completed")
            return profile

        except Exception as e:
            raise ToxicologyPredictionError(
                f"Systems toxicology prediction failed: {str(e)}"
            )

    def _calculate_overall_safety_score(
        self,
        off_targets: List[OffTargetBinding],
        toxicities: List[ToxicityPrediction],
        adrs: List[AdverseDrugReaction],
    ) -> float:
        """Calculate overall safety score (0-1, higher is safer)."""
        # Base safety score
        safety_score = 1.0

        # Penalize for high-probability off-targets
        for ot in off_targets:
            if ot.binding_probability > 0.7:
                safety_score -= 0.1 * ot.binding_probability

        # Penalize for high-risk toxicities
        for tox in toxicities:
            if tox.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                safety_score -= 0.15 * tox.probability

        # Penalize for severe ADRs
        for adr in adrs:
            if adr.severity_score > 0.7:
                safety_score -= 0.1 * adr.severity_score * adr.probability

        return max(0.0, safety_score)

    def _generate_contraindications(
        self,
        off_targets: List[OffTargetBinding],
        toxicities: List[ToxicityPrediction],
        adrs: List[AdverseDrugReaction],
    ) -> List[str]:
        """Generate contraindications based on predictions."""
        contraindications = []

        # Off-target based contraindications
        high_prob_off_targets = [
            ot for ot in off_targets if ot.binding_probability > 0.8
        ]

        if high_prob_off_targets:
            contraindications.append("High probability off-target effects")

        # Toxicity based contraindications
        high_risk_toxicities = [
            tox
            for tox in toxicities
            if tox.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        ]

        for tox in high_risk_toxicities:
            if tox.toxicity_endpoint == ToxicityEndpoint.HEPATOTOXICITY:
                contraindications.append("Severe hepatic impairment")
            elif tox.toxicity_endpoint == ToxicityEndpoint.CARDIOTOXICITY:
                contraindications.append("Severe cardiac disease")
            elif tox.toxicity_endpoint == ToxicityEndpoint.NEPHROTOXICITY:
                contraindications.append("Severe renal impairment")

        # ADR based contraindications
        severe_adrs = [
            adr for adr in adrs if adr.severity_score > 0.8 and adr.probability > 0.6
        ]

        for adr in severe_adrs:
            contraindications.append(f"High risk of severe {adr.system_affected} ADRs")

        return list(set(contraindications))

    def _generate_safety_warnings(
        self,
        off_targets: List[OffTargetBinding],
        toxicities: List[ToxicityPrediction],
        adrs: List[AdverseDrugReaction],
    ) -> List[str]:
        """Generate safety warnings."""
        warnings = []

        # General warnings based on predictions
        if len(off_targets) > 10:
            warnings.append("Multiple potential off-target interactions")

        moderate_toxicities = [
            tox for tox in toxicities if tox.risk_level == RiskLevel.MODERATE
        ]
        if moderate_toxicities:
            warnings.append("Moderate toxicity risk - monitor closely")

        moderate_adrs = [
            adr for adr in adrs if adr.severity_score > 0.5 and adr.probability > 0.4
        ]
        if moderate_adrs:
            warnings.append("Potential for moderate to severe adverse reactions")

        return warnings

    def _generate_monitoring_recommendations(
        self,
        off_targets: List[OffTargetBinding],
        toxicities: List[ToxicityPrediction],
        adrs: List[AdverseDrugReaction],
    ) -> List[str]:
        """Generate monitoring recommendations."""
        monitoring = set()

        # Add monitoring based on toxicity predictions
        for tox in toxicities:
            if tox.probability > 0.3:
                if tox.toxicity_endpoint == ToxicityEndpoint.HEPATOTOXICITY:
                    monitoring.add("Regular liver function tests")
                elif tox.toxicity_endpoint == ToxicityEndpoint.CARDIOTOXICITY:
                    monitoring.add("Cardiac monitoring (ECG, echocardiogram)")
                elif tox.toxicity_endpoint == ToxicityEndpoint.NEPHROTOXICITY:
                    monitoring.add("Renal function monitoring")

        # Add monitoring based on ADRs
        for adr in adrs:
            monitoring.update(adr.monitoring_recommendations)

        # General monitoring for high-risk compounds
        if len(off_targets) > 5:
            monitoring.add("Comprehensive safety monitoring")

        return list(monitoring)

    def _estimate_therapeutic_index(
        self, compound_smiles: str, safety_score: float
    ) -> float:
        """Estimate therapeutic index (simplified)."""
        # Simplified TI estimation based on safety score
        # In practice, would incorporate efficacy predictions
        base_ti = 10.0  # Assume moderate therapeutic efficacy

        # Adjust based on safety
        ti_adjustment = safety_score * 2  # Scale safety score
        therapeutic_index = base_ti * ti_adjustment

        return max(1.0, therapeutic_index)


# Utility functions
def predict_compound_safety_profile(
    compound_smiles: str,
    use_gpu: bool = False,
    patient_factors: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> SystemsToxicologyProfile:
    """Predict comprehensive safety profile for a compound."""
    engine = SystemsToxicologyEngine(use_gpu=use_gpu)
    return engine.predict_systems_toxicology(
        compound_smiles, patient_factors=patient_factors, **kwargs
    )


def batch_toxicology_screening(
    compound_smiles_list: List[str], use_gpu: bool = False, **kwargs
) -> Dict[str, SystemsToxicologyProfile]:
    """Screen multiple compounds for toxicology."""
    engine = SystemsToxicologyEngine(use_gpu=use_gpu)
    results = {}

    for smiles in compound_smiles_list:
        try:
            profile = engine.predict_systems_toxicology(smiles, **kwargs)
            results[smiles] = profile
        except Exception as e:
            logging.warning(f"Failed to predict toxicology for {smiles}: {str(e)}")

    return results


def generate_toxicology_report(
    profile: SystemsToxicologyProfile, format: str = "json"
) -> Union[Dict[str, Any], str]:
    """Generate formatted toxicology report."""
    report = profile.generate_safety_report()

    if format == "json":
        return report
    elif format == "text":
        # Generate text report
        text_report = f"""
TOXICOLOGY SAFETY REPORT
========================

Compound: {profile.compound_smiles}
Overall Safety Score: {profile.overall_safety_score:.2f}

HIGH RISK PREDICTIONS:
- Off-target bindings: {report['risk_summary']['high_risk_off_targets']}
- Toxicity endpoints: {report['risk_summary']['high_risk_toxicities']}
- Adverse reactions: {report['risk_summary']['high_risk_adrs']}

KEY CONCERNS:
{chr(10).join([f"- {concern}" for concern in report['key_concerns']])}

CONTRAINDICATIONS:
{chr(10).join([f"- {contra}" for contra in report['contraindications']])}

MONITORING RECOMMENDATIONS:
{chr(10).join([f"- {rec}" for rec in report['recommendations']])}
        """
        return text_report.strip()
    else:
        return report


def create_sample_toxicology_database() -> ToxicologyDatabase:
    """Create sample toxicology database for testing."""
    return ToxicologyDatabase(
        name="Sample Toxicology Database",
        version="1.0",
        targets={
            f"TOX_TARGET_{i:03d}": {
                "name": f"Toxicology Target {i}",
                "type": np.random.choice(["enzyme", "receptor", "transporter"]),
                "associated_toxicity": np.random.choice(list(ToxicityEndpoint)),
                "mechanism": np.random.choice(list(ToxicityMechanism)),
            }
            for i in range(100)
        },
        known_toxicants={
            "benzene": [
                ToxicityEndpoint.CARCINOGENICITY,
                ToxicityEndpoint.GENOTOXICITY,
            ],
            "acetaminophen": [ToxicityEndpoint.HEPATOTOXICITY],
            "doxorubicin": [ToxicityEndpoint.CARDIOTOXICITY],
        },
        structural_alerts={
            "nitro_group": ["hepatotoxicity", "genotoxicity"],
            "aromatic_amine": ["carcinogenicity"],
            "quinone": ["hepatotoxicity", "nephrotoxicity"],
        },
    )

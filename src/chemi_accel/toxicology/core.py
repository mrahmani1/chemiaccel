"""
Core classes for toxicology and environmental fate modeling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.backend import Backend
from ..core.exceptions import ChemiAccelError
from ..core.molecular import MolecularData


class ToxicityEndpointType(Enum):
    """Types of toxicity endpoints."""

    ACUTE_ORAL = "acute_oral"
    ACUTE_DERMAL = "acute_dermal"
    ACUTE_INHALATION = "acute_inhalation"
    CHRONIC_ORAL = "chronic_oral"
    GENOTOXICITY = "genotoxicity"
    CARCINOGENICITY = "carcinogenicity"
    REPRODUCTIVE_TOXICITY = "reproductive_toxicity"
    DEVELOPMENTAL_TOXICITY = "developmental_toxicity"
    SKIN_SENSITIZATION = "skin_sensitization"
    EYE_IRRITATION = "eye_irritation"
    SKIN_IRRITATION = "skin_irritation"
    AQUATIC_ACUTE = "aquatic_acute"
    AQUATIC_CHRONIC = "aquatic_chronic"
    TERRESTRIAL_TOXICITY = "terrestrial_toxicity"
    BIOACCUMULATION = "bioaccumulation"
    PERSISTENCE = "persistence"
    ENDOCRINE_DISRUPTION = "endocrine_disruption"


class SpeciesType(Enum):
    """Species for toxicity testing."""

    HUMAN = "human"
    RAT = "rat"
    MOUSE = "mouse"
    RABBIT = "rabbit"
    DOG = "dog"
    FISH = "fish"
    DAPHNIA = "daphnia"
    ALGAE = "algae"
    EARTHWORM = "earthworm"
    BEE = "bee"


class ConfidenceLevel(Enum):
    """Confidence levels for predictions."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ToxicityEndpoint:
    """Data structure for toxicity endpoints."""

    endpoint_type: ToxicityEndpointType
    species: SpeciesType
    value: Optional[float] = None
    unit: Optional[str] = None
    study_type: Optional[str] = None
    exposure_duration: Optional[str] = None
    reference: Optional[str] = None
    quality_score: Optional[float] = None

    def __post_init__(self):
        """Validate endpoint data."""
        if self.value is not None and self.value < 0:
            raise ValueError("Toxicity value cannot be negative")


@dataclass
class ToxicityPrediction:
    """Results from toxicity prediction."""

    endpoint: ToxicityEndpoint
    predicted_value: float
    confidence: ConfidenceLevel
    uncertainty: float
    model_type: str
    features_used: List[str] = field(default_factory=list)
    explanation: Optional[Dict[str, Any]] = None

    @property
    def is_toxic(self) -> bool:
        """Determine if prediction indicates toxicity."""
        # This would use endpoint-specific thresholds
        if self.endpoint.endpoint_type == ToxicityEndpointType.ACUTE_ORAL:
            return self.predicted_value < 2000  # mg/kg (GHS Category 4)
        elif self.endpoint.endpoint_type == ToxicityEndpointType.CARCINOGENICITY:
            return self.predicted_value > 0.5  # Probability threshold
        elif self.endpoint.endpoint_type == ToxicityEndpointType.AQUATIC_ACUTE:
            return self.predicted_value < 100  # mg/L
        else:
            return self.predicted_value > 0.5  # Default probability threshold


@dataclass
class ToxicityData:
    """Comprehensive toxicity data for a compound."""

    smiles: str
    molecular_data: Optional[MolecularData] = None
    experimental_endpoints: List[ToxicityEndpoint] = field(default_factory=list)
    predicted_endpoints: List[ToxicityPrediction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_endpoint_predictions(
        self, endpoint_type: ToxicityEndpointType
    ) -> List[ToxicityPrediction]:
        """Get predictions for specific endpoint type."""
        return [
            pred
            for pred in self.predicted_endpoints
            if pred.endpoint.endpoint_type == endpoint_type
        ]

    def get_highest_risk_endpoints(self, n: int = 5) -> List[ToxicityPrediction]:
        """Get highest risk endpoint predictions."""
        toxic_predictions = [pred for pred in self.predicted_endpoints if pred.is_toxic]
        return sorted(toxic_predictions, key=lambda x: x.predicted_value, reverse=True)[
            :n
        ]

    @property
    def overall_risk_score(self) -> float:
        """Calculate overall risk score from all predictions."""
        if not self.predicted_endpoints:
            return 0.0

        # Weight different endpoint types
        weights = {
            ToxicityEndpointType.CARCINOGENICITY: 2.0,
            ToxicityEndpointType.GENOTOXICITY: 1.8,
            ToxicityEndpointType.REPRODUCTIVE_TOXICITY: 1.5,
            ToxicityEndpointType.ACUTE_ORAL: 1.2,
            ToxicityEndpointType.AQUATIC_ACUTE: 1.0,
        }

        total_score = 0.0
        total_weight = 0.0

        for pred in self.predicted_endpoints:
            weight = weights.get(pred.endpoint.endpoint_type, 1.0)
            # Convert to risk score (0-1, higher is worse)
            if pred.is_toxic:
                risk = (
                    pred.predicted_value / 100.0
                    if pred.predicted_value > 1
                    else pred.predicted_value
                )
            else:
                risk = 0.1  # Low baseline risk for non-toxic predictions

            total_score += risk * weight * (1.0 - pred.uncertainty)
            total_weight += weight

        return min(total_score / total_weight if total_weight > 0 else 0.0, 1.0)


class ToxicologyCalculator:
    """
    Main calculator for toxicology and environmental fate modeling.

    Provides unified interface for:
    - Multi-endpoint toxicity prediction
    - Environmental fate modeling
    - Metabolic pathway prediction
    - Risk assessment
    - Safer-by-design optimization
    """

    def __init__(
        self,
        backend: Optional[Union[str, Backend]] = None,
        use_gpu: bool = True,
        uncertainty_quantification: bool = True,
    ):
        """
        Initialize toxicology calculator.

        Args:
            backend: Computational backend ('cuda', 'jax', 'cpu')
            use_gpu: Whether to use GPU acceleration
            uncertainty_quantification: Enable uncertainty estimation
        """
        self.backend = self._setup_backend(backend, use_gpu)
        self.uncertainty_quantification = uncertainty_quantification

        # Initialize submodules
        self._multi_endpoint_predictor = None
        self._environmental_fate_predictor = None
        self._metabolic_predictor = None
        self._risk_assessor = None
        self._safer_designer = None
        self._visualizer = None

    def _setup_backend(
        self, backend: Optional[Union[str, Backend]], use_gpu: bool
    ) -> Backend:
        """Setup computational backend."""
        if isinstance(backend, Backend):
            return backend

        if backend is None:
            backend = "cuda" if use_gpu else "cpu"

        from ..core.backend import get_backend

        return get_backend(backend)

    def predict_toxicity(
        self,
        smiles: Union[str, List[str]],
        endpoints: Optional[List[ToxicityEndpointType]] = None,
        species: Optional[List[SpeciesType]] = None,
        include_uncertainty: bool = None,
        **kwargs,
    ) -> Union[ToxicityData, List[ToxicityData]]:
        """
        Predict toxicity for compound(s).

        Args:
            smiles: SMILES string(s)
            endpoints: Toxicity endpoints to predict
            species: Species to predict for
            include_uncertainty: Include uncertainty quantification

        Returns:
            ToxicityData object(s) with predictions
        """
        if include_uncertainty is None:
            include_uncertainty = self.uncertainty_quantification

        # Default endpoints if not specified
        if endpoints is None:
            endpoints = [
                ToxicityEndpointType.ACUTE_ORAL,
                ToxicityEndpointType.CARCINOGENICITY,
                ToxicityEndpointType.GENOTOXICITY,
                ToxicityEndpointType.AQUATIC_ACUTE,
                ToxicityEndpointType.SKIN_SENSITIZATION,
            ]

        # Default species if not specified
        if species is None:
            species = [SpeciesType.RAT, SpeciesType.FISH, SpeciesType.DAPHNIA]

        # Handle single SMILES
        if isinstance(smiles, str):
            return self._predict_single_compound(
                smiles, endpoints, species, include_uncertainty, **kwargs
            )

        # Handle multiple SMILES
        results = []
        for smi in smiles:
            result = self._predict_single_compound(
                smi, endpoints, species, include_uncertainty, **kwargs
            )
            results.append(result)

        return results

    def _predict_single_compound(
        self,
        smiles: str,
        endpoints: List[ToxicityEndpointType],
        species: List[SpeciesType],
        include_uncertainty: bool,
        **kwargs,
    ) -> ToxicityData:
        """Predict toxicity for single compound."""

        # Create molecular data
        from ..core.molecular import MolecularData

        molecular_data = MolecularData(smiles=smiles)

        # Get multi-endpoint predictions
        predictions = self.multi_endpoint_predictor.predict_endpoints(
            smiles, endpoints, species, include_uncertainty, **kwargs
        )

        return ToxicityData(
            smiles=smiles,
            molecular_data=molecular_data,
            predicted_endpoints=predictions,
        )

    def predict_environmental_fate(
        self,
        smiles: Union[str, List[str]],
        environmental_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Predict environmental fate and degradation."""
        return self.environmental_fate_predictor.predict_fate(
            smiles, environmental_conditions, **kwargs
        )

    def predict_metabolic_pathways(
        self,
        smiles: Union[str, List[str]],
        species: SpeciesType = SpeciesType.HUMAN,
        include_metabolite_toxicity: bool = True,
        **kwargs,
    ):
        """Predict metabolic pathways and metabolite toxicity."""
        return self.metabolic_predictor.predict_metabolism(
            smiles, species, include_metabolite_toxicity, **kwargs
        )

    def assess_holistic_risk(
        self,
        toxicity_data: ToxicityData,
        environmental_fate_data: Optional[Any] = None,
        exposure_scenarios: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        """Perform holistic risk assessment."""
        return self.risk_assessor.assess_risk(
            toxicity_data, environmental_fate_data, exposure_scenarios, **kwargs
        )

    def design_safer_chemicals(
        self,
        base_smiles: str,
        target_properties: Dict[str, float],
        toxicity_constraints: List[Dict[str, Any]],
        **kwargs,
    ):
        """Design safer-by-design chemical variants."""
        return self.safer_designer.optimize_for_safety(
            base_smiles, target_properties, toxicity_constraints, **kwargs
        )

    @property
    def multi_endpoint_predictor(self):
        """Get multi-endpoint toxicology predictor."""
        if self._multi_endpoint_predictor is None:
            from .multi_endpoint import MultiEndpointToxicologyPredictor

            self._multi_endpoint_predictor = MultiEndpointToxicologyPredictor(
                self.backend
            )
        return self._multi_endpoint_predictor

    @property
    def environmental_fate_predictor(self):
        """Get environmental fate predictor."""
        if self._environmental_fate_predictor is None:
            from .environmental_fate import EnvironmentalFatePredictor

            self._environmental_fate_predictor = EnvironmentalFatePredictor(
                self.backend
            )
        return self._environmental_fate_predictor

    @property
    def metabolic_predictor(self):
        """Get metabolic pathway predictor."""
        if self._metabolic_predictor is None:
            from .metabolic_prediction import MetabolicPathwayPredictor

            self._metabolic_predictor = MetabolicPathwayPredictor(self.backend)
        return self._metabolic_predictor

    @property
    def risk_assessor(self):
        """Get holistic risk assessor."""
        if self._risk_assessor is None:
            from .risk_assessment import HolisticRiskAssessment

            self._risk_assessor = HolisticRiskAssessment(self.backend)
        return self._risk_assessor

    @property
    def safer_designer(self):
        """Get safer-by-design optimizer."""
        if self._safer_designer is None:
            from .safer_design import SaferByDesignOptimizer

            self._safer_designer = SaferByDesignOptimizer(self.backend)
        return self._safer_designer

    @property
    def visualizer(self):
        """Get toxicology visualizer."""
        if self._visualizer is None:
            from .visualization import ToxicologyVisualizer

            self._visualizer = ToxicologyVisualizer()
        return self._visualizer

    def create_toxicity_report(
        self,
        toxicity_data: ToxicityData,
        include_environmental_fate: bool = True,
        include_metabolic_pathways: bool = True,
        include_risk_assessment: bool = True,
        output_format: str = "html",
    ) -> str:
        """Generate comprehensive toxicity report."""

        report_data = {
            "compound": toxicity_data.smiles,
            "overall_risk": toxicity_data.overall_risk_score,
            "endpoint_predictions": toxicity_data.predicted_endpoints,
            "high_risk_endpoints": toxicity_data.get_highest_risk_endpoints(),
        }

        # Add environmental fate if requested
        if include_environmental_fate:
            try:
                env_fate = self.predict_environmental_fate(toxicity_data.smiles)
                report_data["environmental_fate"] = env_fate
            except Exception as e:
                report_data["environmental_fate_error"] = str(e)

        # Add metabolic pathways if requested
        if include_metabolic_pathways:
            try:
                metabolism = self.predict_metabolic_pathways(toxicity_data.smiles)
                report_data["metabolic_pathways"] = metabolism
            except Exception as e:
                report_data["metabolism_error"] = str(e)

        # Add risk assessment if requested
        if include_risk_assessment:
            try:
                risk_assessment = self.assess_holistic_risk(toxicity_data)
                report_data["risk_assessment"] = risk_assessment
            except Exception as e:
                report_data["risk_assessment_error"] = str(e)

        # Generate report
        return self._format_report(report_data, output_format)

    def _format_report(self, report_data: Dict[str, Any], format_type: str) -> str:
        """Format toxicity report."""
        if format_type == "html":
            return self._generate_html_report(report_data)
        elif format_type == "json":
            import json

            return json.dumps(report_data, indent=2, default=str)
        elif format_type == "text":
            return self._generate_text_report(report_data)
        else:
            raise ValueError(f"Unsupported report format: {format_type}")

    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML toxicity report."""
        html = f"""
        <html>
        <head><title>Toxicity Report - {data['compound']}</title></head>
        <body>
        <h1>Comprehensive Toxicity Report</h1>
        <h2>Compound: {data['compound']}</h2>
        <h3>Overall Risk Score: {data['overall_risk']:.3f}</h3>

        <h3>High Risk Endpoints:</h3>
        <ul>
        """

        for endpoint in data["high_risk_endpoints"]:
            html += f"<li>{endpoint.endpoint.endpoint_type.value}: "
            html += f"{endpoint.predicted_value:.2f} "
            html += f"(Confidence: {endpoint.confidence.value})</li>"

        html += """
        </ul>
        </body>
        </html>
        """

        return html

    def _generate_text_report(self, data: Dict[str, Any]) -> str:
        """Generate text toxicity report."""
        report = f"Toxicity Report for {data['compound']}\n"
        report += "=" * 50 + "\n\n"
        report += f"Overall Risk Score: {data['overall_risk']:.3f}\n\n"

        report += "High Risk Endpoints:\n"
        for endpoint in data["high_risk_endpoints"]:
            report += f"- {endpoint.endpoint.endpoint_type.value}: "
            report += f"{endpoint.predicted_value:.2f} "
            report += f"(Confidence: {endpoint.confidence.value})\n"

        return report

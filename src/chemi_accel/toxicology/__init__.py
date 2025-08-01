"""
ChemiAccel Predictive Toxicology & Environmental Fate Modeling Module

AI-driven comprehensive toxicology and environmental fate modeling capabilities including:
- GPU-accelerated multi-endpoint toxicology prediction
- Advanced environmental degradation pathway & product prediction
- Metabolic pathway and metabolite toxicity prediction
- Holistic risk assessment and uncertainty quantification
- Inverse design for safer-by-design chemicals
"""

from .core import (
    ToxicityData,
    ToxicityEndpoint,
    ToxicityPrediction,
    ToxicologyCalculator,
)
from .environmental_fate import (
    DegradationPathway,
    DegradationProduct,
    EnvironmentalFatePredictor,
)
from .metabolic_prediction import (
    MetabolicNetwork,
    MetabolicPathwayPredictor,
    MetabolitePredictor,
)
from .multi_endpoint import (
    EndpointResult,
    MultiEndpointToxicologyPredictor,
    ToxicityModel,
)
from .risk_assessment import (
    HolisticRiskAssessment,
    RiskScore,
    UncertaintyQuantification,
)
from .safer_design import (
    SafeDesignCandidate,
    SaferByDesignOptimizer,
    ToxicityConstraint,
)
from .visualization import RiskLandscape, ToxicologyVisualizer

__all__ = [
    # Core classes
    "ToxicologyCalculator",
    "ToxicityData",
    "ToxicityEndpoint",
    "ToxicityPrediction",
    # Multi-endpoint toxicology
    "MultiEndpointToxicologyPredictor",
    "ToxicityModel",
    "EndpointResult",
    # Environmental fate
    "EnvironmentalFatePredictor",
    "DegradationPathway",
    "DegradationProduct",
    # Metabolic prediction
    "MetabolicPathwayPredictor",
    "MetabolitePredictor",
    "MetabolicNetwork",
    # Risk assessment
    "HolisticRiskAssessment",
    "RiskScore",
    "UncertaintyQuantification",
    # Safer design
    "SaferByDesignOptimizer",
    "SafeDesignCandidate",
    "ToxicityConstraint",
    # Visualization
    "ToxicologyVisualizer",
    "RiskLandscape",
]

__version__ = "0.1.0"

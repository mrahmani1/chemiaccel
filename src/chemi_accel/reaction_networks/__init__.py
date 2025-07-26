"""
Reaction Network Construction and Graph Representation Module

GPU-accelerated methods to parse reaction definitions and represent them 
as chemical graphs suitable for GNNs and mechanism prediction.

Main components:
- Reaction Network Construction: Build reaction networks from SMIRKS patterns
- Graph Representation: Chemical graph representations for reactions
- Elementary Step Enumeration: Generate plausible reaction steps
- Mechanism Prediction: AI models for reaction mechanism prediction
- Transition State Elucidation: Predict transition state geometries
"""

# Core Reaction Network Components
from .network_construction import (
    ReactionNetwork,
    ReactionNetworkBuilder,
    ReactionNode,
    ReactionEdge,
    ElementaryStep,
)

# Graph Representation
from .graph_representation import (
    ReactionGraph,
    ChemicalGraphConverter,
    GraphFeatureExtractor,
    ReactionGraphEmbedding,
)

# Elementary Step Generation
from .elementary_steps import (
    ElementaryStepEnumerator,
    BondChangeAnalyzer,
    ReactionPatternMatcher,
    SMIRKSProcessor,
)

# Mechanism Prediction
from .mechanism_prediction import (
    MechanismPredictor,
    TransitionStatePredictor,
    ReactionPathwayGenerator,
    EnergyProfileCalculator,
)

__all__ = [
    # Network Construction
    "ReactionNetwork",
    "ReactionNetworkBuilder", 
    "ReactionNode",
    "ReactionEdge",
    "ElementaryStep",
    # Graph Representation
    "ReactionGraph",
    "ChemicalGraphConverter",
    "GraphFeatureExtractor", 
    "ReactionGraphEmbedding",
    # Elementary Steps
    "ElementaryStepEnumerator",
    "BondChangeAnalyzer",
    "ReactionPatternMatcher",
    "SMIRKSProcessor",
    # Mechanism Prediction
    "MechanismPredictor",
    "TransitionStatePredictor",
    "ReactionPathwayGenerator",
    "EnergyProfileCalculator",
]

# Version information
__version__ = "1.0.0"
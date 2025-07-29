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
- Reaction Intelligence: AI-driven catalyst and condition optimization
- Knowledge Graph Integration: Literature-based reaction optimization
"""

# Elementary Step Generation
from .elementary_steps import (
    BondChangeAnalyzer,
    ElementaryStepEnumerator,
    ReactionPatternMatcher,
    SMIRKSProcessor,
)

# Graph Representation
from .graph_representation import (
    ChemicalGraphConverter,
    GraphFeatureExtractor,
    ReactionGraph,
    ReactionGraphEmbedding,
)

# Reaction Intelligence
from .intelligence import (
    CatalystSuggestion,
    CatalystSuggestionEngine,
    ExperimentalFeedbackProcessor,
    ExperimentalResult,
    ReactionConditions,
    ReactionIntelligenceEngine,
    ReactionScenario,
    WhatIfSimulator,
)

# Knowledge Graph Integration
from .kg_integration import (
    CatalystKnowledgeRetriever,
    KnowledgeGraphEnhancedSuggestionEngine,
    LiteratureBasedOptimizer,
)

# Mechanism Prediction
from .mechanism_prediction import (
    EnergyProfileCalculator,
    MechanismPredictor,
    ReactionPathwayGenerator,
    TransitionStatePredictor,
)

# Core Reaction Network Components
from .network_construction import (
    ElementaryStep,
    ReactionEdge,
    ReactionNetwork,
    ReactionNetworkBuilder,
    ReactionNode,
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
    # Reaction Intelligence
    "ReactionIntelligenceEngine",
    "CatalystSuggestionEngine",
    "WhatIfSimulator",
    "ExperimentalFeedbackProcessor",
    "ReactionConditions",
    "CatalystSuggestion",
    "ReactionScenario",
    "ExperimentalResult",
    # Knowledge Graph Integration
    "CatalystKnowledgeRetriever",
    "LiteratureBasedOptimizer",
    "KnowledgeGraphEnhancedSuggestionEngine",
]

# Version information
__version__ = "1.0.0"

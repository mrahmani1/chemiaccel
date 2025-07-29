"""
Retrosynthesis Module for ChemiAccel.

This module provides comprehensive GPU-accelerated retrosynthetic analysis
capabilities including one-step retrosynthetic disconnections, multi-step
route enumeration, synthesis tree construction, forward reaction prediction,
multi-objective optimization, and interactive visualization tools.
"""

from .generative_engine import RetrosynthesisEngine
from .route_enumeration import RouteEnumerator

__all__ = [
    "RetrosynthesisEngine",
    "RouteEnumerator",
]

# Generative Retrosynthesis Engine (with optional imports)
try:
    from .generative_engine import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
        ConfidenceLevel,
        DisconnectionError,
        DisconnectionPredictor,
        DisconnectionSite,
        DisconnectionStrategy,
        GraphRetroModel,
        ModelLoadingError,
        ModelPrediction,
        ModelType,
        MolecularRepresentation,
        PredictionError,
        ReactionTemplate,
        ReactionTemplateExtractor,
        ReactionType,
        RetrosynthesisConfig,
        RetrosynthesisEngine,
        RetrosynthesisError,
        RetrosynthesisResult,
        TransformerRetroModel,
        create_retro_config,
        decode_molecule,
        encode_molecule,
        extract_disconnection_sites,
        predict_retrosynthesis,
    )

    _GENERATIVE_ENGINE_AVAILABLE = True
except ImportError:
    _GENERATIVE_ENGINE_AVAILABLE = False

# Route Enumeration (with optional imports)
try:
    from .route_enumeration import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
        BuildingBlockDatabase,
        BuildingBlockQuery,
        BuildingBlockSource,
        ComplexityCalculator,
        DatabaseQueryError,
        OptimizationError,
        PruningCriteria,
        RouteEnumerationConfig,
        RouteEnumerationError,
        RouteEnumerator,
        RouteMetrics,
        RouteOptimizer,
        RouteStrategy,
        SynthesisComplexity,
        SynthesisNode,
        SynthesisRoute,
        SynthesisTree,
        SynthesisTreeBuilder,
        TreeConstructionError,
        build_synthesis_tree,
        calculate_complexity,
        create_route_config,
        enumerate_routes,
        optimize_route,
        query_building_blocks,
    )

    _ROUTE_ENUMERATION_AVAILABLE = True
except ImportError:
    _ROUTE_ENUMERATION_AVAILABLE = False

# Update __all__ with conditional imports
if _GENERATIVE_ENGINE_AVAILABLE:
    __all__.extend(
        [
            # Enums
            "ModelType",
            "DisconnectionStrategy",
            "ConfidenceLevel",
            "ReactionType",
            # Data structures
            "RetrosynthesisConfig",
            "MolecularRepresentation",
            "RetrosynthesisResult",
            "DisconnectionSite",
            "ReactionTemplate",
            "ModelPrediction",
            # Main classes
            "TransformerRetroModel",
            "GraphRetroModel",
            "DisconnectionPredictor",
            "ReactionTemplateExtractor",
            # Utility functions
            "create_retro_config",
            "encode_molecule",
            "decode_molecule",
            "predict_retrosynthesis",
            "extract_disconnection_sites",
            # Exceptions
            "RetrosynthesisError",
            "ModelLoadingError",
            "PredictionError",
            "DisconnectionError",
        ]
    )

if _ROUTE_ENUMERATION_AVAILABLE:
    __all__.extend(
        [
            # Enums
            "RouteStrategy",
            "PruningCriteria",
            "BuildingBlockSource",
            "SynthesisComplexity",
            # Data structures
            "SynthesisNode",
            "SynthesisTree",
            "RouteEnumerationConfig",
            "SynthesisRoute",
            "RouteMetrics",
            "BuildingBlockQuery",
            # Main classes
            "SynthesisTreeBuilder",
            "BuildingBlockDatabase",
            "RouteOptimizer",
            "ComplexityCalculator",
            # Utility functions
            "create_route_config",
            "build_synthesis_tree",
            "enumerate_routes",
            "optimize_route",
            "calculate_complexity",
            "query_building_blocks",
            # Exceptions
            "RouteEnumerationError",
            "TreeConstructionError",
            "DatabaseQueryError",
            "OptimizationError",
        ]
    )

# Forward Prediction (with optional imports)
try:
    from .forward_prediction import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
        ByproductPrediction,
        ForwardPredictionError,
        ForwardPredictionResult,
        ForwardReactionPredictor,
        KineticsEstimator,
        KineticsModel,
        KineticsModelingError,
        KineticsParameters,
        MolecularFeaturizer,
        ReactionCondition,
        ReactionConditions,
        RouteEvaluationResult,
        SelectivityPrediction,
        SelectivityPredictionError,
        SelectivityPredictor,
        SelectivityType,
        YieldCategory,
        YieldPredictionError,
        YieldPredictor,
        create_reaction_conditions,
        estimate_reaction_kinetics,
        evaluate_synthesis_route_forward,
        predict_reaction_yield,
    )

    _FORWARD_PREDICTION_AVAILABLE = True
except ImportError:
    _FORWARD_PREDICTION_AVAILABLE = False

# Multi-Objective Optimization (with optional imports)
try:
    from .multi_objective_optimization import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
        ConvergenceError,
        DominanceAnalyzer,
        DominanceRelation,
        MultiObjectiveConfig,
        MultiObjectiveOptimizationError,
        MultiObjectiveOptimizer,
        ObjectiveEvaluationError,
        ObjectiveEvaluator,
        OptimizationAlgorithm,
        OptimizationObjective,
        OptimizationObjectiveSpec,
        OptimizationResult,
        OptimizationSolution,
        ParetoFront,
        ParetoFrontError,
        ParetoFrontNavigator,
        ParetoRanking,
        create_multi_objective_config,
        find_pareto_optimal_routes,
        optimize_synthesis_routes,
    )

    _MULTI_OBJECTIVE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    _MULTI_OBJECTIVE_OPTIMIZATION_AVAILABLE = False

# Route Visualization (with optional imports)
try:
    from .route_visualization import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
        ColorScheme,
        InteractionError,
        InteractionMode,
        InteractiveVisualizer,
        LayoutAlgorithm,
        LayoutCalculator,
        LayoutError,
        MoleculeNode,
        MoleculeRenderer,
        ReactionEdge,
        RenderingError,
        VisualizationConfig,
        VisualizationData,
        VisualizationElement,
        VisualizationError,
        VisualizationType,
        create_pareto_front_plot,
        create_visualization_config,
        launch_streamlit_dashboard,
        visualize_synthesis_route_simple,
    )

    _ROUTE_VISUALIZATION_AVAILABLE = True
except ImportError:
    _ROUTE_VISUALIZATION_AVAILABLE = False

# Update __all__ with new conditional imports
if _FORWARD_PREDICTION_AVAILABLE:
    __all__.extend(
        [
            # Enums
            "ReactionCondition",
            "YieldCategory",
            "SelectivityType",
            "KineticsModel",
            # Data structures
            "ReactionConditions",
            "KineticsParameters",
            "SelectivityPrediction",
            "ByproductPrediction",
            "ForwardPredictionResult",
            "RouteEvaluationResult",
            # Main classes
            "MolecularFeaturizer",
            "YieldPredictor",
            "SelectivityPredictor",
            "KineticsEstimator",
            "ForwardReactionPredictor",
            # Utility functions
            "create_reaction_conditions",
            "predict_reaction_yield",
            "estimate_reaction_kinetics",
            "evaluate_synthesis_route_forward",
            # Exceptions
            "ForwardPredictionError",
            "KineticsModelingError",
            "YieldPredictionError",
            "SelectivityPredictionError",
        ]
    )

if _MULTI_OBJECTIVE_OPTIMIZATION_AVAILABLE:
    __all__.extend(
        [
            # Enums
            "OptimizationObjective",
            "OptimizationAlgorithm",
            "DominanceRelation",
            "ParetoRanking",
            # Data structures
            "OptimizationObjectiveSpec",
            "MultiObjectiveConfig",
            "OptimizationSolution",
            "ParetoFront",
            "OptimizationResult",
            # Main classes
            "ObjectiveEvaluator",
            "DominanceAnalyzer",
            "ParetoFrontNavigator",
            "MultiObjectiveOptimizer",
            # Utility functions
            "create_multi_objective_config",
            "optimize_synthesis_routes",
            "find_pareto_optimal_routes",
            # Exceptions
            "MultiObjectiveOptimizationError",
            "ParetoFrontError",
            "ObjectiveEvaluationError",
            "ConvergenceError",
        ]
    )

if _ROUTE_VISUALIZATION_AVAILABLE:
    __all__.extend(
        [
            # Enums
            "VisualizationType",
            "ColorScheme",
            "LayoutAlgorithm",
            "InteractionMode",
            # Data structures
            "VisualizationConfig",
            "VisualizationElement",
            "MoleculeNode",
            "ReactionEdge",
            "VisualizationData",
            # Main classes
            "MoleculeRenderer",
            "LayoutCalculator",
            "InteractiveVisualizer",
            # Utility functions
            "create_visualization_config",
            "visualize_synthesis_route_simple",
            "create_pareto_front_plot",
            "launch_streamlit_dashboard",
            # Exceptions
            "VisualizationError",
            "RenderingError",
            "LayoutError",
            "InteractionError",
        ]
    )

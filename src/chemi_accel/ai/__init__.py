"""
AI-Driven Discovery Engine for ChemiAccel.

This module provides advanced AI capabilities for molecular discovery including:
- Active Learning with multiple acquisition functions
- Uncertainty Quantification with various methods
- Inverse Design and molecular optimization
- Generative Chemistry engines
- Multi-objective optimization frameworks
"""

from .acquisition_functions import (
    BatchAcquisition,
    DiversityAcquisition,
    ExpectedImprovement,
    GreedyAcquisition,
    MultiObjectiveAcquisition,
    UncertaintyAcquisition,
    UpperConfidenceBound,
)
from .active_learning import ActiveLearningLoop
from .inverse_design import (
    ConstraintConfig,
    ConstraintHandler,
    InverseDesign,
    MolecularLatentSpace,
    MultiObjectiveOptimizer,
    OptimizationResult,
    PropertyPredictor,
    SimpleLatentSpace,
    create_drug_like_constraints,
    create_fragment_like_constraints,
    create_lead_like_constraints,
)
from .multi_objective_optimization import (
    CrowdingDistance,
    HypervolumeCalculator,
    InteractiveVisualization,
    MPOptimizer,
    MultiObjectiveResult,
    NonDominatedSorting,
    ObjectiveConfig,
    ParetoFrontResult,
    ScalarizationFunctions,
    create_drug_discovery_objectives,
    create_polymer_objectives,
    run_multi_objective_optimization,
)
from .uncertainty_quantification import UncertaintyQuantification

__all__ = [
    # Active Learning
    "ActiveLearningLoop",
    # Uncertainty Quantification
    "UncertaintyQuantification",
    # Acquisition Functions
    "UncertaintyAcquisition",
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "DiversityAcquisition",
    "GreedyAcquisition",
    "MultiObjectiveAcquisition",
    "BatchAcquisition",
    # Inverse Design
    "InverseDesign",
    "MultiObjectiveOptimizer",
    "OptimizationResult",
    "ConstraintConfig",
    "MolecularLatentSpace",
    "SimpleLatentSpace",
    "PropertyPredictor",
    "ConstraintHandler",
    "create_drug_like_constraints",
    "create_lead_like_constraints",
    "create_fragment_like_constraints",
    # Multi-Objective Optimization
    "MPOptimizer",
    "ObjectiveConfig",
    "ParetoFrontResult",
    "MultiObjectiveResult",
    "NonDominatedSorting",
    "CrowdingDistance",
    "HypervolumeCalculator",
    "ScalarizationFunctions",
    "InteractiveVisualization",
    "create_drug_discovery_objectives",
    "create_polymer_objectives",
    "run_multi_objective_optimization",
]

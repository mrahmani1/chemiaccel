"""
ChemiAccel Mixture & Formulation Design Module

AI-driven complex mixture and formulation design capabilities including:
- GPU-accelerated phase behavior prediction
- Inter-component interaction modeling
- Inverse design for optimal mixture composition
- Dynamic process integration for formulations
- Interactive visualization of mixture properties
"""

from .core import ComponentData, MixtureCalculator, MixtureData
from .dynamic_processes import DynamicProcessIntegrator, FormulationProcess
from .interactions import InteractionModeler, MolecularInteraction
from .inverse_design import MixtureInverseDesigner, OptimalComposition
from .phase_behavior import PhaseBehaviorPredictor, PhaseEquilibrium
from .visualization import MixtureVisualizer, PropertyLandscape

__all__ = [
    # Core classes
    "MixtureCalculator",
    "MixtureData",
    "ComponentData",
    # Phase behavior
    "PhaseBehaviorPredictor",
    "PhaseEquilibrium",
    # Interactions
    "InteractionModeler",
    "MolecularInteraction",
    # Inverse design
    "MixtureInverseDesigner",
    "OptimalComposition",
    # Dynamic processes
    "DynamicProcessIntegrator",
    "FormulationProcess",
    # Visualization
    "MixtureVisualizer",
    "PropertyLandscape",
]

__version__ = "0.1.0"

"""
Bio-Integrated Materials & Interfaces Module

This module provides AI-driven design tools for bio-integrated materials
and interfaces, including GPU-accelerated simulations, biological response
characterization, inverse design, toxicology integration, and visualization.
"""

from .biological_response import BiologicalResponseCharacterizer
from .integration import ToxicologyLifecycleIntegrator
from .inverse_design import BioOptimizedSurfaceDesigner
from .simulation import BioInterfaceSimulator
from .visualization import BioInterfaceVisualizer

__all__ = [
    "BioInterfaceSimulator",
    "BiologicalResponseCharacterizer",
    "BioOptimizedSurfaceDesigner",
    "ToxicologyLifecycleIntegrator",
    "BioInterfaceVisualizer",
]

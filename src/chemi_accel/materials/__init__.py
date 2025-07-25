"""
Materials Science Module for ChemiAccel.

This module provides comprehensive support for crystalline and periodic materials,
including CIF/POSCAR file parsing, geometric and topological descriptor calculations,
and integration with materials science workflows.
"""

from .crystal_descriptors import CrystalDescriptorCalculator
from .crystal_parser import CIFParser, POSCARParser
from .materials_calculator import MaterialsCalculator
from .periodic_descriptors import PeriodicDescriptorCalculator
from .validation import MaterialsValidator

__all__ = [
    "CIFParser",
    "POSCARParser",
    "CrystalDescriptorCalculator",
    "PeriodicDescriptorCalculator",
    "MaterialsCalculator",
    "MaterialsValidator",
]

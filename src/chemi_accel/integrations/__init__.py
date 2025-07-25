"""
Integration modules for ChemiAccel.

This package provides integration with popular scientific computing
and machine learning libraries including scikit-learn, Dask, and
Jupyter notebooks.
"""

from .dask_integration import (
    DaskChemiAccelCalculator,
    distributed_calculate_descriptors,
)
from .jupyter_integration import MolecularDisplayMixin, enable_rich_display
from .sklearn_integration import ChemiAccelEstimator, ChemiAccelTransformer

__all__ = [
    # scikit-learn integration
    "ChemiAccelTransformer",
    "ChemiAccelEstimator",
    # Jupyter integration
    "MolecularDisplayMixin",
    "enable_rich_display",
    # Dask integration
    "DaskChemiAccelCalculator",
    "distributed_calculate_descriptors",
]

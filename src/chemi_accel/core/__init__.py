"""Core abstractions and interfaces for ChemiAccel."""

from .calculator import Calculator
from .backend import BackendType, get_available_backends
from .molecular import MolecularData
from .exceptions import (
    ChemiAccelError,
    BackendUnavailableError,
    GPUMemoryError,
    InvalidMoleculeError,
    DescriptorCalculationError
)

__all__ = [
    "Calculator",
    "BackendType",
    "get_available_backends", 
    "MolecularData",
    "ChemiAccelError",
    "BackendUnavailableError",
    "GPUMemoryError", 
    "InvalidMoleculeError",
    "DescriptorCalculationError",
]
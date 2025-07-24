"""
ChemiAccel: High-performance GPU-accelerated molecular descriptor calculations.

A Python library providing massively parallel, hardware-accelerated alternatives 
to traditional CPU-based cheminformatics tools like RDKit and Mordred.
"""

__version__ = "0.1.0"
__author__ = "ChemiAccel Team"

# Core API exports
from .core.calculator import Calculator
from .core.backend import BackendType, get_available_backends
from .core.molecular import MolecularData

# Exception classes
from .core.exceptions import (
    ChemiAccelError,
    BackendUnavailableError,
    GPUMemoryError,
    InvalidMoleculeError,
    DescriptorCalculationError,
    DistributedComputingError
)

# Materials science modules (with optional imports)
try:
    from .materials import (
        MaterialsCalculator,
        CIFParser,
        POSCARParser,
        CrystalDescriptorCalculator,
        PeriodicDescriptorCalculator,
        MaterialsValidator
    )
    _MATERIALS_AVAILABLE = True
except ImportError:
    _MATERIALS_AVAILABLE = False

# Integration modules (with optional imports)
try:
    from .integrations.sklearn_integration import (
        ChemiAccelTransformer,
        ChemiAccelEstimator,
        create_molecular_pipeline
    )
    _SKLEARN_INTEGRATION_AVAILABLE = True
except ImportError:
    _SKLEARN_INTEGRATION_AVAILABLE = False

try:
    from .integrations.jupyter_integration import (
        enable_rich_display,
        quick_demo
    )
    _JUPYTER_INTEGRATION_AVAILABLE = True
except ImportError:
    _JUPYTER_INTEGRATION_AVAILABLE = False

try:
    from .integrations.dask_integration import (
        DaskChemiAccelCalculator,
        distributed_calculate_descriptors,
        setup_dask_cluster,
        process_large_dataset
    )
    _DASK_INTEGRATION_AVAILABLE = True
except ImportError:
    _DASK_INTEGRATION_AVAILABLE = False

try:
    from .integrations.hpc_integration import (
        HPCCalculator,
        MPIIntegration,
        SLURMJobManager,
        DaskClusterIntegration,
        RayClusterIntegration,
        detect_hpc_environment,
        create_hpc_calculator
    )
    _HPC_INTEGRATION_AVAILABLE = True
except ImportError:
    _HPC_INTEGRATION_AVAILABLE = False

# Base exports (always available)
__all__ = [
    # Core classes
    "Calculator",
    "BackendType", 
    "get_available_backends",
    "MolecularData",
    # Exceptions
    "ChemiAccelError",
    "BackendUnavailableError", 
    "GPUMemoryError",
    "InvalidMoleculeError",
    "DescriptorCalculationError",
    "DistributedComputingError",
]

# Add integration exports if available
if _SKLEARN_INTEGRATION_AVAILABLE:
    __all__.extend([
        "ChemiAccelTransformer",
        "ChemiAccelEstimator", 
        "create_molecular_pipeline"
    ])

if _JUPYTER_INTEGRATION_AVAILABLE:
    __all__.extend([
        "enable_rich_display",
        "quick_demo"
    ])

if _DASK_INTEGRATION_AVAILABLE:
    __all__.extend([
        "DaskChemiAccelCalculator",
        "distributed_calculate_descriptors",
        "setup_dask_cluster",
        "process_large_dataset"
    ])

if _HPC_INTEGRATION_AVAILABLE:
    __all__.extend([
        "HPCCalculator",
        "MPIIntegration", 
        "SLURMJobManager",
        "DaskClusterIntegration",
        "RayClusterIntegration",
        "detect_hpc_environment",
        "create_hpc_calculator"
    ])

if _MATERIALS_AVAILABLE:
    __all__.extend([
        "MaterialsCalculator",
        "CIFParser",
        "POSCARParser",
        "CrystalDescriptorCalculator",
        "PeriodicDescriptorCalculator",
        "MaterialsValidator"
    ])
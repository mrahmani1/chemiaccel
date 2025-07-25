"""
ChemiAccel Quantum Machine Learning Module

GPU-accelerated quantum machine learning model integration for molecular property prediction.
Provides standardized API for calling pre-trained QML models, on-the-fly property prediction,
streaming capabilities, and comprehensive quantum-derived property extraction.
"""

# Model implementations
from .ani import ANIModel

# Base classes and core functionality
from .base import (
    BaseQMLModel,
    QMLModelError,
    QMLModelRegistry,
    QMLPrediction,
    QuantumProperty,
    qml_registry,
)

# Advanced geometry and conformer handling
from .geometry_handler import (
    AdvancedConformerGenerator,
    AlignmentMethod,
    ConformerEnsemble,
    ConformerEnsembleProcessor,
    ConformerMethod,
    GeometryMetadata,
    GeometryQuality,
)
from .gfn2_xtb import GFN2xTBModel, GFN2xTBSurrogate

# Memory management for large-scale processing
from .memory_manager import (
    BatchingStrategy,
    BatchMetrics,
    MemoryAwareProcessor,
    MemoryMonitor,
    MemoryProfile,
    MemoryStrategy,
    SmartBatcher,
)

# On-the-fly prediction system
from .on_the_fly import (
    BatchPredictionRequest,
    BatchPredictionResult,
    ConformerGenerator,
    ConformerStrategy,
    GeometryInput,
    GPUBatchProcessor,
    OnTheFlyPredictor,
    PredictionMode,
)

# Property extraction and analysis
from .properties import PropertyMetadata, QuantumPropertyExtractor

# Streaming prediction system
from .streaming import (
    AdaptiveBuffer,
    BufferStrategy,
    FlowControl,
    StreamingConfig,
    StreamingMetrics,
    StreamingMode,
    StreamingPredictor,
    StreamingRequest,
    StreamingResponse,
)

# Uncertainty quantification
from .uncertainty_quantification import (
    EnsembleUncertaintyExtractor,
    MonteCarloDropoutExtractor,
    QMLUncertaintyConfig,
    QMLUncertaintyError,
    QMLUncertaintyQuantifier,
    QMLUncertaintyResult,
    QMLUncertaintyVisualizer,
    QMLUQIntegration,
    UncertaintyType,
)

# Utilities and acceleration
from .utils import (
    GPUAccelerator,
    QMLBenchmark,
    QMLModelLoader,
    create_molecular_hash,
    validate_qml_environment,
)

__all__ = [
    # Base classes and core
    "QuantumProperty",
    "QMLPrediction",
    "BaseQMLModel",
    "QMLModelRegistry",
    "QMLModelError",
    "qml_registry",
    # Model implementations
    "ANIModel",
    "GFN2xTBModel",
    "GFN2xTBSurrogate",
    # Property extraction
    "QuantumPropertyExtractor",
    "PropertyMetadata",
    # Utilities and acceleration
    "QMLModelLoader",
    "GPUAccelerator",
    "QMLBenchmark",
    "create_molecular_hash",
    "validate_qml_environment",
    # On-the-fly prediction
    "OnTheFlyPredictor",
    "BatchPredictionRequest",
    "BatchPredictionResult",
    "GeometryInput",
    "ConformerStrategy",
    "ConformerGenerator",
    "GPUBatchProcessor",
    "PredictionMode",
    # Memory management
    "MemoryAwareProcessor",
    "MemoryStrategy",
    "BatchingStrategy",
    "MemoryMonitor",
    "SmartBatcher",
    "MemoryProfile",
    "BatchMetrics",
    # Geometry handling
    "AdvancedConformerGenerator",
    "ConformerEnsemble",
    "ConformerEnsembleProcessor",
    "GeometryQuality",
    "ConformerMethod",
    "AlignmentMethod",
    "GeometryMetadata",
    # Streaming system
    "StreamingPredictor",
    "StreamingConfig",
    "StreamingRequest",
    "StreamingResponse",
    "StreamingMetrics",
    "StreamingMode",
    "BufferStrategy",
    "FlowControl",
    "AdaptiveBuffer",
    # Uncertainty quantification
    "QMLUncertaintyQuantifier",
    "QMLUncertaintyConfig",
    "QMLUncertaintyResult",
    "QMLUncertaintyError",
    "QMLUncertaintyVisualizer",
    "QMLUQIntegration",
    "EnsembleUncertaintyExtractor",
    "MonteCarloDropoutExtractor",
    "UncertaintyType",
]

# Version info
__version__ = "0.2.0"

"""
AI-Driven Predictive Crystallization Engineering Module

This module provides comprehensive tools for AI-driven crystallization process design,
optimization, and control, combining GPU-accelerated kinetics modeling with advanced
machine learning techniques for predictive engineering.

Core Components:
- GPU-accelerated crystallization kinetics modeling
- AI-driven process parameter optimization
- Crystal structure and property prediction integration
- Real-time in-operando crystallization control
- Interactive crystallization process simulation
"""

from .core import (
    CrystallizationCore,
    CrystalProperties,
    ProcessConditions,
    SimulationConfig,
)
from .interactive_simulation import (
    CrystallizationSimulator,
    PolymorphPurityMonitor,
    ProcessVisualization,
    SupersaturationTracker,
)
from .kinetics_modeling import (
    CrystallizationKineticsModel,
    GPUAcceleratedKinetics,
    MultiScaleKineticsModel,
    PopulationBalanceModel,
)
from .operando_control import (
    AdaptiveProcessControl,
    CrystallizationMonitor,
    InOperandoCharacterization,
    RealTimeController,
)
from .process_optimization import (
    AdaptiveDesignEngine,
    CrystallizationOptimizer,
    MultiObjectiveOptimizer,
    ProcessParameterOptimizer,
)
from .structure_integration import (
    CrystalHabitPredictor,
    ParticleSizeDistributor,
    PolymorphPredictor,
    StructurePropertyPredictor,
)

__all__ = [
    # Kinetics modeling
    "CrystallizationKineticsModel",
    "GPUAcceleratedKinetics",
    "MultiScaleKineticsModel",
    "PopulationBalanceModel",
    # Process optimization
    "CrystallizationOptimizer",
    "ProcessParameterOptimizer",
    "AdaptiveDesignEngine",
    "MultiObjectiveOptimizer",
    # Structure integration
    "StructurePropertyPredictor",
    "PolymorphPredictor",
    "CrystalHabitPredictor",
    "ParticleSizeDistributor",
    # Operando control
    "RealTimeController",
    "InOperandoCharacterization",
    "AdaptiveProcessControl",
    "CrystallizationMonitor",
    # Interactive simulation
    "CrystallizationSimulator",
    "ProcessVisualization",
    "SupersaturationTracker",
    "PolymorphPurityMonitor",
    # Core components
    "CrystallizationCore",
    "ProcessConditions",
    "CrystalProperties",
    "SimulationConfig",
]

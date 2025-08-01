"""
ChemiAccel Extreme Conditions Module

AI-Driven Design for Extreme Conditions & Non-Equilibrium Materials

This module provides advanced capabilities for simulating and analyzing materials
under extreme conditions such as high strain rates, radiation exposure, and
extreme temperatures/pressures.

Components:
    - GPU-accelerated non-equilibrium simulation engine
    - Inverse design for dynamic/non-equilibrium properties
    - Dynamic property characterization and failure mechanism identification
    - Experimental data integration for extreme conditions
    - Interactive visualization of dynamic material response
"""

from .experimental_integration import (
    ExtremeConditionsDataPipeline,
    HighTemperatureDataProcessor,
    RadiationDataProcessor,
    ShockExperimentProcessor,
)
from .inverse_design import (
    DynamicInverseDesigner,
    NonEquilibriumOptimizer,
    ProcessingPathwayOptimizer,
)
from .property_characterization import (
    DynamicPropertyAnalyzer,
    ExtremeConditionsCharacterizer,
    FailureMechanismIdentifier,
)
from .simulation_engine import (
    MultiScaleSimulator,
    NEMDSimulator,
    NonEquilibriumSimulationEngine,
    ReactiveForceFieldSimulator,
)
from .visualization import (
    DynamicResponseVisualizer,
    ExtremConditionsPlotter,
    MaterialDamageVisualizer,
)

__all__ = [
    # Simulation Engine
    "NonEquilibriumSimulationEngine",
    "NEMDSimulator",
    "ReactiveForceFieldSimulator",
    "MultiScaleSimulator",
    # Inverse Design
    "DynamicInverseDesigner",
    "NonEquilibriumOptimizer",
    "ProcessingPathwayOptimizer",
    # Property Characterization
    "DynamicPropertyAnalyzer",
    "FailureMechanismIdentifier",
    "ExtremeConditionsCharacterizer",
    # Experimental Integration
    "ExtremeConditionsDataPipeline",
    "ShockExperimentProcessor",
    "RadiationDataProcessor",
    "HighTemperatureDataProcessor",
    # Visualization
    "DynamicResponseVisualizer",
    "MaterialDamageVisualizer",
    "ExtremConditionsPlotter",
]

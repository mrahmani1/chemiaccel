"""
Digital Twin Module for Multi-Scale Material Modeling

This module provides a unified framework for linking and orchestrating models
across different length and time scales (atomistic, mesoscopic, continuum)
to create real-time material digital twins.

Main components:
- Multi-Scale Integration Framework: Links atomistic, mesoscopic, and continuum models
- Physics-Informed AI: Neural networks that incorporate physical laws
- Real-time Simulation Engine: GPU-accelerated multi-scale simulations
- Inverse Material Design: Digital twin-guided optimization
- Self-Calibration: Experimental data integration and model refinement
"""

# Multi-Scale Framework Core
from .multiscale_framework import (
    LengthScale,
    MaterialComponent,
    MultiScaleFramework,
    ScaleInterface,
    TimeScale,
)

# Physics-Informed AI Models
from .physics_informed_ai import (
    PDEConstraint,
    PhysicsInformedNeuralNetwork,
    PhysicsLoss,
    SurrogateModel,
    SurrogateModelTrainer,
)

# Real-time Simulation Engine
from .realtime_simulation import (
    InteractiveController,
    RealtimeSimulationEngine,
    SimulationParameters,
    SimulationState,
    UserInterface,
)

# Inverse Design Integration
from .inverse_design import (
    DigitalTwinGuidedOptimizer,
    MaterialDesignObjective,
    MaterialOptimizationConfig,
    MultiObjectiveDesigner,
)

# Self-Calibration and Data Integration
from .self_calibration import (
    CalibrationData,
    CalibrationResult,
    ExperimentalDataIntegrator,
    ModelCalibrator,
    SelfCalibrationEngine,
    UncertaintyQuantifier,
)

__all__ = [
    # Multi-Scale Framework
    "MultiScaleFramework",
    "LengthScale",
    "TimeScale",
    "ScaleInterface",
    "MaterialComponent",
    # Physics-Informed AI
    "PhysicsInformedNeuralNetwork",
    "SurrogateModel",
    "SurrogateModelTrainer",
    "PhysicsLoss",
    "PDEConstraint",
    # Real-time Simulation
    "RealtimeSimulationEngine",
    "SimulationState",
    "SimulationParameters",
    "InteractiveController",
    "UserInterface",
    # Inverse Design
    "DigitalTwinGuidedOptimizer",
    "MaterialDesignObjective",
    "MaterialOptimizationConfig",
    "MultiObjectiveDesigner",
    # Self-Calibration
    "SelfCalibrationEngine",
    "ModelCalibrator",
    "ExperimentalDataIntegrator",
    "UncertaintyQuantifier",
    "CalibrationData",
    "CalibrationResult",
]

# Version information
__version__ = "1.0.0"
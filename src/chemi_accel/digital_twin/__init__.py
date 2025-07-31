"""
Digital Twin Module for Multi-Scale Material Modeling and Multi-Product Processes

This module provides a unified framework for linking and orchestrating models
across different length and time scales (atomistic, mesoscopic, continuum)
to create real-time material digital twins and multi-product process digital twins.

Main components:
- Multi-Scale Integration Framework: Links atomistic, mesoscopic, and continuum models
- Physics-Informed AI: Neural networks that incorporate physical laws
- Real-time Simulation Engine: GPU-accelerated multi-scale simulations
- Inverse Material Design: Digital twin-guided optimization
- Self-Calibration: Experimental data integration and model refinement
- Multi-Product Process Digital Twin: Adaptive framework for multi-product chemical processes (Req 38.1-38.4)
- Rapid Changeover Optimization: GPU-accelerated optimization for product changeover (Req 38.2)
- Real-time Plant Data Calibration: Continuous sensor data calibration and validation (Req 38.3)
"""

# Fault Scenario Simulation and Corrective Actions
from .fault_scenario_simulation import (
    AlternativeRouteOptimizer,
    CorrectiveAction,
    CorrectiveActionType,
    DecisionSupportSystem,
    FaultAnalysisResult,
    FaultCause,
    FaultDetector,
    FaultInjector,
    FaultScenario,
    FaultScenarioSimulator,
    FaultSeverity,
    FaultType,
    create_contamination_fault,
    create_pressure_fault,
    create_temperature_fault,
)

# Fluid Dynamics Coupling
from .fluid_dynamics_coupling import (
    AdaptiveMeshRefinement,
    CouplingScheme,
    FiniteDifferenceHeatSolver,
    FluidDomain,
    FluidDynamicsCouplingManager,
    FluidDynamicsSolver,
    FluidProperties,
    FluidState,
    FluidThermalCoupledSolver,
    HeatTransferSolver,
    MaterialPropertyCoupling,
    SimpleFDFluidSolver,
)

# Inverse Design Integration
from .inverse_design import (
    DigitalTwinGuidedOptimizer,
    MaterialDesignObjective,
    MaterialOptimizationConfig,
    MultiObjectiveDesigner,
)

# Manufacturing Process Integration
from .manufacturing_integration import (
    FluidDynamicsSimulator,
    HeatTransferSimulator,
    ManufacturingDigitalTwin,
    MaterialFlowProperties,
    ProcessConditions,
    ProcessOutput,
    ProcessType,
)

# Multi-Product Process Digital Twin (Requirements 38.1-38.4)
from .multi_product_digital_twin import (
    ChangeoverPlan,
    ChangeoverStrategy,
    FlexibilityAssessment,
    FlexibilityMetric,
    MultiProductDigitalTwin,
    PlantDataSnapshot,
    ProcessConfiguration,
    ProcessMode,
    ProductFamily,
    ProductSpecification,
    ProductType,
    create_multi_product_digital_twin,
    create_product_family,
    create_product_specification,
    run_multi_product_digital_twin_demo,
)

# Multi-Scale Framework Core
from .multiscale_framework import (
    LengthScale,
    MaterialComponent,
    MultiScaleFramework,
    ScaleInterface,
    TimeScale,
)

# Parameter Space Exploration
from .parameter_space_exploration import (
    AdaptiveGridExplorer,
    ExplorationConfig,
    ExplorationStrategy,
    GaussianProcessExplorer,
    MultiObjectiveExplorer,
    ParameterExplorer,
    ParameterExplorerFactory,
    ParameterSpace,
    RLParameterExplorer,
)

# Physics-Informed AI Models
from .physics_informed_ai import (
    PDEConstraint,
    PhysicsInformedNeuralNetwork,
    PhysicsLoss,
    SurrogateModel,
    SurrogateModelTrainer,
)

# Real-time Plant Data Calibration and Validation (Requirement 38.3)
from .plant_data_calibration import (
    CalibrationEvent,
    CalibrationTrigger,
    DataQualityLevel,
    DataQualityMetrics,
    DataSourceType,
    PlantDataCalibrationEngine,
    PlantDataIngestionConfig,
    PlantDataValidator,
    SensorConfiguration,
    ValidationMethod,
    create_plant_data_calibration_engine,
    create_sensor_configuration,
    run_plant_data_calibration_demo,
)

# Process Engineering Interfaces
from .process_engineering_interfaces import (
    ANSYSFluentInterface,
    COMSOLInterface,
    ExternalSimulationInterface,
    MeshGeometry,
    OpenFOAMInterface,
    ProcessCouplingManager,
    ProcessEngineeringInterfaceFactory,
    SimulationConfig,
)

# Rapid Product Changeover Optimization (Requirement 38.2)
from .rapid_changeover_optimization import (
    ChangeoverOptimizationConfig,
    ChangeoverSchedule,
    ChangeoverTask,
    OptimizationObjective,
    RapidChangeoverOptimizer,
    ResourceConstraint,
    ResourceType,
    SchedulingAlgorithm,
    create_changeover_task,
    create_rapid_changeover_optimizer,
    create_resource_constraint,
    run_rapid_changeover_optimization_demo,
)

# Real-time Data Streaming
from .realtime_data_streaming import (
    AnomalyDetectionProcessor,
    DataExporter,
    DataFormat,
    DataPacket,
    DataStreamManager,
    PropertyAggregationProcessor,
    QualityOfService,
    RealTimeDataConnector,
    StreamingConfig,
    StreamingMode,
    StreamProcessor,
)

# Real-time Process Simulation
from .realtime_process_simulation import (
    MaterialProperty,
    MechanicalPropertyModel,
    MicrostructureState,
    PropertyEvolutionModel,
    PropertyType,
    RealTimeSimulationCore,
    SimulationConfig,
    SimulationMode,
    SimulationMonitor,
    SimulationState,
    ThermalPropertyModel,
    UpdateStrategy,
)

# Real-time Simulation Engine
from .realtime_simulation import (
    InteractiveController,
    RealtimeSimulationEngine,
    SimulationParameters,
    SimulationState,
    UserInterface,
)

# Real-time Visualization
from .realtime_visualization import (
    MatplotlibVisualizationEngine,
    PlotConfiguration,
    PlotlyVisualizationEngine,
    PlotType,
    RealTimeVisualizer,
    VisualizationConfig,
    VisualizationEngine,
    VisualizationMode,
    VisualizationServer,
)

# RL Agents
from .rl_agents import (
    DDPGAgent,
    PPOAgent,
    ProcessOptimizer,
    RLAgent,
    RLAgentFactory,
    SACAgent,
)

# Reinforcement Learning Process Optimization
from .rl_process_optimization import (
    Actor,
    Critic,
    EnergyEfficiencyRewardFunction,
    MultiObjectiveRewardFunction,
    OptimizationObjective,
    ProcessAction,
    ProcessEnvironment,
    ProcessState,
    QualityRewardFunction,
    ReplayBuffer,
    RLAlgorithm,
    RLConfig,
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
    # Manufacturing Process Integration
    "ProcessType",
    "ProcessConditions",
    "MaterialFlowProperties",
    "ProcessOutput",
    "FluidDynamicsSimulator",
    "HeatTransferSimulator",
    "ManufacturingDigitalTwin",
    # Process Engineering Interfaces
    "SimulationConfig",
    "MeshGeometry",
    "ExternalSimulationInterface",
    "OpenFOAMInterface",
    "ANSYSFluentInterface",
    "COMSOLInterface",
    "ProcessEngineeringInterfaceFactory",
    "ProcessCouplingManager",
    # Fluid Dynamics Coupling
    "CouplingScheme",
    "FluidProperties",
    "FluidDomain",
    "FluidState",
    "FluidDynamicsSolver",
    "HeatTransferSolver",
    "SimpleFDFluidSolver",
    "FiniteDifferenceHeatSolver",
    "FluidThermalCoupledSolver",
    "MaterialPropertyCoupling",
    "AdaptiveMeshRefinement",
    "FluidDynamicsCouplingManager",
    # Reinforcement Learning Process Optimization
    "RLConfig",
    "RLAlgorithm",
    "OptimizationObjective",
    "ProcessState",
    "ProcessAction",
    "ProcessEnvironment",
    "QualityRewardFunction",
    "EnergyEfficiencyRewardFunction",
    "MultiObjectiveRewardFunction",
    "Actor",
    "Critic",
    "ReplayBuffer",
    # RL Agents
    "RLAgent",
    "PPOAgent",
    "DDPGAgent",
    "SACAgent",
    "RLAgentFactory",
    "ProcessOptimizer",
    # Parameter Space Exploration
    "ExplorationStrategy",
    "ExplorationConfig",
    "ParameterSpace",
    "ParameterExplorer",
    "GaussianProcessExplorer",
    "MultiObjectiveExplorer",
    "AdaptiveGridExplorer",
    "ParameterExplorerFactory",
    "RLParameterExplorer",
    # Real-time Process Simulation
    "SimulationConfig",
    "SimulationMode",
    "PropertyType",
    "UpdateStrategy",
    "MaterialProperty",
    "MicrostructureState",
    "SimulationState",
    "PropertyEvolutionModel",
    "MechanicalPropertyModel",
    "ThermalPropertyModel",
    "RealTimeSimulationCore",
    "SimulationMonitor",
    # Real-time Visualization
    "VisualizationConfig",
    "VisualizationMode",
    "PlotType",
    "PlotConfiguration",
    "VisualizationEngine",
    "PlotlyVisualizationEngine",
    "MatplotlibVisualizationEngine",
    "RealTimeVisualizer",
    "VisualizationServer",
    # Real-time Data Streaming
    "StreamingConfig",
    "StreamingMode",
    "DataFormat",
    "QualityOfService",
    "DataPacket",
    "StreamProcessor",
    "PropertyAggregationProcessor",
    "AnomalyDetectionProcessor",
    "DataStreamManager",
    "RealTimeDataConnector",
    "DataExporter",
    # Fault Scenario Simulation and Corrective Actions
    "FaultType",
    "FaultSeverity",
    "FaultCause",
    "CorrectiveActionType",
    "FaultScenario",
    "CorrectiveAction",
    "FaultAnalysisResult",
    "FaultInjector",
    "FaultDetector",
    "DecisionSupportSystem",
    "AlternativeRouteOptimizer",
    "FaultScenarioSimulator",
    "create_temperature_fault",
    "create_contamination_fault",
    "create_pressure_fault",
    # Multi-Product Process Digital Twin (Requirements 38.1-38.4)
    "MultiProductDigitalTwin",
    "ProductSpecification",
    "ProductFamily",
    "ProductType",
    "ProcessConfiguration",
    "ProcessMode",
    "ChangeoverPlan",
    "ChangeoverStrategy",
    "PlantDataSnapshot",
    "FlexibilityAssessment",
    "FlexibilityMetric",
    "create_multi_product_digital_twin",
    "create_product_specification",
    "create_product_family",
    "run_multi_product_digital_twin_demo",
    # Rapid Product Changeover Optimization (Requirement 38.2)
    "RapidChangeoverOptimizer",
    "ChangeoverOptimizationConfig",
    "ChangeoverSchedule",
    "ChangeoverTask",
    "ResourceConstraint",
    "ResourceType",
    "OptimizationObjective",
    "SchedulingAlgorithm",
    "create_rapid_changeover_optimizer",
    "create_changeover_task",
    "create_resource_constraint",
    "run_rapid_changeover_optimization_demo",
    # Real-time Plant Data Calibration and Validation (Requirement 38.3)
    "PlantDataCalibrationEngine",
    "PlantDataValidator",
    "SensorConfiguration",
    "DataQualityMetrics",
    "DataQualityLevel",
    "CalibrationEvent",
    "CalibrationTrigger",
    "DataSourceType",
    "ValidationMethod",
    "PlantDataIngestionConfig",
    "create_plant_data_calibration_engine",
    "create_sensor_configuration",
    "run_plant_data_calibration_demo",
]

# Version information
__version__ = "1.0.0"

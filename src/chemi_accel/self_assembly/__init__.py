"""
Self-Assembly Simulation and Characterization Module for ChemiAccel.

This module provides comprehensive tools for GPU-accelerated coarse-grained molecular dynamics
and Dissipative Particle Dynamics (DPD) simulations to model mesoscale self-assembly phenomena.
It includes sophisticated characterization algorithms for analyzing emergent structures such as
micelles, vesicles, liquid crystals, and hydrogel networks.

Features:
- GPU-accelerated coarse-grained and DPD simulation engines
- Efficient particle interaction potentials for self-assembling systems
- Advanced structure classification and morphology analysis
- Specialized analyzers for micelles, vesicles, and liquid crystals
- Comprehensive clustering algorithms and phase behavior analysis
- Performance optimization for large-scale simulations
"""

from .characterization import (  # Main characterization engine; Data structures; Enums; Utility functions; Exceptions
    CharacterizationError,
    CharacterizationResult,
    ClusterAnalyzer,
    ClusteringError,
    ClusteringMethod,
    HydrogelCharacteristics,
    LiquidCrystalAnalyzer,
    LiquidCrystalCharacteristics,
    MicelleAnalyzer,
    MicelleCharacteristics,
    MorphologyAnalysisError,
    MorphologyType,
    PhaseAnalysisResult,
    PhaseType,
    SelfAssemblyCharacterizer,
    StructureClassifier,
    StructureMetrics,
    StructureType,
    VesicleAnalyzer,
    VesicleCharacteristics,
    analyze_micelle_formation,
    calculate_phase_diagram,
    characterize_trajectory,
)
from .coarse_grained_simulation import (  # Main simulation engine; Data structures; Enums; Utility functions; Exceptions
    Bond,
    BoundaryCondition,
    CoarseGrainedSimulationEngine,
    CoarseGrainedSimulationError,
    GPUMemoryError,
    IntegrationError,
    IntegratorType,
    InteractionParameters,
    InteractionType,
    MDIntegrator,
    Particle,
    ParticleInteractionCalculator,
    ParticleType,
    SimulationBox,
    SimulationMethod,
    SimulationParameters,
    SimulationState,
    TrajectoryFrame,
    create_amphiphilic_system,
    run_dpd_simulation,
)
from .interactive_visualization import (  # Main visualization engine; Configuration classes; Data structures; Enums; Utility functions; Exceptions
    AnimationSpeed,
    ColorScheme,
    DashboardConfig,
    DashboardError,
    DataProcessingError,
    ExportError,
    ExportFormat,
    InteractiveSelfAssemblyVisualizer,
    ParameterSweepData,
    PhaseTransitionAnalyzer,
    PhaseTransitionEvent,
    PropertyTrackingDashboard,
    RenderingError,
    TimeSeriesData,
    VisualizationConfig,
    VisualizationError,
    VisualizationFrame,
    VisualizationMode,
    create_comparative_visualization,
    create_demo_visualization,
    export_dashboard_data,
)
from .inverse_design import (  # Main inverse design engine; Optimizers; Data structures; Enums; Utility functions; Exceptions
    BaseOptimizer,
    BayesianOptimizer,
    DesignParameter,
    DesignSpace,
    DesignSpaceError,
    DesignTarget,
    EmergentPropertyInverseDesigner,
    EvaluationError,
    GeneticAlgorithmOptimizer,
    InverseDesignError,
    ObjectiveFunction,
    OptimizationError,
    OptimizationMethod,
    OptimizationObjective,
    OptimizationResult,
    PerformanceMetric,
    design_micelle_forming_surfactant,
    multi_objective_surfactant_design,
    optimize_environmental_conditions,
)
from .multiscale_integration import (  # Main multi-scale integration; Scale models; Data structures; Enums; Utility functions; Exceptions
    AtomisticModel,
    CouplingConfiguration,
    CouplingMethod,
    DataTransferEngine,
    DataTransferError,
    DataTransferMethod,
    MesoscaleModel,
    MolecularModel,
    MultiScaleIntegrationError,
    MultiScaleIntegrator,
    MultiScaleSystem,
    ScaleCouplingError,
    ScaleData,
    ScaleParameters,
    ScaleType,
    TransferResult,
    ValidationError,
    ValidationLevel,
    create_hierarchical_system,
    run_multiscale_workflow,
    validate_multiscale_consistency,
)
from .real_time_dashboard import (  # Real-time dashboard system; Configuration; Data structures; Enums; Utility functions
    AlertLevel,
    MetricStatus,
    PerformanceMetrics,
    RealTimeDashboardConfig,
    RealTimeDataManager,
    RealTimeMetric,
    RealTimeSelfAssemblyDashboard,
    SimulationAlert,
    create_demo_realtime_dashboard,
)

__all__ = [
    # Simulation engine exports
    "CoarseGrainedSimulationEngine",
    "ParticleInteractionCalculator",
    "MDIntegrator",
    # Simulation data structures
    "Particle",
    "ParticleType",
    "Bond",
    "SimulationBox",
    "SimulationParameters",
    "InteractionParameters",
    "SimulationState",
    "TrajectoryFrame",
    # Simulation enums
    "SimulationMethod",
    "InteractionType",
    "BoundaryCondition",
    "IntegratorType",
    # Simulation utilities
    "create_amphiphilic_system",
    "run_dpd_simulation",
    # Simulation exceptions
    "CoarseGrainedSimulationError",
    "GPUMemoryError",
    "IntegrationError",
    # Characterization engines
    "SelfAssemblyCharacterizer",
    "StructureClassifier",
    "ClusterAnalyzer",
    "MicelleAnalyzer",
    "VesicleAnalyzer",
    "LiquidCrystalAnalyzer",
    # Characterization data structures
    "StructureMetrics",
    "MicelleCharacteristics",
    "VesicleCharacteristics",
    "LiquidCrystalCharacteristics",
    "HydrogelCharacteristics",
    "PhaseAnalysisResult",
    "CharacterizationResult",
    # Characterization enums
    "StructureType",
    "MorphologyType",
    "PhaseType",
    "ClusteringMethod",
    # Characterization utilities
    "characterize_trajectory",
    "analyze_micelle_formation",
    "calculate_phase_diagram",
    # Characterization exceptions
    "CharacterizationError",
    "ClusteringError",
    "MorphologyAnalysisError",
    # Inverse design engines
    "EmergentPropertyInverseDesigner",
    "BaseOptimizer",
    "GeneticAlgorithmOptimizer",
    "BayesianOptimizer",
    # Inverse design data structures
    "DesignParameter",
    "ObjectiveFunction",
    "DesignSpace",
    "OptimizationResult",
    # Inverse design enums
    "OptimizationMethod",
    "DesignTarget",
    "OptimizationObjective",
    "PerformanceMetric",
    # Inverse design utilities
    "design_micelle_forming_surfactant",
    "optimize_environmental_conditions",
    "multi_objective_surfactant_design",
    # Inverse design exceptions
    "InverseDesignError",
    "OptimizationError",
    "DesignSpaceError",
    "EvaluationError",
    # Multi-scale integration
    "MultiScaleIntegrator",
    "DataTransferEngine",
    "AtomisticModel",
    "MolecularModel",
    "MesoscaleModel",
    # Multi-scale data structures
    "ScaleParameters",
    "CouplingConfiguration",
    "MultiScaleSystem",
    "ScaleData",
    "TransferResult",
    # Multi-scale enums
    "ScaleType",
    "CouplingMethod",
    "DataTransferMethod",
    "ValidationLevel",
    # Multi-scale utilities
    "create_hierarchical_system",
    "run_multiscale_workflow",
    "validate_multiscale_consistency",
    # Multi-scale exceptions
    "MultiScaleIntegrationError",
    "ScaleCouplingError",
    "DataTransferError",
    "ValidationError",
    # Interactive visualization engines
    "InteractiveSelfAssemblyVisualizer",
    "PropertyTrackingDashboard",
    "PhaseTransitionAnalyzer",
    # Visualization configuration classes
    "VisualizationConfig",
    "DashboardConfig",
    # Visualization data structures
    "VisualizationFrame",
    "TimeSeriesData",
    "PhaseTransitionEvent",
    "ParameterSweepData",
    # Visualization enums
    "VisualizationMode",
    "ColorScheme",
    "AnimationSpeed",
    "ExportFormat",
    # Visualization utility functions
    "create_comparative_visualization",
    "export_dashboard_data",
    "create_demo_visualization",
    # Visualization exceptions
    "VisualizationError",
    "RenderingError",
    "DashboardError",
    "ExportError",
    "DataProcessingError",
    # Real-time dashboard system
    "RealTimeSelfAssemblyDashboard",
    "RealTimeDataManager",
    "RealTimeDashboardConfig",
    "RealTimeMetric",
    "SimulationAlert",
    "PerformanceMetrics",
    "AlertLevel",
    "MetricStatus",
    "create_demo_realtime_dashboard",
]

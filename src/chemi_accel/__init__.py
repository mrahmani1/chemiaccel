"""
ChemiAccel: High-performance GPU-accelerated molecular descriptor calculations.

A Python library providing massively parallel, hardware-accelerated alternatives
to traditional CPU-based cheminformatics tools like RDKit and Mordred.
"""

__version__ = "0.1.0"
__author__ = "ChemiAccel Team"

from .core.backend import BackendType, get_available_backends

# Core API exports
from .core.calculator import Calculator

# Exception classes
from .core.exceptions import (
    BackendUnavailableError,
    ChemiAccelError,
    DescriptorCalculationError,
    DistributedComputingError,
    GPUMemoryError,
    InvalidMoleculeError,
)
from .core.molecular import MolecularData

# Materials science modules (with optional imports)
try:
    from .materials import (
        CIFParser,
        CrystalDescriptorCalculator,
        MaterialsCalculator,
        MaterialsValidator,
        PeriodicDescriptorCalculator,
        POSCARParser,
    )

    _MATERIALS_AVAILABLE = True
except ImportError:
    _MATERIALS_AVAILABLE = False

# Crystal Structure Prediction (with optional imports)
try:
    from .materials import (  # GPU-accelerated CSP; AI-driven structure prediction; Polymorphic landscape exploration; Lattice energy minimization; Packing motif prediction; Search algorithms; Neural networks; Enums; Data structures; Main classes; Utility functions; Exceptions; Enums; Data structures; Neural network classes; Main classes; Utility functions; Exceptions
        AtomicPosition,
        ConvergenceError,
        CrystalGraphConv,
        CrystalPredictionConfig,
        CrystalStructure,
        CrystalStructurePredictionError,
        CrystalSystemType,
        CSPMethod,
        CSPPrediction,
        GPUAcceleratedCSPEngine,
        LatticeEnergyError,
        LatticeEnergyMethod,
        LatticeEnergyPredictor,
        PackingMotifGNN,
        PackingMotifType,
        StructureGenerationError,
        SymmetryError,
        SymmetryHandling,
        UnitCellGenerator,
        UnitCellParameters,
        analyze_polymorphs,
        create_csp_config,
        predict_crystal_structure,
    )

    _CSP_AVAILABLE = True
except ImportError:
    _CSP_AVAILABLE = False

# Intermolecular Interactions (with optional imports)
try:
    from .materials import (  # GPU-accelerated intermolecular interactions; Hydrogen bonding networks; π-π stacking; van der Waals; Crystal packing optimization; Enums; Data structures; Main classes; Utility functions; Exceptions; Enums; Data structures; Main classes; Utility functions; Exceptions
        AtomicCoordinates,
        CrystalPackingObjective,
        CrystalPackingOptimizationError,
        CrystalPackingOptimizer,
        CrystalPackingResult,
        GPUAcceleratedInteractionEngine,
        HydrogenBond,
        HydrogenBondCalculationError,
        HydrogenBondCalculator,
        HydrogenBondType,
        InteractionConfiguration,
        InteractionEngineError,
        InteractionPair,
        InteractionType,
        IntermolecularInteractionError,
        MolecularFragment,
        PiPiInteraction,
        PiPiStackingCalculationError,
        PiPiStackingCalculator,
        PiPiStackingType,
        VanDerWaalsCalculationError,
        VanDerWaalsCalculator,
        VanDerWaalsInteraction,
        analyze_crystal_packing,
        create_interaction_config,
        create_molecular_fragment_from_smiles,
        optimize_intermolecular_interactions,
    )

    _INTERACTIONS_AVAILABLE = True
except ImportError:
    _INTERACTIONS_AVAILABLE = False

# Polymorph Inverse Design (with optional imports)
try:
    from .materials import (  # AI-driven polymorph control; Molecular modification strategies; Inverse design optimization; Polymorph prediction; Enums; Data structures; Main classes; Utility functions; Exceptions; Enums; Data structures; Main classes; Utility functions; Exceptions
        ConfigurationError,
        DesignCandidate,
        DesignObjective,
        DesignOptimizationError,
        ModificationStrategy,
        MolecularModification,
        MolecularModificationError,
        MolecularModifier,
        PolymorphDesignConfig,
        PolymorphInverseDesigner,
        PolymorphInverseDesignError,
        PolymorphPredictionError,
        PolymorphPredictor,
        PolymorphTarget,
        PolymorphType,
        analyze_polymorph_landscape,
        create_design_config,
        create_polymorph_target,
        design_for_polymorph,
        predict_polymorph_stability,
    )

    _POLYMORPH_DESIGN_AVAILABLE = True
except ImportError:
    _POLYMORPH_DESIGN_AVAILABLE = False

# Solid-State Properties (with optional imports)
try:
    from .materials import (  # GPU-accelerated property prediction; Density calculations; Elastic moduli; Thermal properties; Solubility prediction; ML-based models; Enums; Data structures; Main classes; Utility functions; Exceptions; Enums; Data structures; Main classes; Utility functions; Exceptions
        CrystalStructureData,
        DensityCalculationError,
        DensityCalculator,
        ElasticModuliCalculationError,
        ElasticModuliPredictor,
        ElasticProperties,
        ElasticProperty,
        ElasticPropertyNet,
        PropertyPredictionConfig,
        PropertyPredictionError,
        PropertyResult,
        PropertyType,
        SolidStatePropertyEngine,
        SolidStatePropertyError,
        SolubilityPredictionError,
        SolubilityPredictor,
        SolubilityResult,
        SolubilityType,
        ThermalProperties,
        ThermalProperty,
        ThermalPropertyError,
        ThermalPropertyPredictor,
        calculate_structure_density,
        create_crystal_structure,
        create_property_config,
        predict_elastic_moduli,
        predict_solid_state_properties,
        predict_solubility,
        predict_thermal_properties,
    )

    _SOLID_STATE_PROPERTIES_AVAILABLE = True
except ImportError:
    _SOLID_STATE_PROPERTIES_AVAILABLE = False

# Experimental Validation (with optional imports)
try:
    from .materials import (  # Experimental validation and refinement; XRD pattern simulation; Data comparison; Feedback loops; Model refinement; Enums; Data structures; Main classes; Utility functions; Exceptions; Enums; Data structures; Main classes; Utility functions; Exceptions
        ComparisonResult,
        DataValidationError,
        ExperimentalComparator,
        ExperimentalData,
        ExperimentalDataValidator,
        ExperimentalTechnique,
        ExperimentalValidationEngine,
        ExperimentalValidationError,
        FeedbackData,
        FeedbackLoop,
        FeedbackLoopError,
        ModelRefinementEngine,
        ModelRefinementError,
        PowderXRDPattern,
        RefinementConfig,
        RefinementStrategy,
        ValidationMetric,
        ValidationResult,
        XRDComparisonError,
        XRDPatternSimulator,
        compare_with_experiment,
        create_experimental_data,
        create_refinement_config,
        setup_feedback_loop,
        simulate_xrd_pattern,
        validate_experimental_quality,
    )

    _EXPERIMENTAL_VALIDATION_AVAILABLE = True
except ImportError:
    _EXPERIMENTAL_VALIDATION_AVAILABLE = False

# Sustainability modules (with optional imports)
try:
    from .sustainability import (  # Degradation modeling; Recyclability assessment; LCA integration; Inverse design for sustainable materials; Interactive lifecycle simulation and visualization
        AtomisticMDModel,
        BiologicalRecyclingPredictor,
        ChemicalRecyclingPredictor,
        ComparisonMetric,
        ComprehensiveRecyclabilityAssessor,
        DesignResult,
        GPUAcceleratedLCACalculator,
        LCAComparator,
        LifecycleEvent,
        LifecyclePhase,
        LifecycleSimulator,
        LifecycleTrajectory,
        LifecycleVisualizer,
        MaterialComparator,
        MaterialParameter,
        MatrixLCACalculator,
        MechanicalRecyclingPredictor,
        MultiScaleDegradationPredictor,
        OptimizationAlgorithm,
        OptimizationConfig,
        OptimizationObjective,
        SimulationConfig,
        SurrogateAIModel,
        SustainabilityDashboard,
        SustainabilityPropertyPredictor,
        SustainabilityTarget,
        SustainableInverseDesigner,
        analyze_material_trade_offs,
        assess_material_recyclability,
        calculate_material_lca,
        create_accelerated_aging_conditions,
        create_polymer_design_parameters,
        create_polymer_manufacturing_process,
        create_standard_material,
        create_standard_plastic_composition,
        create_sustainability_dashboard,
        create_sustainability_objectives,
        design_sustainable_material,
        run_degradation_prediction,
        simulate_material_lifecycle,
    )

    _SUSTAINABILITY_AVAILABLE = True
except ImportError:
    _SUSTAINABILITY_AVAILABLE = False

# Spectroscopy modules (with optional imports)
try:
    from .spectroscopy import (  # Predictive models; Experimental processing; Inverse spectroscopy; Quantum methods; Utility functions; Predictive models; Model configuration; Enums; Experimental processing; Processing enums; Inverse spectroscopy; Inverse enums; Quantum methods; Quantum data structures; Quantum enums; Utility functions; Exceptions
        ApproximateMethodEngine,
        ApproximationLevel,
        BaselineCorrectionMethod,
        BasisSetError,
        BasisSetOptimizer,
        BasisSetType,
        ConvergenceError,
        ConvolutionalSpectralNet,
        DataQualityError,
        DensityFunctionalApproximation,
        DFTSurrogate,
        ElectronicStructure,
        ExperimentalSpectraProcessor,
        ExperimentalSpectrum,
        FastQuantumChemistry,
        FormatError,
        FunctionalType,
        GPUSpectroscopyPredictor,
        GraphSpectralNet,
        InverseError,
        InverseResult,
        InverseSpectroscopyEngine,
        IRPredictor,
        MatchingError,
        MatchingMethod,
        MatchingParameters,
        ModelArchitecture,
        ModelConfiguration,
        ModelError,
        MolecularOrbitals,
        MSPredictor,
        NMRPredictor,
        NormalizationMethod,
        PredictionError,
        PredictionResult,
        ProcessingError,
        ProcessingMethod,
        ProcessingParameters,
        ProcessingResult,
        QMLSurrogateModel,
        QuantumCalculation,
        QuantumChemistryError,
        QuantumMethod,
        ScoringFunction,
        SemiEmpiricalMethod,
        SpectraAligner,
        SpectralMatch,
        SpectralMatcher,
        SpectralPeak,
        SpectralPrediction,
        SpectralProperties,
        SpectralTransformer,
        SpectraPreprocessor,
        SpectroscopyType,
        SpectrumFormat,
        StructureCandidate,
        StructureGenerationError,
        StructureGenerationMethod,
        StructureProposer,
        TrainingData,
        TrainingError,
        UVVisPredictor,
        calculate_electronic_structure,
        compute_spectral_properties,
        evaluate_model_accuracy,
        load_fid_data,
        load_jcamp_dx,
        match_experimental_spectrum,
        optimize_geometry_for_spectroscopy,
        predict_spectrum_batch,
        process_spectrum_batch,
        propose_structures_from_spectrum,
        train_spectral_model,
        validate_structural_proposals,
    )

    _SPECTROSCOPY_AVAILABLE = True
except ImportError:
    _SPECTROSCOPY_AVAILABLE = False

# Self-Assembly modules (with optional imports)
try:
    from .self_assembly import (  # Simulation engine; Simulation data structures; Simulation enums; Simulation utilities; Characterization engines; Characterization data structures; Characterization enums; Characterization utilities; Inverse design engines; Inverse design data structures; Inverse design enums; Inverse design utilities; Multi-scale integration; Multi-scale data structures; Multi-scale enums; Multi-scale utilities; Interactive visualization engines; Visualization configuration classes; Visualization data structures; Visualization enums; Visualization utility functions; Real-time dashboard system; Self-assembly exceptions
        AlertLevel,
        AnimationSpeed,
        AtomisticModel,
        BaseOptimizer,
        BayesianOptimizer,
        Bond,
        BoundaryCondition,
        CharacterizationError,
        CharacterizationResult,
        ClusterAnalyzer,
        ClusteringError,
        ClusteringMethod,
        CoarseGrainedSimulationEngine,
        CoarseGrainedSimulationError,
        ColorScheme,
        CouplingConfiguration,
        CouplingMethod,
        DashboardConfig,
        DashboardError,
        DataProcessingError,
        DataTransferEngine,
        DataTransferError,
        DataTransferMethod,
        DesignParameter,
        DesignSpace,
        DesignSpaceError,
        DesignTarget,
        EmergentPropertyInverseDesigner,
        EvaluationError,
        ExportError,
        ExportFormat,
        GeneticAlgorithmOptimizer,
        GPUMemoryError,
        HydrogelCharacteristics,
        IntegrationError,
        IntegratorType,
        InteractionParameters,
        InteractionType,
        InteractiveSelfAssemblyVisualizer,
        InverseDesignError,
        LiquidCrystalAnalyzer,
        LiquidCrystalCharacteristics,
        MDIntegrator,
        MesoscaleModel,
        MetricStatus,
        MicelleAnalyzer,
        MicelleCharacteristics,
        MolecularModel,
        MorphologyAnalysisError,
        MorphologyType,
        MultiScaleIntegrationError,
        MultiScaleIntegrator,
        MultiScaleSystem,
        ObjectiveFunction,
        OptimizationError,
        OptimizationMethod,
        OptimizationObjective,
        OptimizationResult,
        ParameterSweepData,
        Particle,
        ParticleInteractionCalculator,
        ParticleType,
        PerformanceMetric,
        PerformanceMetrics,
        PhaseAnalysisResult,
        PhaseTransitionAnalyzer,
        PhaseTransitionEvent,
        PhaseType,
        PropertyTrackingDashboard,
        RealTimeDashboardConfig,
        RealTimeDataManager,
        RealTimeMetric,
        RealTimeSelfAssemblyDashboard,
        RenderingError,
        ScaleCouplingError,
        ScaleData,
        ScaleParameters,
        ScaleType,
        SelfAssemblyCharacterizer,
        SimulationAlert,
        SimulationBox,
        SimulationMethod,
        SimulationParameters,
        SimulationState,
        StructureClassifier,
        StructureMetrics,
        StructureType,
        TimeSeriesData,
        TrajectoryFrame,
        TransferResult,
        ValidationError,
        ValidationLevel,
        VesicleAnalyzer,
        VesicleCharacteristics,
        VisualizationConfig,
        VisualizationError,
        VisualizationFrame,
        VisualizationMode,
        analyze_micelle_formation,
        calculate_phase_diagram,
        characterize_trajectory,
        create_amphiphilic_system,
        create_comparative_visualization,
        create_demo_realtime_dashboard,
        create_demo_visualization,
        create_hierarchical_system,
        design_micelle_forming_surfactant,
        export_dashboard_data,
        multi_objective_surfactant_design,
        optimize_environmental_conditions,
        run_dpd_simulation,
        run_multiscale_workflow,
        validate_multiscale_consistency,
    )

    _SELF_ASSEMBLY_AVAILABLE = True
except ImportError:
    _SELF_ASSEMBLY_AVAILABLE = False

# Integration modules (with optional imports)
try:
    from .integrations.sklearn_integration import (
        ChemiAccelEstimator,
        ChemiAccelTransformer,
        create_molecular_pipeline,
    )

    _SKLEARN_INTEGRATION_AVAILABLE = True
except ImportError:
    _SKLEARN_INTEGRATION_AVAILABLE = False

try:
    from .integrations.jupyter_integration import enable_rich_display, quick_demo

    _JUPYTER_INTEGRATION_AVAILABLE = True
except ImportError:
    _JUPYTER_INTEGRATION_AVAILABLE = False

try:
    from .integrations.dask_integration import (
        DaskChemiAccelCalculator,
        distributed_calculate_descriptors,
        process_large_dataset,
        setup_dask_cluster,
    )

    _DASK_INTEGRATION_AVAILABLE = True
except ImportError:
    _DASK_INTEGRATION_AVAILABLE = False

try:
    from .integrations.hpc_integration import (
        DaskClusterIntegration,
        HPCCalculator,
        MPIIntegration,
        RayClusterIntegration,
        SLURMJobManager,
        create_hpc_calculator,
        detect_hpc_environment,
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

# Add sustainability exports if available
if _SUSTAINABILITY_AVAILABLE:
    __all__.extend(
        [
            # Degradation modeling
            "MultiScaleDegradationPredictor",
            "AtomisticMDModel",
            "SurrogateAIModel",
            "create_standard_material",
            "create_accelerated_aging_conditions",
            "run_degradation_prediction",
            # Recyclability assessment
            "ComprehensiveRecyclabilityAssessor",
            "MechanicalRecyclingPredictor",
            "ChemicalRecyclingPredictor",
            "BiologicalRecyclingPredictor",
            "create_standard_plastic_composition",
            "assess_material_recyclability",
            # LCA integration
            "MatrixLCACalculator",
            "GPUAcceleratedLCACalculator",
            "LCAComparator",
            "create_polymer_manufacturing_process",
            "calculate_material_lca",
            # Inverse design for sustainable materials
            "SustainableInverseDesigner",
            "SustainabilityPropertyPredictor",
            "OptimizationObjective",
            "OptimizationAlgorithm",
            "MaterialParameter",
            "SustainabilityTarget",
            "DesignResult",
            "OptimizationConfig",
            "create_polymer_design_parameters",
            "create_sustainability_objectives",
            "design_sustainable_material",
            # Interactive lifecycle simulation and visualization
            "LifecycleSimulator",
            "LifecycleVisualizer",
            "MaterialComparator",
            "SustainabilityDashboard",
            "LifecyclePhase",
            "ComparisonMetric",
            "LifecycleEvent",
            "LifecycleTrajectory",
            "SimulationConfig",
            "simulate_material_lifecycle",
            "create_sustainability_dashboard",
            "analyze_material_trade_offs",
        ]
    )

# Add self-assembly exports if available
if _SELF_ASSEMBLY_AVAILABLE:
    __all__.extend(
        [
            # Simulation engine
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
            "AlertLevel",
            "MetricStatus",
            # Visualization utility functions
            "create_comparative_visualization",
            "export_dashboard_data",
            "create_demo_visualization",
            # Real-time dashboard system
            "RealTimeSelfAssemblyDashboard",
            "RealTimeDataManager",
            "RealTimeDashboardConfig",
            "RealTimeMetric",
            "SimulationAlert",
            "PerformanceMetrics",
            "create_demo_realtime_dashboard",
            # Self-assembly exceptions
            "CoarseGrainedSimulationError",
            "GPUMemoryError",
            "IntegrationError",
            "CharacterizationError",
            "ClusteringError",
            "MorphologyAnalysisError",
            "InverseDesignError",
            "OptimizationError",
            "DesignSpaceError",
            "EvaluationError",
            "MultiScaleIntegrationError",
            "ScaleCouplingError",
            "DataTransferError",
            "ValidationError",
            "VisualizationError",
            "RenderingError",
            "DashboardError",
            "ExportError",
            "DataProcessingError",
        ]
    )

# Add integration exports if available
if _SKLEARN_INTEGRATION_AVAILABLE:
    __all__.extend(
        ["ChemiAccelTransformer", "ChemiAccelEstimator", "create_molecular_pipeline"]
    )

if _JUPYTER_INTEGRATION_AVAILABLE:
    __all__.extend(["enable_rich_display", "quick_demo"])

if _DASK_INTEGRATION_AVAILABLE:
    __all__.extend(
        [
            "DaskChemiAccelCalculator",
            "distributed_calculate_descriptors",
            "setup_dask_cluster",
            "process_large_dataset",
        ]
    )

if _HPC_INTEGRATION_AVAILABLE:
    __all__.extend(
        [
            "HPCCalculator",
            "MPIIntegration",
            "SLURMJobManager",
            "DaskClusterIntegration",
            "RayClusterIntegration",
            "detect_hpc_environment",
            "create_hpc_calculator",
        ]
    )

if _MATERIALS_AVAILABLE:
    __all__.extend(
        [
            "MaterialsCalculator",
            "CIFParser",
            "POSCARParser",
            "CrystalDescriptorCalculator",
            "PeriodicDescriptorCalculator",
            "MaterialsValidator",
        ]
    )

# Add CSP exports if available
if _CSP_AVAILABLE:
    __all__.extend(
        [
            # Enums
            "CrystalSystemType",
            "CSPMethod",
            "PackingMotifType",
            "LatticeEnergyMethod",
            "SymmetryHandling",
            # Data structures
            "CrystalPredictionConfig",
            "UnitCellParameters",
            "AtomicPosition",
            "CrystalStructure",
            "CSPPrediction",
            # Neural network classes
            "CrystalGraphConv",
            "PackingMotifGNN",
            "UnitCellGenerator",
            "LatticeEnergyPredictor",
            # Main classes
            "GPUAcceleratedCSPEngine",
            # Utility functions
            "predict_crystal_structure",
            "create_csp_config",
            "analyze_polymorphs",
            # Exceptions
            "CrystalStructurePredictionError",
            "LatticeEnergyError",
            "StructureGenerationError",
            "SymmetryError",
            "ConvergenceError",
        ]
    )

# Add Intermolecular Interactions exports if available
if _INTERACTIONS_AVAILABLE:
    __all__.extend(
        [
            # Enums
            "InteractionType",
            "HydrogenBondType",
            "PiPiStackingType",
            "CrystalPackingObjective",
            # Data structures
            "AtomicCoordinates",
            "MolecularFragment",
            "InteractionPair",
            "HydrogenBond",
            "PiPiInteraction",
            "VanDerWaalsInteraction",
            "InteractionConfiguration",
            "CrystalPackingResult",
            # Main classes
            "HydrogenBondCalculator",
            "PiPiStackingCalculator",
            "VanDerWaalsCalculator",
            "CrystalPackingOptimizer",
            "GPUAcceleratedInteractionEngine",
            # Utility functions
            "create_molecular_fragment_from_smiles",
            "create_interaction_config",
            "analyze_crystal_packing",
            "optimize_intermolecular_interactions",
            # Exceptions
            "IntermolecularInteractionError",
            "HydrogenBondCalculationError",
            "PiPiStackingCalculationError",
            "VanDerWaalsCalculationError",
            "CrystalPackingOptimizationError",
            "InteractionEngineError",
        ]
    )

# Add Polymorph Inverse Design exports if available
if _POLYMORPH_DESIGN_AVAILABLE:
    __all__.extend(
        [
            # Enums
            "PolymorphType",
            "ModificationStrategy",
            "DesignObjective",
            # Data structures
            "PolymorphTarget",
            "MolecularModification",
            "DesignCandidate",
            "PolymorphDesignConfig",
            # Main classes
            "PolymorphPredictor",
            "MolecularModifier",
            "PolymorphInverseDesigner",
            # Utility functions
            "create_polymorph_target",
            "create_design_config",
            "design_for_polymorph",
            "analyze_polymorph_landscape",
            "predict_polymorph_stability",
            # Exceptions
            "PolymorphInverseDesignError",
            "PolymorphPredictionError",
            "MolecularModificationError",
            "DesignOptimizationError",
            "ConfigurationError",
        ]
    )

# Add Solid-State Properties exports if available
if _SOLID_STATE_PROPERTIES_AVAILABLE:
    __all__.extend(
        [
            # Enums
            "PropertyType",
            "ElasticProperty",
            "ThermalProperty",
            "SolubilityType",
            # Data structures
            "CrystalStructureData",
            "PropertyPredictionConfig",
            "PropertyResult",
            "ElasticProperties",
            "ThermalProperties",
            "SolubilityResult",
            # Main classes
            "DensityCalculator",
            "ElasticModuliPredictor",
            "ThermalPropertyPredictor",
            "SolubilityPredictor",
            "SolidStatePropertyEngine",
            "ElasticPropertyNet",
            # Utility functions
            "create_property_config",
            "create_crystal_structure",
            "predict_solid_state_properties",
            "calculate_structure_density",
            "predict_elastic_moduli",
            "predict_thermal_properties",
            "predict_solubility",
            # Exceptions
            "SolidStatePropertyError",
            "PropertyPredictionError",
            "DensityCalculationError",
            "ElasticModuliCalculationError",
            "ThermalPropertyError",
            "SolubilityPredictionError",
        ]
    )

# Add Experimental Validation exports if available
if _EXPERIMENTAL_VALIDATION_AVAILABLE:
    __all__.extend(
        [
            # Enums
            "ExperimentalTechnique",
            "ValidationMetric",
            "RefinementStrategy",
            # Data structures
            "ExperimentalData",
            "PowderXRDPattern",
            "ValidationResult",
            "ComparisonResult",
            "RefinementConfig",
            "FeedbackData",
            # Main classes
            "XRDPatternSimulator",
            "ExperimentalDataValidator",
            "ExperimentalComparator",
            "ModelRefinementEngine",
            "FeedbackLoop",
            "ExperimentalValidationEngine",
            # Utility functions
            "create_experimental_data",
            "create_refinement_config",
            "simulate_xrd_pattern",
            "validate_experimental_quality",
            "compare_with_experiment",
            "setup_feedback_loop",
            # Exceptions
            "ExperimentalValidationError",
            "XRDComparisonError",
            "DataValidationError",
            "ModelRefinementError",
            "FeedbackLoopError",
        ]
    )

# Biological interactions modules (with optional imports)
try:
    from .biological_interactions import (  # Multi-target prediction; Pathway modeling; Multi-omics integration; Systems toxicology; Inverse design; Visualization; Multi-target interaction prediction; Multi-omics data integration; Systems toxicology and off-target effects; Systems inverse design; Systems visualization and explanation
        ActivationState,
        ADRPredictionError,
        ADRPredictor,
        AdverseDrugReaction,
        AdverseDrugReactionType,
        AdverseEffectsConstraints,
        BaseDataIngester,
        BiologicalPathway,
        BiologicalPathwayModeler,
        BiologicalTarget,
        BiologicalTargetEncoder,
        CellularPhenotype,
        CellularPhenotypePredictor,
        ColorScheme,
        CompoundTargetInteraction,
        CompoundTargetTransformer,
        ConfidenceLevel,
        ConstraintViolationError,
        DashboardCreationError,
        DatabaseIntegrationError,
        DatabaseLoadingError,
        DatabaseSource,
        DataFormat,
        DataIngestionError,
        DesignConstraints,
        DesignStrategy,
        ExplanationGenerationError,
        ExplanationResult,
        GenomicsDataIngester,
        GPUBioInteractionPredictor,
        IntegrationMethod,
        IntegrationResult,
        InteractionPrediction,
        InteractionScorer,
        InteractionType,
        KEGGIntegration,
        MetabolomicsDataIngester,
        ModelArchitecture,
        ModelLoadingError,
        MolecularConstraint,
        MolecularGenerationError,
        MultiOmicsIntegrationError,
        MultiOmicsIntegrationPipeline,
        MultiOmicsIntegrator,
        MultiTargetInteractionPredictor,
        MultiTargetPredictionError,
        MultiTaskGNN,
        NetworkAnalysisError,
        NetworkEdge,
        NetworkLayout,
        NetworkNode,
        NetworkPropagationEngine,
        NetworkType,
        NetworkVisualization,
        OffTargetBinding,
        OffTargetPredictionError,
        OffTargetPredictor,
        OmicsDataset,
        OmicsDataType,
        OmicsQualityController,
        OptimizationError,
        OptimizationObjective,
        OptimizationResult,
        PathwayActivation,
        PathwayActivationPredictor,
        PathwayDatabase,
        PathwayInhibitionAnalyzer,
        PathwayModelingError,
        PathwayNetworkIntegrator,
        PathwayType,
        PhenotypeCategory,
        PhenotypeChange,
        PolypharmacologyProfile,
        PPINetworkLoader,
        PredictionConfidence,
        PredictionRefinement,
        PredictionRefinementEngine,
        PredictionRefinementError,
        PredictionRefinementStrategy,
        PredictionResult,
        ProteomicsDataIngester,
        QualityControlError,
        QualityControlMethod,
        QualityControlReport,
        ReactomeIntegration,
        RiskLevel,
        SystemsInverseDesigner,
        SystemsInverseDesignError,
        SystemsToxicologyEngine,
        SystemsToxicologyProfile,
        SystemsVisualizationError,
        SystemsVisualizer,
        TargetEncodingError,
        TargetPanel,
        TargetType,
        ToxicityEndpoint,
        ToxicityMechanism,
        ToxicityPrediction,
        ToxicityPredictor,
        ToxicologyDatabase,
        ToxicologyPredictionError,
        VisualizationConfig,
        VisualizationCreationError,
        VisualizationType,
        analyze_phenotype_changes,
        batch_toxicology_screening,
        create_design_constraints,
        create_interactive_systems_dashboard,
        create_polypharmacology_profile,
        create_safety_constraints,
        create_sample_omics_data,
        create_sample_toxicology_database,
        create_systems_visualization,
        design_polypharmacology_molecule,
        evaluate_prediction_accuracy,
        explain_biological_prediction,
        generate_toxicology_report,
        integrate_omics_data_batch,
        integrate_ppi_networks,
        load_omics_dataset,
        load_pathway_database,
        load_target_panel,
        optimize_for_safety,
        predict_compound_safety_profile,
        predict_interactions_batch,
        predict_pathway_activation,
    )

    _BIOLOGICAL_INTERACTIONS_AVAILABLE = True
except ImportError:
    _BIOLOGICAL_INTERACTIONS_AVAILABLE = False

# Add spectroscopy exports if available
if _SPECTROSCOPY_AVAILABLE:
    __all__.extend(
        [
            # Predictive models
            "GPUSpectroscopyPredictor",
            "QMLSurrogateModel",
            "NMRPredictor",
            "IRPredictor",
            "MSPredictor",
            "UVVisPredictor",
            "SpectralTransformer",
            "ConvolutionalSpectralNet",
            "GraphSpectralNet",
            # Model configuration
            "ModelConfiguration",
            "TrainingData",
            "SpectralPrediction",
            "PredictionResult",
            # Enums
            "SpectroscopyType",
            "ModelArchitecture",
            "QuantumMethod",
            # Experimental processing
            "ExperimentalSpectraProcessor",
            "SpectraPreprocessor",
            "SpectraAligner",
            "ExperimentalSpectrum",
            "ProcessingParameters",
            "ProcessingResult",
            "SpectralPeak",
            # Processing enums
            "SpectrumFormat",
            "ProcessingMethod",
            "BaselineCorrectionMethod",
            "NormalizationMethod",
            # Inverse spectroscopy
            "InverseSpectroscopyEngine",
            "StructureProposer",
            "SpectralMatcher",
            "StructureCandidate",
            "SpectralMatch",
            "InverseResult",
            "MatchingParameters",
            # Inverse enums
            "MatchingMethod",
            "ScoringFunction",
            "StructureGenerationMethod",
            # Quantum methods
            "FastQuantumChemistry",
            "ApproximateMethodEngine",
            "DFTSurrogate",
            "SemiEmpiricalMethod",
            "DensityFunctionalApproximation",
            "BasisSetOptimizer",
            # Quantum data structures
            "QuantumCalculation",
            "ElectronicStructure",
            "MolecularOrbitals",
            "SpectralProperties",
            # Quantum enums
            "ApproximationLevel",
            "BasisSetType",
            "FunctionalType",
            # Utility functions
            "predict_spectrum_batch",
            "train_spectral_model",
            "evaluate_model_accuracy",
            "load_jcamp_dx",
            "load_fid_data",
            "process_spectrum_batch",
            "propose_structures_from_spectrum",
            "match_experimental_spectrum",
            "validate_structural_proposals",
            "calculate_electronic_structure",
            "compute_spectral_properties",
            "optimize_geometry_for_spectroscopy",
            # Spectral comparison
            "SpectralComparator",
            "SpectralAligner",
            "SpectralVisualizer",
            "SpectralComparison",
            "DiscrepancyRegion",
            "VisualizationConfig",
            "ComparisonMethod",
            "DiscrepancyType",
            "VisualizationType",
            "compare_spectra_batch",
            "generate_comparison_report",
            # Knowledge graph integration
            "KnowledgeGraphValidator",
            "NoveltyDetector",
            "SpectralFingerprintGenerator",
            "SpectralFingerprint",
            "KnowledgeGraphEntry",
            "ValidationReport",
            "NoveltyDetectionResult",
            "ValidationResult",
            "NoveltyType",
            "ConfidenceLevel",
            "validate_structure_batch",
            "detect_novelty_batch",
            # Exceptions
            "PredictionError",
            "ModelError",
            "TrainingError",
            "ProcessingError",
            "FormatError",
            "DataQualityError",
            "InverseError",
            "MatchingError",
            "StructureGenerationError",
            "QuantumChemistryError",
            "ConvergenceError",
            "BasisSetError",
            "SpectralComparisonError",
            "AlignmentError",
            "VisualizationError",
            "KnowledgeGraphIntegrationError",
            "ValidationError",
            "NoveltyDetectionError",
        ]
    )

# Retrosynthesis modules (with optional imports)
try:
    from .retrosynthesis import (  # GPU-accelerated generative retrosynthesis; Multi-step route enumeration; Forward prediction; Multi-objective optimization; Interactive visualization; Generative Engine; Route Enumeration; Forward Prediction; Multi-Objective Optimization; Route Visualization
        BuildingBlockDatabase,
        BuildingBlockQuery,
        BuildingBlockSource,
        ByproductPrediction,
        ColorScheme,
        ComplexityCalculator,
        ConfidenceLevel,
        ConvergenceError,
        DatabaseQueryError,
        DisconnectionError,
        DisconnectionPredictor,
        DisconnectionSite,
        DisconnectionStrategy,
        DominanceAnalyzer,
        DominanceRelation,
        ForwardPredictionError,
        ForwardPredictionResult,
        ForwardReactionPredictor,
        GraphRetroModel,
        InteractionError,
        InteractionMode,
        InteractiveVisualizer,
        KineticsEstimator,
        KineticsModel,
        KineticsModelingError,
        KineticsParameters,
        LayoutAlgorithm,
        LayoutCalculator,
        LayoutError,
        ModelLoadingError,
        ModelPrediction,
        ModelType,
        MolecularFeaturizer,
        MolecularRepresentation,
        MoleculeNode,
        MoleculeRenderer,
        MultiObjectiveConfig,
        MultiObjectiveOptimizationError,
        MultiObjectiveOptimizer,
        ObjectiveEvaluationError,
        ObjectiveEvaluator,
        OptimizationAlgorithm,
        OptimizationError,
        OptimizationObjective,
        OptimizationObjectiveSpec,
        OptimizationResult,
        OptimizationSolution,
        ParetoFront,
        ParetoFrontError,
        ParetoFrontNavigator,
        ParetoRanking,
        PredictionError,
        PruningCriteria,
        ReactionCondition,
        ReactionConditions,
        ReactionEdge,
        ReactionTemplate,
        ReactionTemplateExtractor,
        ReactionType,
        RenderingError,
        RetrosynthesisConfig,
        RetrosynthesisEngine,
        RetrosynthesisError,
        RetrosynthesisResult,
        RouteEnumerationConfig,
        RouteEnumerationError,
        RouteEnumerator,
        RouteEvaluationResult,
        RouteMetrics,
        RouteOptimizer,
        RouteStrategy,
        SelectivityPrediction,
        SelectivityPredictionError,
        SelectivityPredictor,
        SelectivityType,
        SynthesisComplexity,
        SynthesisNode,
        SynthesisRoute,
        SynthesisTree,
        SynthesisTreeBuilder,
        TransformerRetroModel,
        TreeConstructionError,
        VisualizationConfig,
        VisualizationData,
        VisualizationElement,
        VisualizationError,
        VisualizationType,
        YieldCategory,
        YieldPredictionError,
        YieldPredictor,
        build_synthesis_tree,
        calculate_complexity,
        create_multi_objective_config,
        create_pareto_front_plot,
        create_reaction_conditions,
        create_retro_config,
        create_route_config,
        create_visualization_config,
        decode_molecule,
        encode_molecule,
        enumerate_routes,
        estimate_reaction_kinetics,
        evaluate_synthesis_route_forward,
        extract_disconnection_sites,
        find_pareto_optimal_routes,
        launch_streamlit_dashboard,
        optimize_route,
        optimize_synthesis_routes,
        predict_reaction_yield,
        predict_retrosynthesis,
        query_building_blocks,
        visualize_synthesis_route_simple,
    )

    _RETROSYNTHESIS_AVAILABLE = True
except ImportError:
    _RETROSYNTHESIS_AVAILABLE = False

# Add biological interactions exports if available
if _BIOLOGICAL_INTERACTIONS_AVAILABLE:
    __all__.extend(
        [
            # Multi-target interaction prediction
            "MultiTargetInteractionPredictor",
            "GPUBioInteractionPredictor",
            "MultiTaskGNN",
            "CompoundTargetTransformer",
            "BiologicalTargetEncoder",
            "InteractionScorer",
            "BiologicalTarget",
            "CompoundTargetInteraction",
            "InteractionPrediction",
            "PredictionResult",
            "TargetPanel",
            "TargetType",
            "InteractionType",
            "ConfidenceLevel",
            "ModelArchitecture",
            "predict_interactions_batch",
            "load_target_panel",
            "evaluate_prediction_accuracy",
            "MultiTargetPredictionError",
            "ModelLoadingError",
            "TargetEncodingError",
            # Pathway modeling
            "BiologicalPathwayModeler",
            "PathwayNetworkIntegrator",
            "CellularPhenotypePredictor",
            "PathwayActivationPredictor",
            "PathwayInhibitionAnalyzer",
            "NetworkPropagationEngine",
            "BiologicalPathway",
            "PathwayDatabase",
            "NetworkNode",
            "NetworkEdge",
            "PathwayActivation",
            "CellularPhenotype",
            "PhenotypeChange",
            "KEGGIntegration",
            "ReactomeIntegration",
            "PPINetworkLoader",
            "PathwayType",
            "DatabaseSource",
            "ActivationState",
            "PhenotypeCategory",
            "NetworkType",
            "load_pathway_database",
            "predict_pathway_activation",
            "analyze_phenotype_changes",
            "integrate_ppi_networks",
            "PathwayModelingError",
            "DatabaseIntegrationError",
            "NetworkAnalysisError",
            # Multi-omics data integration
            "OmicsDataset",
            "IntegrationResult",
            "PredictionRefinement",
            "QualityControlReport",
            "OmicsDataType",
            "DataFormat",
            "IntegrationMethod",
            "QualityControlMethod",
            "PredictionRefinementStrategy",
            "BaseDataIngester",
            "GenomicsDataIngester",
            "ProteomicsDataIngester",
            "MetabolomicsDataIngester",
            "OmicsQualityController",
            "MultiOmicsIntegrator",
            "PredictionRefinementEngine",
            "MultiOmicsIntegrationPipeline",
            "load_omics_dataset",
            "create_sample_omics_data",
            "integrate_omics_data_batch",
            "MultiOmicsIntegrationError",
            "DataIngestionError",
            "QualityControlError",
            "PredictionRefinementError",
            # Systems toxicology and off-target effects
            "OffTargetBinding",
            "ToxicityPrediction",
            "AdverseDrugReaction",
            "SystemsToxicologyProfile",
            "ToxicologyDatabase",
            "ToxicityEndpoint",
            "AdverseDrugReactionType",
            "ToxicityMechanism",
            "RiskLevel",
            "PredictionConfidence",
            "OffTargetPredictor",
            "ToxicityPredictor",
            "ADRPredictor",
            "SystemsToxicologyEngine",
            "predict_compound_safety_profile",
            "batch_toxicology_screening",
            "generate_toxicology_report",
            "create_sample_toxicology_database",
            "ToxicologyPredictionError",
            "OffTargetPredictionError",
            "ADRPredictionError",
            "DatabaseLoadingError",
            # Systems inverse design
            "OptimizationObjective",
            "DesignStrategy",
            "MolecularConstraint",
            "PolypharmacologyProfile",
            "AdverseEffectsConstraints",
            "DesignConstraints",
            "OptimizationResult",
            "SystemsInverseDesigner",
            "design_polypharmacology_molecule",
            "optimize_for_safety",
            "create_polypharmacology_profile",
            "create_safety_constraints",
            "create_design_constraints",
            "SystemsInverseDesignError",
            "OptimizationError",
            "ConstraintViolationError",
            "MolecularGenerationError",
            # Systems visualization and explanation
            "VisualizationType",
            "NetworkLayout",
            "ColorScheme",
            "VisualizationConfig",
            "ExplanationResult",
            "NetworkVisualization",
            "SystemsVisualizer",
            "create_systems_visualization",
            "explain_biological_prediction",
            "create_interactive_systems_dashboard",
            "SystemsVisualizationError",
            "VisualizationCreationError",
            "ExplanationGenerationError",
            "DashboardCreationError",
        ]
    )

# Add retrosynthesis exports if available
if _RETROSYNTHESIS_AVAILABLE:
    __all__.extend(
        [
            # Generative Engine - Enums
            "ModelType",
            "DisconnectionStrategy",
            "ConfidenceLevel",
            "ReactionType",
            # Generative Engine - Data structures
            "RetrosynthesisConfig",
            "MolecularRepresentation",
            "RetrosynthesisResult",
            "DisconnectionSite",
            "ReactionTemplate",
            "ModelPrediction",
            # Generative Engine - Main classes
            "TransformerRetroModel",
            "GraphRetroModel",
            "RetrosynthesisEngine",
            "DisconnectionPredictor",
            "ReactionTemplateExtractor",
            # Generative Engine - Utility functions
            "create_retro_config",
            "encode_molecule",
            "decode_molecule",
            "predict_retrosynthesis",
            "extract_disconnection_sites",
            # Generative Engine - Exceptions
            "RetrosynthesisError",
            "ModelLoadingError",
            "PredictionError",
            "DisconnectionError",
            # Route Enumeration - Enums
            "RouteStrategy",
            "PruningCriteria",
            "BuildingBlockSource",
            "SynthesisComplexity",
            # Route Enumeration - Data structures
            "SynthesisNode",
            "SynthesisTree",
            "RouteEnumerationConfig",
            "SynthesisRoute",
            "RouteMetrics",
            "BuildingBlockQuery",
            # Route Enumeration - Main classes
            "RouteEnumerator",
            "SynthesisTreeBuilder",
            "BuildingBlockDatabase",
            "RouteOptimizer",
            "ComplexityCalculator",
            # Route Enumeration - Utility functions
            "create_route_config",
            "build_synthesis_tree",
            "enumerate_routes",
            "optimize_route",
            "calculate_complexity",
            "query_building_blocks",
            # Route Enumeration - Exceptions
            "RouteEnumerationError",
            "TreeConstructionError",
            "DatabaseQueryError",
            "OptimizationError",
            # Forward Prediction - Enums
            "ReactionCondition",
            "YieldCategory",
            "SelectivityType",
            "KineticsModel",
            # Forward Prediction - Data structures
            "ReactionConditions",
            "KineticsParameters",
            "SelectivityPrediction",
            "ByproductPrediction",
            "ForwardPredictionResult",
            "RouteEvaluationResult",
            # Forward Prediction - Main classes
            "MolecularFeaturizer",
            "YieldPredictor",
            "SelectivityPredictor",
            "KineticsEstimator",
            "ForwardReactionPredictor",
            # Forward Prediction - Utility functions
            "create_reaction_conditions",
            "predict_reaction_yield",
            "estimate_reaction_kinetics",
            "evaluate_synthesis_route_forward",
            # Forward Prediction - Exceptions
            "ForwardPredictionError",
            "KineticsModelingError",
            "YieldPredictionError",
            "SelectivityPredictionError",
            # Multi-Objective Optimization - Enums
            "OptimizationObjective",
            "OptimizationAlgorithm",
            "DominanceRelation",
            "ParetoRanking",
            # Multi-Objective Optimization - Data structures
            "OptimizationObjectiveSpec",
            "MultiObjectiveConfig",
            "OptimizationSolution",
            "ParetoFront",
            "OptimizationResult",
            # Multi-Objective Optimization - Main classes
            "ObjectiveEvaluator",
            "DominanceAnalyzer",
            "ParetoFrontNavigator",
            "MultiObjectiveOptimizer",
            # Multi-Objective Optimization - Utility functions
            "create_multi_objective_config",
            "optimize_synthesis_routes",
            "find_pareto_optimal_routes",
            # Multi-Objective Optimization - Exceptions
            "MultiObjectiveOptimizationError",
            "ParetoFrontError",
            "ObjectiveEvaluationError",
            "ConvergenceError",
            # Route Visualization - Enums
            "VisualizationType",
            "ColorScheme",
            "LayoutAlgorithm",
            "InteractionMode",
            # Route Visualization - Data structures
            "VisualizationConfig",
            "VisualizationElement",
            "MoleculeNode",
            "ReactionEdge",
            "VisualizationData",
            # Route Visualization - Main classes
            "MoleculeRenderer",
            "LayoutCalculator",
            "InteractiveVisualizer",
            # Route Visualization - Utility functions
            "create_visualization_config",
            "visualize_synthesis_route_simple",
            "create_pareto_front_plot",
            "launch_streamlit_dashboard",
            # Route Visualization - Exceptions
            "VisualizationError",
            "RenderingError",
            "LayoutError",
            "InteractionError",
        ]
    )

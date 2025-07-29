"""
Materials Science Module for ChemiAccel.

This module provides comprehensive support for crystalline and periodic materials,
including CIF/POSCAR file parsing, geometric and topological descriptor calculations,
GPU-accelerated crystal structure prediction, intermolecular interaction modeling,
polymorph inverse design, and integration with materials science workflows.
"""

from .crystal_descriptors import CrystalDescriptorCalculator
from .crystal_parser import CIFParser, POSCARParser
from .materials_calculator import MaterialsCalculator
from .periodic_descriptors import PeriodicDescriptorCalculator
from .validation import MaterialsValidator

# Crystal Structure Prediction (with optional imports)
try:
    from .crystal_structure_prediction import (  # GPU-accelerated CSP; AI-driven structure prediction; Polymorphic landscape exploration; Lattice energy minimization; Packing motif prediction; Search algorithms; Neural networks; Enums; Data structures; Main classes; Utility functions; Exceptions; Enums; Data structures; Neural network classes (require PyTorch); Main classes; Utility functions; Exceptions
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
    from .intermolecular_interactions import (  # GPU-accelerated intermolecular interactions; Hydrogen bonding networks; π-π stacking; van der Waals; Crystal packing optimization; Enums; Data structures; Main classes; Utility functions; Exceptions; Enums; Data structures; Main classes; Utility functions; Exceptions
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
        OptimizationAlgorithm,
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
    from .polymorph_inverse_design import (  # AI-driven polymorph control; Molecular modification strategies; Inverse design optimization; Polymorph prediction; Enums; Data structures; Main classes; Utility functions; Exceptions; Enums; Data structures; Main classes; Utility functions; Exceptions
        ConfigurationError,
        DesignCandidate,
        DesignObjective,
        DesignOptimizationError,
        DesignResult,
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
    from .solid_state_properties import (  # GPU-accelerated property prediction; Density calculations; Elastic moduli; Thermal properties; Solubility prediction; ML-based models; Enums; Data structures; Main classes; Utility functions; Exceptions; Enums; Data structures; Main classes; Utility functions; Exceptions
        CrystalStructureData,
        DataQuality,
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
    from .experimental_validation import (  # Experimental validation and refinement; XRD pattern simulation; Data comparison; Feedback loops; Model refinement; Enums; Data structures; Main classes; Utility functions; Exceptions; Enums; Data structures; Main classes; Utility functions; Exceptions
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

__all__ = [
    "CIFParser",
    "POSCARParser",
    "CrystalDescriptorCalculator",
    "PeriodicDescriptorCalculator",
    "MaterialsCalculator",
    "MaterialsValidator",
]

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
            "OptimizationAlgorithm",
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
            "DesignResult",
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
            "DataQuality",
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

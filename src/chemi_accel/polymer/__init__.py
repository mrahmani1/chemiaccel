"""
Polymer processing module for ChemiAccel.

This module provides specialized functionality for analyzing and processing
polymer structures, including monomers, dimers, oligomers, and polymer chains.
"""

from .monomer import (
    MonomerAnalyzer,
    ReactivityAnalyzer,
    PolymerizationPotentialAnalyzer,
    StereochemistryHandler,
    RingOpeningAnalyzer
)

from .oligomer import (
    OligomerProcessor,
    ChainGrowthSimulator,
    ConformationalFlexibilityAnalyzer,
    EndGroupHandler,
    SequenceAnalyzer
)

from .conformers import (
    PolymerConformerGenerator,
    ChainFlexibilityAnalyzer,
    IntramolecularInteractionAnalyzer,
    ImplicitSolventHandler
)

from .repeating import (
    RepeatingUnitIdentifier,
    PolymerFragmentAnalyzer,
    BranchedPolymerHandler
)

from .predictive import (
    PolymerPropertyEngine,
    PolymerDescriptorCalculator,
    PolymerModelTrainer,
    predict_polymer_properties,
    predict_glass_transition_temp,
    predict_youngs_modulus
)

from .copolymer import (
    CopolymerType,
    MixingRule,
    PhaseState,
    MonomerComponent,
    CopolymerSystem,
    CopolymerProperties,
    MicrophaseInfo,
    BlendMiscibility,
    CopolymerPropertyPredictor,
    MicrophaseAnalyzer,
    BlendMiscibilityAnalyzer,
    CopolymerBlendModeler
)

from .inverse_design import (
    InversePolymerDesigner,
    PolymerConstraintConfig,
    PolymerDesignObjective,
    MonomerGenerationResult,
    BioDerivedScorer,
    PolymerConstraintHandler,
    MonomerGenerator,
    create_high_performance_polymer_objectives,
    create_biodegradable_polymer_objectives,
    create_bio_derived_constraints,
    create_high_performance_constraints
)

from .thermoset import (
    CuringMechanism,
    NetworkState,
    ReactiveGroup,
    CrosslinkBond,
    NetworkTopology,
    CuringKinetics,
    CuringResult,
    ThermosetCuringSimulator,
    NetworkTopologyAnalyzer,
    GelPointDetector,
    validate_against_experimental_data
)

__all__ = [
    # Monomer analysis
    "MonomerAnalyzer",
    "ReactivityAnalyzer", 
    "PolymerizationPotentialAnalyzer",
    "StereochemistryHandler",
    "RingOpeningAnalyzer",
    
    # Oligomer processing
    "OligomerProcessor",
    "ChainGrowthSimulator",
    "ConformationalFlexibilityAnalyzer", 
    "EndGroupHandler",
    "SequenceAnalyzer",
    
    # Conformer generation
    "PolymerConformerGenerator",
    "ChainFlexibilityAnalyzer",
    "IntramolecularInteractionAnalyzer",
    "ImplicitSolventHandler",
    
    # Repeating units
    "RepeatingUnitIdentifier",
    "PolymerFragmentAnalyzer",
    "BranchedPolymerHandler",
    
    # Predictive property engine
    "PolymerPropertyEngine",
    "PolymerDescriptorCalculator", 
    "PolymerModelTrainer",
    "predict_polymer_properties",
    "predict_glass_transition_temp",
    "predict_youngs_modulus",
    
    # Copolymer and blend modeling
    "CopolymerType",
    "MixingRule",
    "PhaseState",
    "MonomerComponent",
    "CopolymerSystem",
    "CopolymerProperties",
    "MicrophaseInfo",
    "BlendMiscibility",
    "CopolymerPropertyPredictor",
    "MicrophaseAnalyzer",
    "BlendMiscibilityAnalyzer",
    "CopolymerBlendModeler",
    
    # Inverse polymer design
    "InversePolymerDesigner",
    "PolymerConstraintConfig",
    "PolymerDesignObjective",
    "MonomerGenerationResult",
    "BioDerivedScorer",
    "PolymerConstraintHandler",
    "MonomerGenerator",
    "create_high_performance_polymer_objectives",
    "create_biodegradable_polymer_objectives",
    "create_bio_derived_constraints",
    "create_high_performance_constraints",
    
    # Thermoset curing simulation
    "CuringMechanism",
    "NetworkState",
    "ReactiveGroup",
    "CrosslinkBond",
    "NetworkTopology",
    "CuringKinetics",
    "CuringResult",
    "ThermosetCuringSimulator",
    "NetworkTopologyAnalyzer",
    "GelPointDetector",
    "validate_against_experimental_data"
]
"""Molecular descriptor calculation modules."""

from .basic import (
    BasicDescriptorCalculator,
    calculate_basic_properties,
    get_lipinski_descriptors
)
from .constitutional import (
    ConstitutionalDescriptorCalculator,
    calculate_constitutional_descriptors,
    get_constitutional_descriptor_info
)
from .graph import (
    MolecularGraphBuilder,
    GraphDescriptorCalculator,
    build_molecular_graphs,
    calculate_graph_descriptors
)
from .fingerprints import (
    FingerprintGenerator,
    FingerprintSimilarity,
    generate_morgan_fingerprints,
    generate_maccs_fingerprints,
    calculate_fingerprint_similarities
)
from .topological import (
    TopologicalDescriptorCalculator,
    TopologicalCUDAKernels,
    calculate_topological_descriptors,
    get_topological_descriptor_info
)
from .advanced import (
    SubstructureSearcher,
    MaximumCommonSubstructure,
    PharmacophoreAnalyzer,
    ReactionEnumerator,
    DrugLikenessFilters,
    search_substructures,
    calculate_mcs,
    analyze_pharmacophores,
    filter_drug_like_compounds
)
from .geometrical_3d import (
    ConformerGenerator,
    Shape3DDescriptors,
    ElectrostaticDescriptors,
    Geometrical3DCalculator,
    generate_conformers,
    calculate_3d_descriptors,
    calculate_shape_descriptors,
    calculate_electrostatic_descriptors
)
from .gnn_features import (
    MolecularGraphEmbedding,
    CPUGraphEmbedding,
    CUDAGraphEmbedding,
    JAXGraphEmbedding,
    AdvancedGraphFeatures,
    compute_molecular_graph_embeddings,
    compute_advanced_graph_features,
    compute_graph_similarity_matrix,
    benchmark_graph_features_performance
)
from .admet import (
    ADMETDescriptorCalculator
)
from .drug_discovery import (
    DrugDiscoveryDescriptorCalculator
)
from .pharmacophore_screening import (
    PharmacophoreScreeningEngine,
    create_screening_engine,
    quick_screen
)

__all__ = [
    # Basic descriptors
    "BasicDescriptorCalculator",
    "calculate_basic_properties",
    "get_lipinski_descriptors",
    
    # Constitutional descriptors
    "ConstitutionalDescriptorCalculator",
    "calculate_constitutional_descriptors",
    "get_constitutional_descriptor_info",
    
    # Graph descriptors
    "MolecularGraphBuilder",
    "GraphDescriptorCalculator", 
    "build_molecular_graphs",
    "calculate_graph_descriptors",
    
    # Topological descriptors
    "TopologicalDescriptorCalculator",
    "TopologicalCUDAKernels",
    "calculate_topological_descriptors",
    "get_topological_descriptor_info",
    
    # Fingerprints
    "FingerprintGenerator",
    "FingerprintSimilarity",
    "generate_morgan_fingerprints",
    "generate_maccs_fingerprints",
    "calculate_fingerprint_similarities",
    
    # Advanced features
    "SubstructureSearcher",
    "MaximumCommonSubstructure",
    "PharmacophoreAnalyzer",
    "ReactionEnumerator",
    "DrugLikenessFilters",
    "search_substructures",
    "calculate_mcs",
    "analyze_pharmacophores",
    "filter_drug_like_compounds",
    
    # 3D and geometrical descriptors
    "ConformerGenerator",
    "Shape3DDescriptors",
    "ElectrostaticDescriptors",
    "Geometrical3DCalculator",
    "generate_conformers",
    "calculate_3d_descriptors",
    "calculate_shape_descriptors",
    "calculate_electrostatic_descriptors",
    
    # GNN features
    "MolecularGraphEmbedding",
    "CPUGraphEmbedding",
    "CUDAGraphEmbedding", 
    "JAXGraphEmbedding",
    "AdvancedGraphFeatures",
    "compute_molecular_graph_embeddings",
    "compute_advanced_graph_features",
    "compute_graph_similarity_matrix",
    "benchmark_graph_features_performance",
    
    # ADMET descriptors
    "ADMETDescriptorCalculator",
    
    # Drug discovery descriptors
    "DrugDiscoveryDescriptorCalculator",
    
    # GPU-accelerated pharmacophore screening
    "PharmacophoreScreeningEngine",
    "create_screening_engine",
    "quick_screen"
]
"""Molecular descriptor calculation modules."""

from .admet import ADMETDescriptorCalculator
from .advanced import (
    DrugLikenessFilters,
    MaximumCommonSubstructure,
    PharmacophoreAnalyzer,
    ReactionEnumerator,
    SubstructureSearcher,
    analyze_pharmacophores,
    calculate_mcs,
    filter_drug_like_compounds,
    search_substructures,
)
from .basic import (
    BasicDescriptorCalculator,
    calculate_basic_properties,
    get_lipinski_descriptors,
)
from .constitutional import (
    ConstitutionalDescriptorCalculator,
    calculate_constitutional_descriptors,
    get_constitutional_descriptor_info,
)
from .drug_discovery import DrugDiscoveryDescriptorCalculator
from .fingerprints import (
    FingerprintGenerator,
    FingerprintSimilarity,
    calculate_fingerprint_similarities,
    generate_maccs_fingerprints,
    generate_morgan_fingerprints,
)
from .geometrical_3d import (
    ConformerGenerator,
    ElectrostaticDescriptors,
    Geometrical3DCalculator,
    Shape3DDescriptors,
    calculate_3d_descriptors,
    calculate_electrostatic_descriptors,
    calculate_shape_descriptors,
    generate_conformers,
)
from .gnn_features import (
    AdvancedGraphFeatures,
    CPUGraphEmbedding,
    CUDAGraphEmbedding,
    JAXGraphEmbedding,
    MolecularGraphEmbedding,
    benchmark_graph_features_performance,
    compute_advanced_graph_features,
    compute_graph_similarity_matrix,
    compute_molecular_graph_embeddings,
)
from .graph import (
    GraphDescriptorCalculator,
    MolecularGraphBuilder,
    build_molecular_graphs,
    calculate_graph_descriptors,
)
from .pharmacophore_screening import (
    PharmacophoreScreeningEngine,
    create_screening_engine,
    quick_screen,
)
from .topological import (
    TopologicalCUDAKernels,
    TopologicalDescriptorCalculator,
    calculate_topological_descriptors,
    get_topological_descriptor_info,
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
    "quick_screen",
]

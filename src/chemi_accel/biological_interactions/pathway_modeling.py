"""
Biological Pathway and Network Modeling Module.

This module provides comprehensive tools for integrating molecular interactions with
biological pathway databases, modeling pathway activation/inhibition, and predicting
cellular phenotype changes from molecular input.
"""

import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.preprocessing import StandardScaler

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration will be limited.")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn(
        "Pandas not available. Some data processing features will be limited."
    )


class PathwayType(Enum):
    """Types of biological pathways."""

    METABOLIC = "metabolic"
    SIGNALING = "signaling"
    REGULATORY = "regulatory"
    TRANSPORT = "transport"
    IMMUNE_RESPONSE = "immune_response"
    APOPTOSIS = "apoptosis"
    CELL_CYCLE = "cell_cycle"
    DNA_REPAIR = "dna_repair"
    PROTEIN_SYNTHESIS = "protein_synthesis"
    OXIDATIVE_STRESS = "oxidative_stress"


class DatabaseSource(Enum):
    """Biological pathway database sources."""

    KEGG = "kegg"
    REACTOME = "reactome"
    WIKIPATHWAYS = "wikipathways"
    BIOCARTA = "biocarta"
    PID = "pid"
    PANTHER = "panther"
    CUSTOM = "custom"


class ActivationState(Enum):
    """Pathway activation states."""

    HIGHLY_ACTIVATED = "highly_activated"  # >2 fold change
    ACTIVATED = "activated"  # 1.5-2 fold change
    NORMAL = "normal"  # 0.67-1.5 fold change
    SUPPRESSED = "suppressed"  # 0.5-0.67 fold change
    HIGHLY_SUPPRESSED = "highly_suppressed"  # <0.5 fold change


class PhenotypeCategory(Enum):
    """Categories of cellular phenotypes."""

    PROLIFERATION = "proliferation"
    APOPTOSIS = "apoptosis"
    DIFFERENTIATION = "differentiation"
    MIGRATION = "migration"
    METABOLISM = "metabolism"
    STRESS_RESPONSE = "stress_response"
    IMMUNE_RESPONSE = "immune_response"
    SENESCENCE = "senescence"
    AUTOPHAGY = "autophagy"
    DNA_DAMAGE = "dna_damage"


class NetworkType(Enum):
    """Types of biological networks."""

    PROTEIN_PROTEIN_INTERACTION = "ppi"
    METABOLIC_NETWORK = "metabolic"
    GENE_REGULATORY = "gene_regulatory"
    SIGNALING_NETWORK = "signaling"
    CO_EXPRESSION = "co_expression"
    PHYSICAL_INTERACTION = "physical"


@dataclass
class NetworkNode:
    """Represents a node in a biological network."""

    node_id: str
    name: str
    node_type: str  # gene, protein, metabolite, etc.
    organism: str = "Homo sapiens"
    external_ids: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)
    expression_level: Optional[float] = None
    fold_change: Optional[float] = None
    p_value: Optional[float] = None

    def __post_init__(self):
        """Initialize computed properties."""
        if self.external_ids is None:
            self.external_ids = {}
        if self.annotations is None:
            self.annotations = {}


@dataclass
class NetworkEdge:
    """Represents an edge in a biological network."""

    source_id: str
    target_id: str
    interaction_type: str
    confidence_score: float = 1.0
    evidence_types: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    weight: float = 1.0

    def __post_init__(self):
        """Initialize lists."""
        if self.evidence_types is None:
            self.evidence_types = []
        if self.publications is None:
            self.publications = []
        if self.databases is None:
            self.databases = []


@dataclass
class BiologicalPathway:
    """Represents a biological pathway."""

    pathway_id: str
    name: str
    pathway_type: PathwayType
    database_source: DatabaseSource
    organism: str = "Homo sapiens"
    description: Optional[str] = None
    genes: List[str] = field(default_factory=list)
    proteins: List[str] = field(default_factory=list)
    metabolites: List[str] = field(default_factory=list)
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    hierarchical_level: int = 1
    parent_pathways: List[str] = field(default_factory=list)
    child_pathways: List[str] = field(default_factory=list)
    cross_references: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize lists and dictionaries."""
        if self.genes is None:
            self.genes = []
        if self.proteins is None:
            self.proteins = []
        if self.metabolites is None:
            self.metabolites = []
        if self.reactions is None:
            self.reactions = []
        if self.parent_pathways is None:
            self.parent_pathways = []
        if self.child_pathways is None:
            self.child_pathways = []
        if self.cross_references is None:
            self.cross_references = {}

    def get_all_components(self) -> Set[str]:
        """Get all pathway components (genes, proteins, metabolites)."""
        return set(self.genes + self.proteins + self.metabolites)


@dataclass
class PathwayActivation:
    """Represents pathway activation results."""

    pathway_id: str
    pathway_name: str
    activation_state: ActivationState
    activation_score: float
    p_value: float
    adjusted_p_value: float
    activated_genes: List[str] = field(default_factory=list)
    suppressed_genes: List[str] = field(default_factory=list)
    leading_edge_genes: List[str] = field(default_factory=list)
    enrichment_score: float = 0.0

    def __post_init__(self):
        """Initialize gene lists."""
        if self.activated_genes is None:
            self.activated_genes = []
        if self.suppressed_genes is None:
            self.suppressed_genes = []
        if self.leading_edge_genes is None:
            self.leading_edge_genes = []


@dataclass
class CellularPhenotype:
    """Represents a cellular phenotype."""

    phenotype_id: str
    name: str
    category: PhenotypeCategory
    description: Optional[str] = None
    associated_pathways: List[str] = field(default_factory=list)
    biomarkers: List[str] = field(default_factory=list)
    confidence_score: float = 1.0

    def __post_init__(self):
        """Initialize lists."""
        if self.associated_pathways is None:
            self.associated_pathways = []
        if self.biomarkers is None:
            self.biomarkers = []


@dataclass
class PhenotypeChange:
    """Represents predicted phenotype changes."""

    compound_smiles: str
    phenotype: CellularPhenotype
    predicted_change: float  # Fold change or effect size
    confidence: float
    contributing_pathways: List[PathwayActivation] = field(default_factory=list)
    time_course: Optional[Dict[str, float]] = None  # Time -> predicted change

    def __post_init__(self):
        """Initialize lists and dictionaries."""
        if self.contributing_pathways is None:
            self.contributing_pathways = []
        if self.time_course is None:
            self.time_course = {}


@dataclass
class PathwayDatabase:
    """Container for pathway database information."""

    database_id: str
    name: str
    version: str
    source: DatabaseSource
    pathways: Dict[str, BiologicalPathway] = field(default_factory=dict)
    organism: str = "Homo sapiens"
    last_updated: Optional[str] = None

    def __post_init__(self):
        """Initialize pathway dictionary."""
        if self.pathways is None:
            self.pathways = {}

    def get_pathway(self, pathway_id: str) -> Optional[BiologicalPathway]:
        """Get pathway by ID."""
        return self.pathways.get(pathway_id)

    def get_pathways_by_type(
        self, pathway_type: PathwayType
    ) -> List[BiologicalPathway]:
        """Get pathways filtered by type."""
        return [p for p in self.pathways.values() if p.pathway_type == pathway_type]


class KEGGIntegration:
    """Integration with KEGG pathway database."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.base_url = "https://rest.kegg.jp"

    def load_pathways(self, organism: str = "hsa") -> PathwayDatabase:
        """Load KEGG pathways for specified organism."""
        # Placeholder implementation - would make API calls to KEGG
        pathways = {}

        # Create sample KEGG pathways
        sample_pathways = [
            ("hsa00010", "Glycolysis / Gluconeogenesis", PathwayType.METABOLIC),
            ("hsa00020", "Citrate cycle (TCA cycle)", PathwayType.METABOLIC),
            ("hsa04010", "MAPK signaling pathway", PathwayType.SIGNALING),
            ("hsa04110", "Cell cycle", PathwayType.REGULATORY),
            ("hsa04210", "Apoptosis", PathwayType.APOPTOSIS),
            (
                "hsa04060",
                "Cytokine-cytokine receptor interaction",
                PathwayType.IMMUNE_RESPONSE,
            ),
        ]

        for pathway_id, name, pathway_type in sample_pathways:
            pathway = BiologicalPathway(
                pathway_id=pathway_id,
                name=name,
                pathway_type=pathway_type,
                database_source=DatabaseSource.KEGG,
                organism="Homo sapiens",
                genes=self._get_sample_genes(pathway_id),
                proteins=self._get_sample_proteins(pathway_id),
            )
            pathways[pathway_id] = pathway

        return PathwayDatabase(
            database_id="kegg_hsa",
            name="KEGG Human Pathways",
            version="2024.1",
            source=DatabaseSource.KEGG,
            pathways=pathways,
            organism="Homo sapiens",
        )

    def _get_sample_genes(self, pathway_id: str) -> List[str]:
        """Get sample genes for pathway (placeholder)."""
        # In practice, this would query KEGG API
        base_genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        # Modify based on pathway ID to create variety
        hash_val = hash(pathway_id) % 100
        return [f"{gene}_{hash_val}" for gene in base_genes[: 3 + hash_val % 3]]

    def _get_sample_proteins(self, pathway_id: str) -> List[str]:
        """Get sample proteins for pathway (placeholder)."""
        base_proteins = ["PROT1", "PROT2", "PROT3", "PROT4", "PROT5"]
        hash_val = hash(pathway_id) % 100
        return [f"{prot}_{hash_val}" for prot in base_proteins[: 2 + hash_val % 4]]


class ReactomeIntegration:
    """Integration with Reactome pathway database."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.base_url = "https://reactome.org"

    def load_pathways(self, organism: str = "9606") -> PathwayDatabase:
        """Load Reactome pathways for specified organism."""
        # Placeholder implementation - would make API calls to Reactome
        pathways = {}

        # Create sample Reactome pathways
        sample_pathways = [
            ("R-HSA-1640170", "Cell Cycle", PathwayType.CELL_CYCLE),
            ("R-HSA-162582", "Signal Transduction", PathwayType.SIGNALING),
            ("R-HSA-73857", "RNA Polymerase II Transcription", PathwayType.REGULATORY),
            ("R-HSA-109581", "Apoptosis", PathwayType.APOPTOSIS),
            ("R-HSA-168256", "Immune System", PathwayType.IMMUNE_RESPONSE),
            ("R-HSA-1430728", "Metabolism", PathwayType.METABOLIC),
        ]

        for pathway_id, name, pathway_type in sample_pathways:
            pathway = BiologicalPathway(
                pathway_id=pathway_id,
                name=name,
                pathway_type=pathway_type,
                database_source=DatabaseSource.REACTOME,
                organism="Homo sapiens",
                genes=self._get_sample_genes(pathway_id),
                proteins=self._get_sample_proteins(pathway_id),
            )
            pathways[pathway_id] = pathway

        return PathwayDatabase(
            database_id="reactome_hsa",
            name="Reactome Human Pathways",
            version="2024.1",
            source=DatabaseSource.REACTOME,
            pathways=pathways,
            organism="Homo sapiens",
        )

    def _get_sample_genes(self, pathway_id: str) -> List[str]:
        """Get sample genes for pathway (placeholder)."""
        base_genes = ["REAC_GENE1", "REAC_GENE2", "REAC_GENE3", "REAC_GENE4"]
        hash_val = abs(hash(pathway_id)) % 100
        return [f"{gene}_{hash_val}" for gene in base_genes[: 2 + hash_val % 3]]

    def _get_sample_proteins(self, pathway_id: str) -> List[str]:
        """Get sample proteins for pathway (placeholder)."""
        base_proteins = ["REAC_PROT1", "REAC_PROT2", "REAC_PROT3"]
        hash_val = abs(hash(pathway_id)) % 100
        return [f"{prot}_{hash_val}" for prot in base_proteins[: 1 + hash_val % 3]]


class PPINetworkLoader:
    """Loader for protein-protein interaction networks."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

    def load_string_network(
        self, organism: str = "9606", confidence_threshold: float = 0.4
    ) -> nx.Graph:
        """Load STRING protein-protein interaction network."""
        # Placeholder implementation - would download from STRING database
        G = nx.Graph()

        # Create sample PPI network
        proteins = [f"PROT_{i:04d}" for i in range(100)]

        # Add nodes
        for protein in proteins:
            G.add_node(protein, type="protein")

        # Add edges with confidence scores
        np.random.seed(42)  # For reproducible results
        for i in range(len(proteins)):
            for j in range(i + 1, len(proteins)):
                if np.random.random() < 0.1:  # 10% edge density
                    confidence = np.random.uniform(confidence_threshold, 1.0)
                    G.add_edge(
                        proteins[i],
                        proteins[j],
                        confidence=confidence,
                        interaction_type="binding",
                    )

        return G

    def load_biogrid_network(self, organism: str = "9606") -> nx.Graph:
        """Load BioGRID protein-protein interaction network."""
        # Placeholder implementation
        G = nx.Graph()

        # Create sample BioGRID network
        proteins = [f"BIOGRID_{i:04d}" for i in range(80)]

        # Add nodes
        for protein in proteins:
            G.add_node(protein, type="protein", database="BioGRID")

        # Add edges
        np.random.seed(123)
        for i in range(len(proteins)):
            for j in range(i + 1, len(proteins)):
                if np.random.random() < 0.08:  # 8% edge density
                    G.add_edge(
                        proteins[i],
                        proteins[j],
                        interaction_type="physical",
                        evidence="experimental",
                    )

        return G


class NetworkPropagationEngine:
    """Engine for network propagation algorithms."""

    def __init__(self, use_gpu: bool = True, alpha: float = 0.7):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.alpha = alpha  # Restart probability for random walk

    def random_walk_with_restart(
        self,
        network: nx.Graph,
        seed_nodes: List[str],
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> Dict[str, float]:
        """Perform random walk with restart on network."""
        # Convert network to adjacency matrix
        nodes = list(network.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)

        # Create adjacency matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for edge in network.edges(data=True):
            i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
            weight = edge[2].get("weight", 1.0)
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight  # Undirected

        # Normalize to transition matrix
        row_sums = adj_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = adj_matrix / row_sums[:, np.newaxis]

        # Initialize seed vector
        seed_vector = np.zeros(n_nodes)
        for seed_node in seed_nodes:
            if seed_node in node_to_idx:
                seed_vector[node_to_idx[seed_node]] = 1.0
        seed_vector = (
            seed_vector / seed_vector.sum() if seed_vector.sum() > 0 else seed_vector
        )

        # Convert to GPU if available
        if self.use_gpu and CUPY_AVAILABLE:
            transition_matrix = cp.asarray(transition_matrix)
            seed_vector = cp.asarray(seed_vector)
            current_vector = cp.copy(seed_vector)
        else:
            current_vector = seed_vector.copy()

        # Iterative random walk
        for iteration in range(max_iterations):
            if self.use_gpu and CUPY_AVAILABLE:
                next_vector = (1 - self.alpha) * cp.dot(
                    transition_matrix.T, current_vector
                ) + self.alpha * seed_vector
                change = cp.abs(next_vector - current_vector).max()
            else:
                next_vector = (1 - self.alpha) * np.dot(
                    transition_matrix.T, current_vector
                ) + self.alpha * seed_vector
                change = np.abs(next_vector - current_vector).max()

            current_vector = next_vector

            if change < tolerance:
                break

        # Convert back to CPU if needed
        if self.use_gpu and CUPY_AVAILABLE:
            current_vector = cp.asnumpy(current_vector)

        # Return as dictionary
        return {node: float(score) for node, score in zip(nodes, current_vector)}

    def heat_diffusion(
        self,
        network: nx.Graph,
        initial_heat: Dict[str, float],
        diffusion_rate: float = 0.1,
        max_iterations: int = 100,
    ) -> Dict[str, float]:
        """Perform heat diffusion on network."""
        nodes = list(network.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)

        # Create Laplacian matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for edge in network.edges(data=True):
            i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
            weight = edge[2].get("weight", 1.0)
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight

        degree_matrix = np.diag(adj_matrix.sum(axis=1))
        laplacian = degree_matrix - adj_matrix

        # Initialize heat vector
        heat_vector = np.zeros(n_nodes)
        for node, heat in initial_heat.items():
            if node in node_to_idx:
                heat_vector[node_to_idx[node]] = heat

        # Heat diffusion
        for _ in range(max_iterations):
            heat_vector = heat_vector - diffusion_rate * np.dot(laplacian, heat_vector)

        return {node: float(heat) for node, heat in zip(nodes, heat_vector)}


class PathwayActivationPredictor:
    """Predicts pathway activation from molecular data."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.scaler = StandardScaler()

    def predict_activation_gsea(
        self,
        gene_expression: Dict[str, float],
        pathway_database: PathwayDatabase,
        min_pathway_size: int = 5,
        max_pathway_size: int = 500,
    ) -> List[PathwayActivation]:
        """Predict pathway activation using Gene Set Enrichment Analysis."""
        results = []

        # Sort genes by expression change
        sorted_genes = sorted(gene_expression.items(), key=lambda x: x[1], reverse=True)
        gene_ranks = {gene: rank for rank, (gene, _) in enumerate(sorted_genes)}

        for pathway in pathway_database.pathways.values():
            # Filter pathway genes present in expression data
            pathway_genes = [g for g in pathway.genes if g in gene_expression]

            if (
                len(pathway_genes) < min_pathway_size
                or len(pathway_genes) > max_pathway_size
            ):
                continue

            # Calculate enrichment score
            enrichment_score = self._calculate_enrichment_score(
                gene_ranks, pathway_genes, len(sorted_genes)
            )

            # Calculate p-value (simplified)
            p_value = self._calculate_gsea_pvalue(enrichment_score, len(pathway_genes))

            # Determine activation state
            activation_state = self._determine_activation_state(enrichment_score)

            # Get leading edge genes
            leading_edge = self._get_leading_edge_genes(
                gene_expression, pathway_genes, enrichment_score > 0
            )

            activation = PathwayActivation(
                pathway_id=pathway.pathway_id,
                pathway_name=pathway.name,
                activation_state=activation_state,
                activation_score=enrichment_score,
                p_value=p_value,
                adjusted_p_value=p_value * len(pathway_database.pathways),  # Bonferroni
                activated_genes=[
                    g for g in pathway_genes if gene_expression.get(g, 0) > 0
                ],
                suppressed_genes=[
                    g for g in pathway_genes if gene_expression.get(g, 0) < 0
                ],
                leading_edge_genes=leading_edge,
                enrichment_score=enrichment_score,
            )

            results.append(activation)

        return sorted(results, key=lambda x: x.p_value)

    def _calculate_enrichment_score(
        self, gene_ranks: Dict[str, int], pathway_genes: List[str], total_genes: int
    ) -> float:
        """Calculate GSEA enrichment score."""
        # Simplified GSEA calculation
        pathway_ranks = [gene_ranks.get(gene, total_genes) for gene in pathway_genes]
        pathway_ranks.sort()

        # Calculate running sum
        n_genes = len(pathway_genes)
        running_sum = 0
        max_deviation = 0

        for i, rank in enumerate(pathway_ranks):
            # Weight by expression rank
            weight = (total_genes - rank) / total_genes
            running_sum += weight - (1.0 / n_genes)

            if abs(running_sum) > abs(max_deviation):
                max_deviation = running_sum

        return max_deviation

    def _calculate_gsea_pvalue(
        self, enrichment_score: float, pathway_size: int
    ) -> float:
        """Calculate GSEA p-value (simplified)."""
        # Simplified p-value calculation
        # In practice, this would use permutation testing
        normalized_score = abs(enrichment_score) * np.sqrt(pathway_size)
        p_value = 1.0 / (1.0 + np.exp(normalized_score * 2))
        return max(1e-10, min(1.0, p_value))

    def _determine_activation_state(self, enrichment_score: float) -> ActivationState:
        """Determine activation state from enrichment score."""
        if enrichment_score > 0.5:
            return ActivationState.HIGHLY_ACTIVATED
        elif enrichment_score > 0.2:
            return ActivationState.ACTIVATED
        elif enrichment_score < -0.5:
            return ActivationState.HIGHLY_SUPPRESSED
        elif enrichment_score < -0.2:
            return ActivationState.SUPPRESSED
        else:
            return ActivationState.NORMAL

    def _get_leading_edge_genes(
        self,
        gene_expression: Dict[str, float],
        pathway_genes: List[str],
        is_upregulated: bool,
    ) -> List[str]:
        """Get leading edge genes contributing to enrichment."""
        # Sort pathway genes by expression
        sorted_pathway_genes = sorted(
            [(g, gene_expression.get(g, 0)) for g in pathway_genes],
            key=lambda x: x[1],
            reverse=is_upregulated,
        )

        # Return top contributing genes
        n_leading = max(1, len(pathway_genes) // 4)
        return [gene for gene, _ in sorted_pathway_genes[:n_leading]]


class PathwayInhibitionAnalyzer:
    """Analyzes pathway inhibition and suppression."""

    def __init__(self):
        pass

    def analyze_inhibition_cascade(
        self, network: nx.Graph, inhibited_nodes: List[str], propagation_depth: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """Analyze cascade effects of pathway inhibition."""
        results = {}

        for depth in range(1, propagation_depth + 1):
            affected_nodes = self._propagate_inhibition(network, inhibited_nodes, depth)
            results[f"depth_{depth}"] = affected_nodes

        return results

    def _propagate_inhibition(
        self, network: nx.Graph, inhibited_nodes: List[str], depth: int
    ) -> Dict[str, float]:
        """Propagate inhibition effects through network."""
        affected = {}
        current_layer = set(inhibited_nodes)

        for _ in range(depth):
            next_layer = set()

            for node in current_layer:
                if node in network:
                    for neighbor in network.neighbors(node):
                        # Calculate inhibition strength based on edge properties
                        edge_data = network.get_edge_data(node, neighbor, {})
                        inhibition_strength = (
                            edge_data.get("weight", 0.5) * 0.8
                        )  # Decay

                        if neighbor not in affected:
                            affected[neighbor] = 0
                        affected[neighbor] = max(
                            affected[neighbor], inhibition_strength
                        )
                        next_layer.add(neighbor)

            current_layer = next_layer

        return affected


class CellularPhenotypePredictor:
    """Predicts cellular phenotype changes from molecular input."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # Load predefined phenotypes
        self.phenotypes = self._load_cellular_phenotypes()

        # Initialize neural network for phenotype prediction
        self.phenotype_model = self._build_phenotype_model()

    def _load_cellular_phenotypes(self) -> Dict[str, CellularPhenotype]:
        """Load predefined cellular phenotypes."""
        phenotypes = {}

        # Define sample phenotypes
        phenotype_data = [
            (
                "PROLIF_001",
                "Cell Proliferation",
                PhenotypeCategory.PROLIFERATION,
                ["hsa04110", "hsa04010"],
                ["PCNA", "Ki67", "CYCLIN_D1"],
            ),
            (
                "APOPT_001",
                "Apoptosis Induction",
                PhenotypeCategory.APOPTOSIS,
                ["hsa04210"],
                ["CASP3", "CASP9", "BAX", "BCL2"],
            ),
            (
                "MIGRAT_001",
                "Cell Migration",
                PhenotypeCategory.MIGRATION,
                ["hsa04810"],
                ["ACTB", "VIM", "CDH1"],
            ),
            (
                "METAB_001",
                "Metabolic Reprogramming",
                PhenotypeCategory.METABOLISM,
                ["hsa00010", "hsa00020"],
                ["HK2", "PKM", "LDHA"],
            ),
            (
                "STRESS_001",
                "Oxidative Stress Response",
                PhenotypeCategory.STRESS_RESPONSE,
                ["hsa04141"],
                ["SOD1", "CAT", "GPX1"],
            ),
        ]

        for phenotype_id, name, category, pathways, biomarkers in phenotype_data:
            phenotype = CellularPhenotype(
                phenotype_id=phenotype_id,
                name=name,
                category=category,
                associated_pathways=pathways,
                biomarkers=biomarkers,
            )
            phenotypes[phenotype_id] = phenotype

        return phenotypes

    def _build_phenotype_model(self) -> nn.Module:
        """Build neural network for phenotype prediction."""
        model = nn.Sequential(
            nn.Linear(1000, 512),  # Input: pathway activation scores
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.phenotypes)),  # Output: phenotype probabilities
            nn.Sigmoid(),
        )
        return model.to(self.device)

    def predict_phenotype_changes(
        self,
        compound_smiles: str,
        pathway_activations: List[PathwayActivation],
        time_points: Optional[List[float]] = None,
    ) -> List[PhenotypeChange]:
        """Predict cellular phenotype changes from pathway activations."""
        if time_points is None:
            time_points = [24.0]  # 24 hours default

        results = []

        # Prepare input features
        activation_features = self._prepare_activation_features(pathway_activations)

        # Predict phenotype probabilities
        with torch.no_grad():
            features_tensor = torch.tensor(
                activation_features, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            phenotype_probs = self.phenotype_model(features_tensor).cpu().numpy()[0]

        # Create phenotype change predictions
        for i, (phenotype_id, phenotype) in enumerate(self.phenotypes.items()):
            probability = phenotype_probs[i]

            if probability > 0.1:  # Only include significant predictions
                # Calculate predicted change magnitude
                predicted_change = self._calculate_change_magnitude(
                    pathway_activations, phenotype
                )

                # Generate time course
                time_course = {}
                for time_point in time_points:
                    time_course[str(time_point)] = predicted_change * self._time_decay(
                        time_point
                    )

                # Get contributing pathways
                contributing_pathways = self._get_contributing_pathways(
                    pathway_activations, phenotype
                )

                phenotype_change = PhenotypeChange(
                    compound_smiles=compound_smiles,
                    phenotype=phenotype,
                    predicted_change=predicted_change,
                    confidence=probability,
                    contributing_pathways=contributing_pathways,
                    time_course=time_course,
                )

                results.append(phenotype_change)

        return sorted(results, key=lambda x: x.confidence, reverse=True)

    def _prepare_activation_features(
        self, pathway_activations: List[PathwayActivation]
    ) -> np.ndarray:
        """Prepare features from pathway activations."""
        # Create feature vector (simplified)
        features = np.zeros(1000)  # Fixed size feature vector

        for i, activation in enumerate(pathway_activations[:1000]):
            features[i] = activation.activation_score

        return features

    def _calculate_change_magnitude(
        self, pathway_activations: List[PathwayActivation], phenotype: CellularPhenotype
    ) -> float:
        """Calculate predicted change magnitude for phenotype."""
        relevant_activations = [
            act
            for act in pathway_activations
            if act.pathway_id in phenotype.associated_pathways
        ]

        if not relevant_activations:
            return 0.0

        # Average activation score of relevant pathways
        avg_activation = np.mean([act.activation_score for act in relevant_activations])

        # Convert to fold change
        fold_change = (
            2**avg_activation if avg_activation > 0 else -(2 ** abs(avg_activation))
        )

        return fold_change

    def _time_decay(self, time_hours: float) -> float:
        """Calculate time-dependent decay factor."""
        # Simple exponential decay
        half_life = 48.0  # hours
        return np.exp(-np.log(2) * time_hours / half_life)

    def _get_contributing_pathways(
        self, pathway_activations: List[PathwayActivation], phenotype: CellularPhenotype
    ) -> List[PathwayActivation]:
        """Get pathways contributing to phenotype prediction."""
        return [
            act
            for act in pathway_activations
            if act.pathway_id in phenotype.associated_pathways
            and abs(act.activation_score) > 0.1
        ]


class BiologicalPathwayModeler:
    """Main class for biological pathway modeling."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.pathway_databases = {}
        self.ppi_networks = {}

        # Initialize components
        self.kegg_integration = KEGGIntegration()
        self.reactome_integration = ReactomeIntegration()
        self.ppi_loader = PPINetworkLoader(use_gpu=use_gpu)
        self.propagation_engine = NetworkPropagationEngine(use_gpu=use_gpu)
        self.activation_predictor = PathwayActivationPredictor(use_gpu=use_gpu)
        self.inhibition_analyzer = PathwayInhibitionAnalyzer()
        self.phenotype_predictor = CellularPhenotypePredictor(use_gpu=use_gpu)

    def load_pathway_databases(
        self, sources: List[DatabaseSource]
    ) -> Dict[str, PathwayDatabase]:
        """Load pathway databases from specified sources."""
        databases = {}

        for source in sources:
            if source == DatabaseSource.KEGG:
                db = self.kegg_integration.load_pathways()
                databases[source.value] = db
                self.pathway_databases[source.value] = db
            elif source == DatabaseSource.REACTOME:
                db = self.reactome_integration.load_pathways()
                databases[source.value] = db
                self.pathway_databases[source.value] = db

        return databases

    def load_interaction_networks(
        self, network_types: List[str]
    ) -> Dict[str, nx.Graph]:
        """Load protein-protein interaction networks."""
        networks = {}

        for network_type in network_types:
            if network_type == "string":
                network = self.ppi_loader.load_string_network()
                networks[network_type] = network
                self.ppi_networks[network_type] = network
            elif network_type == "biogrid":
                network = self.ppi_loader.load_biogrid_network()
                networks[network_type] = network
                self.ppi_networks[network_type] = network

        return networks

    def analyze_compound_effects(
        self,
        compound_smiles: str,
        target_interactions: Dict[str, float],  # target_id -> interaction_strength
        gene_expression_changes: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Analyze compound effects on biological pathways and phenotypes."""
        results = {}

        # Predict pathway activations
        if gene_expression_changes:
            pathway_activations = {}
            for db_name, db in self.pathway_databases.items():
                activations = self.activation_predictor.predict_activation_gsea(
                    gene_expression_changes, db
                )
                pathway_activations[db_name] = activations
            results["pathway_activations"] = pathway_activations

        # Analyze network propagation effects
        if self.ppi_networks and target_interactions:
            propagation_results = {}
            for network_name, network in self.ppi_networks.items():
                # Find targets in network
                network_targets = [
                    t for t in target_interactions.keys() if t in network
                ]

                if network_targets:
                    propagation_scores = (
                        self.propagation_engine.random_walk_with_restart(
                            network, network_targets
                        )
                    )
                    propagation_results[network_name] = propagation_scores

            results["network_propagation"] = propagation_results

        # Predict phenotype changes
        if "pathway_activations" in results:
            phenotype_changes = {}
            for db_name, activations in results["pathway_activations"].items():
                changes = self.phenotype_predictor.predict_phenotype_changes(
                    compound_smiles, activations
                )
                phenotype_changes[db_name] = changes
            results["phenotype_changes"] = phenotype_changes

        return results


class PathwayNetworkIntegrator:
    """Integrates pathway information with network topology."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

    def integrate_pathway_network(
        self, pathway_database: PathwayDatabase, ppi_network: nx.Graph
    ) -> nx.Graph:
        """Integrate pathway information with PPI network."""
        # Create integrated network
        integrated_network = ppi_network.copy()

        # Add pathway information to nodes
        for node in integrated_network.nodes():
            integrated_network.nodes[node]["pathways"] = []
            integrated_network.nodes[node]["pathway_types"] = set()

            # Find pathways containing this node
            for pathway in pathway_database.pathways.values():
                if node in pathway.get_all_components():
                    integrated_network.nodes[node]["pathways"].append(
                        pathway.pathway_id
                    )
                    integrated_network.nodes[node]["pathway_types"].add(
                        pathway.pathway_type.value
                    )

        # Add pathway-based edges
        for pathway in pathway_database.pathways.values():
            components = list(pathway.get_all_components())

            # Add edges between pathway components (if not already present)
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    node1, node2 = components[i], components[j]

                    if (
                        node1 in integrated_network
                        and node2 in integrated_network
                        and not integrated_network.has_edge(node1, node2)
                    ):
                        integrated_network.add_edge(
                            node1,
                            node2,
                            interaction_type="pathway_association",
                            pathway_id=pathway.pathway_id,
                            weight=0.3,  # Lower weight for pathway associations
                        )

        return integrated_network


# Utility functions
def load_pathway_database(
    source: DatabaseSource, organism: str = "hsa"
) -> PathwayDatabase:
    """Load pathway database from specified source."""
    if source == DatabaseSource.KEGG:
        integration = KEGGIntegration()
        return integration.load_pathways(organism)
    elif source == DatabaseSource.REACTOME:
        integration = ReactomeIntegration()
        return integration.load_pathways(organism)
    else:
        raise ValueError(f"Unsupported database source: {source}")


def predict_pathway_activation(
    gene_expression: Dict[str, float],
    pathway_database: PathwayDatabase,
    method: str = "gsea",
) -> List[PathwayActivation]:
    """Predict pathway activation from gene expression data."""
    predictor = PathwayActivationPredictor()

    if method.lower() == "gsea":
        return predictor.predict_activation_gsea(gene_expression, pathway_database)
    else:
        raise ValueError(f"Unsupported activation prediction method: {method}")


def analyze_phenotype_changes(
    compound_smiles: str,
    pathway_activations: List[PathwayActivation],
    time_points: Optional[List[float]] = None,
) -> List[PhenotypeChange]:
    """Analyze phenotype changes from pathway activations."""
    predictor = CellularPhenotypePredictor()
    return predictor.predict_phenotype_changes(
        compound_smiles, pathway_activations, time_points
    )


def integrate_ppi_networks(networks: List[nx.Graph]) -> nx.Graph:
    """Integrate multiple PPI networks into a single network."""
    if not networks:
        return nx.Graph()

    # Start with first network
    integrated = networks[0].copy()

    # Add edges from other networks
    for network in networks[1:]:
        for edge in network.edges(data=True):
            node1, node2, data = edge

            if integrated.has_edge(node1, node2):
                # Combine edge weights
                current_weight = integrated.get_edge_data(node1, node2).get(
                    "weight", 1.0
                )
                new_weight = data.get("weight", 1.0)
                integrated[node1][node2]["weight"] = max(current_weight, new_weight)
            else:
                integrated.add_edge(node1, node2, **data)

    return integrated


# Exceptions
class PathwayModelingError(Exception):
    """Base exception for pathway modeling errors."""

    pass


class DatabaseIntegrationError(PathwayModelingError):
    """Exception raised when database integration fails."""

    pass


class NetworkAnalysisError(PathwayModelingError):
    """Exception raised when network analysis fails."""

    pass

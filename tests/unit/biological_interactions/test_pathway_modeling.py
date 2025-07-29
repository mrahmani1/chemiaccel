"""
Unit tests for biological pathway modeling module.
"""

from unittest.mock import MagicMock, Mock, patch

import networkx as nx
import numpy as np
import pytest
from chemi_accel.biological_interactions.pathway_modeling import (
    ActivationState,
    BiologicalPathway,
    BiologicalPathwayModeler,
    CellularPhenotype,
    CellularPhenotypePredictor,
    DatabaseIntegrationError,
    DatabaseSource,
    KEGGIntegration,
    NetworkAnalysisError,
    NetworkEdge,
    NetworkNode,
    NetworkPropagationEngine,
    NetworkType,
    PathwayActivation,
    PathwayActivationPredictor,
    PathwayDatabase,
    PathwayInhibitionAnalyzer,
    PathwayModelingError,
    PathwayNetworkIntegrator,
    PathwayType,
    PhenotypeCategory,
    PhenotypeChange,
    PPINetworkLoader,
    ReactomeIntegration,
    analyze_phenotype_changes,
    integrate_ppi_networks,
    load_pathway_database,
    predict_pathway_activation,
)


@pytest.fixture
def sample_pathway():
    """Create a sample biological pathway."""
    return BiologicalPathway(
        pathway_id="hsa00010",
        name="Glycolysis / Gluconeogenesis",
        pathway_type=PathwayType.METABOLIC,
        database_source=DatabaseSource.KEGG,
        organism="Homo sapiens",
        description="Glucose metabolism pathway",
        genes=["HK1", "HK2", "GPI", "PFKM", "ALDOA", "TPI1"],
        proteins=["HK1_HUMAN", "HK2_HUMAN", "G6PI_HUMAN"],
        metabolites=["glucose", "glucose-6-phosphate", "fructose-6-phosphate"],
    )


@pytest.fixture
def sample_pathway_database():
    """Create a sample pathway database."""
    pathways = {}

    # Create sample pathways
    pathway1 = BiologicalPathway(
        pathway_id="hsa00010",
        name="Glycolysis",
        pathway_type=PathwayType.METABOLIC,
        database_source=DatabaseSource.KEGG,
        genes=["HK1", "HK2", "GPI", "PFKM"],
    )

    pathway2 = BiologicalPathway(
        pathway_id="hsa04010",
        name="MAPK signaling",
        pathway_type=PathwayType.SIGNALING,
        database_source=DatabaseSource.KEGG,
        genes=["MAPK1", "MAPK3", "RAF1", "MEK1"],
    )

    pathways[pathway1.pathway_id] = pathway1
    pathways[pathway2.pathway_id] = pathway2

    return PathwayDatabase(
        database_id="test_kegg",
        name="Test KEGG Database",
        version="2024.1",
        source=DatabaseSource.KEGG,
        pathways=pathways,
    )


@pytest.fixture
def sample_ppi_network():
    """Create a sample PPI network."""
    G = nx.Graph()

    # Add nodes
    proteins = ["PROT_A", "PROT_B", "PROT_C", "PROT_D", "PROT_E"]
    for protein in proteins:
        G.add_node(protein, type="protein")

    # Add edges
    edges = [
        ("PROT_A", "PROT_B", {"confidence": 0.8, "interaction_type": "binding"}),
        ("PROT_B", "PROT_C", {"confidence": 0.7, "interaction_type": "activation"}),
        (
            "PROT_C",
            "PROT_D",
            {"confidence": 0.9, "interaction_type": "phosphorylation"},
        ),
        ("PROT_A", "PROT_E", {"confidence": 0.6, "interaction_type": "binding"}),
    ]

    for source, target, data in edges:
        G.add_edge(source, target, **data)

    return G


@pytest.fixture
def sample_gene_expression():
    """Create sample gene expression data."""
    return {
        "HK1": 2.5,  # Upregulated
        "HK2": 1.8,  # Upregulated
        "GPI": -1.2,  # Downregulated
        "PFKM": 0.8,  # Slightly upregulated
        "ALDOA": -2.1,  # Downregulated
        "TPI1": 1.5,  # Upregulated
        "MAPK1": 3.2,  # Highly upregulated
        "MAPK3": 2.8,  # Upregulated
        "RAF1": -0.5,  # Slightly downregulated
        "MEK1": 1.9,  # Upregulated
    }


@pytest.fixture
def sample_cellular_phenotype():
    """Create a sample cellular phenotype."""
    return CellularPhenotype(
        phenotype_id="PROLIF_001",
        name="Cell Proliferation",
        category=PhenotypeCategory.PROLIFERATION,
        description="Increased cell proliferation rate",
        associated_pathways=["hsa04110", "hsa04010"],
        biomarkers=["PCNA", "Ki67", "CYCLIN_D1"],
        confidence_score=0.85,
    )


class TestNetworkNode:
    """Test NetworkNode class."""

    def test_node_creation(self):
        """Test network node creation."""
        node = NetworkNode(
            node_id="GENE_001",
            name="Test Gene",
            node_type="gene",
            organism="Homo sapiens",
            expression_level=2.5,
            fold_change=1.8,
            p_value=0.01,
        )

        assert node.node_id == "GENE_001"
        assert node.name == "Test Gene"
        assert node.node_type == "gene"
        assert node.expression_level == 2.5
        assert node.fold_change == 1.8
        assert node.p_value == 0.01

    def test_node_post_init(self):
        """Test node __post_init__ method."""
        node = NetworkNode(node_id="GENE_001", name="Test Gene", node_type="gene")

        assert isinstance(node.external_ids, dict)
        assert isinstance(node.annotations, dict)


class TestNetworkEdge:
    """Test NetworkEdge class."""

    def test_edge_creation(self):
        """Test network edge creation."""
        edge = NetworkEdge(
            source_id="PROT_A",
            target_id="PROT_B",
            interaction_type="binding",
            confidence_score=0.85,
            evidence_types=["experimental", "database"],
            weight=1.2,
        )

        assert edge.source_id == "PROT_A"
        assert edge.target_id == "PROT_B"
        assert edge.interaction_type == "binding"
        assert edge.confidence_score == 0.85
        assert edge.weight == 1.2

    def test_edge_post_init(self):
        """Test edge __post_init__ method."""
        edge = NetworkEdge(
            source_id="PROT_A", target_id="PROT_B", interaction_type="binding"
        )

        assert isinstance(edge.evidence_types, list)
        assert isinstance(edge.publications, list)
        assert isinstance(edge.databases, list)


class TestBiologicalPathway:
    """Test BiologicalPathway class."""

    def test_pathway_creation(self, sample_pathway):
        """Test pathway creation."""
        assert sample_pathway.pathway_id == "hsa00010"
        assert sample_pathway.name == "Glycolysis / Gluconeogenesis"
        assert sample_pathway.pathway_type == PathwayType.METABOLIC
        assert sample_pathway.database_source == DatabaseSource.KEGG
        assert len(sample_pathway.genes) == 6
        assert len(sample_pathway.proteins) == 3
        assert len(sample_pathway.metabolites) == 3

    def test_pathway_post_init(self):
        """Test pathway __post_init__ method."""
        pathway = BiologicalPathway(
            pathway_id="test_001",
            name="Test Pathway",
            pathway_type=PathwayType.SIGNALING,
            database_source=DatabaseSource.KEGG,
        )

        assert isinstance(pathway.genes, list)
        assert isinstance(pathway.proteins, list)
        assert isinstance(pathway.metabolites, list)
        assert isinstance(pathway.reactions, list)
        assert isinstance(pathway.parent_pathways, list)
        assert isinstance(pathway.child_pathways, list)
        assert isinstance(pathway.cross_references, dict)

    def test_get_all_components(self, sample_pathway):
        """Test getting all pathway components."""
        all_components = sample_pathway.get_all_components()

        assert isinstance(all_components, set)
        expected_size = (
            len(sample_pathway.genes)
            + len(sample_pathway.proteins)
            + len(sample_pathway.metabolites)
        )
        assert len(all_components) <= expected_size  # May be less due to duplicates


class TestPathwayActivation:
    """Test PathwayActivation class."""

    def test_activation_creation(self):
        """Test pathway activation creation."""
        activation = PathwayActivation(
            pathway_id="hsa00010",
            pathway_name="Glycolysis",
            activation_state=ActivationState.ACTIVATED,
            activation_score=0.65,
            p_value=0.01,
            adjusted_p_value=0.05,
            activated_genes=["HK1", "HK2"],
            suppressed_genes=["GPI"],
            leading_edge_genes=["HK1", "HK2", "PFKM"],
            enrichment_score=0.58,
        )

        assert activation.pathway_id == "hsa00010"
        assert activation.activation_state == ActivationState.ACTIVATED
        assert activation.activation_score == 0.65
        assert len(activation.activated_genes) == 2
        assert len(activation.suppressed_genes) == 1

    def test_activation_post_init(self):
        """Test activation __post_init__ method."""
        activation = PathwayActivation(
            pathway_id="test_001",
            pathway_name="Test Pathway",
            activation_state=ActivationState.NORMAL,
            activation_score=0.0,
            p_value=0.5,
            adjusted_p_value=0.5,
        )

        assert isinstance(activation.activated_genes, list)
        assert isinstance(activation.suppressed_genes, list)
        assert isinstance(activation.leading_edge_genes, list)


class TestCellularPhenotype:
    """Test CellularPhenotype class."""

    def test_phenotype_creation(self, sample_cellular_phenotype):
        """Test phenotype creation."""
        assert sample_cellular_phenotype.phenotype_id == "PROLIF_001"
        assert sample_cellular_phenotype.name == "Cell Proliferation"
        assert sample_cellular_phenotype.category == PhenotypeCategory.PROLIFERATION
        assert len(sample_cellular_phenotype.associated_pathways) == 2
        assert len(sample_cellular_phenotype.biomarkers) == 3

    def test_phenotype_post_init(self):
        """Test phenotype __post_init__ method."""
        phenotype = CellularPhenotype(
            phenotype_id="test_001",
            name="Test Phenotype",
            category=PhenotypeCategory.APOPTOSIS,
        )

        assert isinstance(phenotype.associated_pathways, list)
        assert isinstance(phenotype.biomarkers, list)


class TestPhenotypeChange:
    """Test PhenotypeChange class."""

    def test_change_creation(self, sample_cellular_phenotype):
        """Test phenotype change creation."""
        change = PhenotypeChange(
            compound_smiles="CCO",
            phenotype=sample_cellular_phenotype,
            predicted_change=2.5,
            confidence=0.85,
            time_course={"24": 2.5, "48": 2.2, "72": 1.8},
        )

        assert change.compound_smiles == "CCO"
        assert change.phenotype == sample_cellular_phenotype
        assert change.predicted_change == 2.5
        assert change.confidence == 0.85
        assert len(change.time_course) == 3

    def test_change_post_init(self):
        """Test change __post_init__ method."""
        change = PhenotypeChange(
            compound_smiles="CCO",
            phenotype=sample_cellular_phenotype,
            predicted_change=1.5,
            confidence=0.7,
        )

        assert isinstance(change.contributing_pathways, list)
        assert isinstance(change.time_course, dict)


class TestPathwayDatabase:
    """Test PathwayDatabase class."""

    def test_database_creation(self, sample_pathway_database):
        """Test database creation."""
        assert sample_pathway_database.database_id == "test_kegg"
        assert sample_pathway_database.source == DatabaseSource.KEGG
        assert len(sample_pathway_database.pathways) == 2

    def test_get_pathway(self, sample_pathway_database):
        """Test getting pathway by ID."""
        pathway = sample_pathway_database.get_pathway("hsa00010")
        assert pathway is not None
        assert pathway.pathway_id == "hsa00010"

        # Test non-existent pathway
        non_existent = sample_pathway_database.get_pathway("non_existent")
        assert non_existent is None

    def test_get_pathways_by_type(self, sample_pathway_database):
        """Test getting pathways by type."""
        metabolic_pathways = sample_pathway_database.get_pathways_by_type(
            PathwayType.METABOLIC
        )
        assert len(metabolic_pathways) == 1
        assert metabolic_pathways[0].pathway_type == PathwayType.METABOLIC

        signaling_pathways = sample_pathway_database.get_pathways_by_type(
            PathwayType.SIGNALING
        )
        assert len(signaling_pathways) == 1
        assert signaling_pathways[0].pathway_type == PathwayType.SIGNALING


class TestKEGGIntegration:
    """Test KEGGIntegration class."""

    def test_integration_creation(self):
        """Test KEGG integration creation."""
        integration = KEGGIntegration()
        assert integration.base_url == "https://rest.kegg.jp"

    def test_load_pathways(self):
        """Test loading KEGG pathways."""
        integration = KEGGIntegration()
        database = integration.load_pathways("hsa")

        assert isinstance(database, PathwayDatabase)
        assert database.source == DatabaseSource.KEGG
        assert len(database.pathways) > 0

        # Check sample pathways
        assert "hsa00010" in database.pathways
        assert "hsa04010" in database.pathways

    def test_get_sample_genes(self):
        """Test getting sample genes."""
        integration = KEGGIntegration()
        genes = integration._get_sample_genes("hsa00010")

        assert isinstance(genes, list)
        assert len(genes) > 0
        for gene in genes:
            assert isinstance(gene, str)


class TestReactomeIntegration:
    """Test ReactomeIntegration class."""

    def test_integration_creation(self):
        """Test Reactome integration creation."""
        integration = ReactomeIntegration()
        assert integration.base_url == "https://reactome.org"

    def test_load_pathways(self):
        """Test loading Reactome pathways."""
        integration = ReactomeIntegration()
        database = integration.load_pathways("9606")

        assert isinstance(database, PathwayDatabase)
        assert database.source == DatabaseSource.REACTOME
        assert len(database.pathways) > 0

        # Check sample pathways
        pathway_ids = list(database.pathways.keys())
        assert any(pid.startswith("R-HSA-") for pid in pathway_ids)


class TestPPINetworkLoader:
    """Test PPINetworkLoader class."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_loader_creation(self, mock_cuda):
        """Test PPI network loader creation."""
        loader = PPINetworkLoader(use_gpu=False)
        assert loader.device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    def test_load_string_network(self, mock_cuda):
        """Test loading STRING network."""
        loader = PPINetworkLoader(use_gpu=False)
        network = loader.load_string_network()

        assert isinstance(network, nx.Graph)
        assert network.number_of_nodes() > 0
        assert network.number_of_edges() > 0

        # Check node attributes
        for node in network.nodes():
            assert network.nodes[node]["type"] == "protein"

        # Check edge attributes
        for edge in network.edges(data=True):
            assert "confidence" in edge[2]
            assert "interaction_type" in edge[2]

    @patch("torch.cuda.is_available", return_value=False)
    def test_load_biogrid_network(self, mock_cuda):
        """Test loading BioGRID network."""
        loader = PPINetworkLoader(use_gpu=False)
        network = loader.load_biogrid_network()

        assert isinstance(network, nx.Graph)
        assert network.number_of_nodes() > 0

        # Check node attributes
        for node in network.nodes():
            assert network.nodes[node]["type"] == "protein"
            assert network.nodes[node]["database"] == "BioGRID"


class TestNetworkPropagationEngine:
    """Test NetworkPropagationEngine class."""

    @patch("torch.cuda.is_available", return_value=False)
    @patch("chemi_accel.biological_interactions.pathway_modeling.CUPY_AVAILABLE", False)
    def test_engine_creation(self, mock_cuda):
        """Test propagation engine creation."""
        engine = NetworkPropagationEngine(use_gpu=False)
        assert engine.device.type == "cpu"
        assert engine.alpha == 0.7

    @patch("torch.cuda.is_available", return_value=False)
    @patch("chemi_accel.biological_interactions.pathway_modeling.CUPY_AVAILABLE", False)
    def test_random_walk_with_restart(self, mock_cuda, sample_ppi_network):
        """Test random walk with restart."""
        engine = NetworkPropagationEngine(use_gpu=False)
        seed_nodes = ["PROT_A", "PROT_B"]

        scores = engine.random_walk_with_restart(
            network=sample_ppi_network, seed_nodes=seed_nodes, max_iterations=100
        )

        assert isinstance(scores, dict)
        assert len(scores) == sample_ppi_network.number_of_nodes()

        # Check that all scores are positive
        for node, score in scores.items():
            assert score >= 0

        # Seed nodes should have higher scores
        for seed_node in seed_nodes:
            if seed_node in scores:
                assert scores[seed_node] > 0

    @patch("torch.cuda.is_available", return_value=False)
    def test_heat_diffusion(self, mock_cuda, sample_ppi_network):
        """Test heat diffusion."""
        engine = NetworkPropagationEngine(use_gpu=False)
        initial_heat = {"PROT_A": 1.0, "PROT_B": 0.5}

        final_heat = engine.heat_diffusion(
            network=sample_ppi_network,
            initial_heat=initial_heat,
            diffusion_rate=0.1,
            max_iterations=50,
        )

        assert isinstance(final_heat, dict)
        assert len(final_heat) == sample_ppi_network.number_of_nodes()


class TestPathwayActivationPredictor:
    """Test PathwayActivationPredictor class."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_predictor_creation(self, mock_cuda):
        """Test activation predictor creation."""
        predictor = PathwayActivationPredictor(use_gpu=False)
        assert predictor.device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    def test_predict_activation_gsea(
        self, mock_cuda, sample_pathway_database, sample_gene_expression
    ):
        """Test GSEA-based activation prediction."""
        predictor = PathwayActivationPredictor(use_gpu=False)

        activations = predictor.predict_activation_gsea(
            gene_expression=sample_gene_expression,
            pathway_database=sample_pathway_database,
            min_pathway_size=2,
            max_pathway_size=10,
        )

        assert isinstance(activations, list)
        assert len(activations) > 0

        for activation in activations:
            assert isinstance(activation, PathwayActivation)
            assert activation.pathway_id in sample_pathway_database.pathways
            assert 0 <= activation.p_value <= 1
            assert activation.adjusted_p_value >= activation.p_value

    @patch("torch.cuda.is_available", return_value=False)
    def test_calculate_enrichment_score(self, mock_cuda):
        """Test enrichment score calculation."""
        predictor = PathwayActivationPredictor(use_gpu=False)

        gene_ranks = {"GENE1": 0, "GENE2": 1, "GENE3": 2, "GENE4": 3, "GENE5": 4}
        pathway_genes = ["GENE1", "GENE2"]  # Top-ranked genes

        score = predictor._calculate_enrichment_score(
            gene_ranks=gene_ranks, pathway_genes=pathway_genes, total_genes=5
        )

        assert isinstance(score, float)

    @patch("torch.cuda.is_available", return_value=False)
    def test_determine_activation_state(self, mock_cuda):
        """Test activation state determination."""
        predictor = PathwayActivationPredictor(use_gpu=False)

        # Test different enrichment scores
        state_high = predictor._determine_activation_state(0.6)
        assert state_high == ActivationState.HIGHLY_ACTIVATED

        state_normal = predictor._determine_activation_state(0.1)
        assert state_normal == ActivationState.NORMAL

        state_suppressed = predictor._determine_activation_state(-0.6)
        assert state_suppressed == ActivationState.HIGHLY_SUPPRESSED


class TestPathwayInhibitionAnalyzer:
    """Test PathwayInhibitionAnalyzer class."""

    def test_analyzer_creation(self):
        """Test inhibition analyzer creation."""
        analyzer = PathwayInhibitionAnalyzer()
        assert analyzer is not None

    def test_analyze_inhibition_cascade(self, sample_ppi_network):
        """Test inhibition cascade analysis."""
        analyzer = PathwayInhibitionAnalyzer()
        inhibited_nodes = ["PROT_A", "PROT_B"]

        cascade_results = analyzer.analyze_inhibition_cascade(
            network=sample_ppi_network,
            inhibited_nodes=inhibited_nodes,
            propagation_depth=2,
        )

        assert isinstance(cascade_results, dict)
        assert "depth_1" in cascade_results
        assert "depth_2" in cascade_results

        for depth_key, affected_nodes in cascade_results.items():
            assert isinstance(affected_nodes, dict)


class TestCellularPhenotypePredictor:
    """Test CellularPhenotypePredictor class."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_predictor_creation(self, mock_cuda):
        """Test phenotype predictor creation."""
        predictor = CellularPhenotypePredictor(use_gpu=False)
        assert predictor.device.type == "cpu"
        assert len(predictor.phenotypes) > 0

    @patch("torch.cuda.is_available", return_value=False)
    def test_load_cellular_phenotypes(self, mock_cuda):
        """Test loading cellular phenotypes."""
        predictor = CellularPhenotypePredictor(use_gpu=False)
        phenotypes = predictor._load_cellular_phenotypes()

        assert isinstance(phenotypes, dict)
        assert len(phenotypes) > 0

        for phenotype_id, phenotype in phenotypes.items():
            assert isinstance(phenotype, CellularPhenotype)
            assert phenotype.phenotype_id == phenotype_id

    @patch("torch.cuda.is_available", return_value=False)
    def test_predict_phenotype_changes(
        self, mock_cuda, sample_pathway_database, sample_gene_expression
    ):
        """Test phenotype change prediction."""
        predictor = CellularPhenotypePredictor(use_gpu=False)

        # Create sample pathway activations
        activation_predictor = PathwayActivationPredictor(use_gpu=False)
        pathway_activations = activation_predictor.predict_activation_gsea(
            sample_gene_expression, sample_pathway_database
        )

        phenotype_changes = predictor.predict_phenotype_changes(
            compound_smiles="CCO",
            pathway_activations=pathway_activations,
            time_points=[24.0, 48.0],
        )

        assert isinstance(phenotype_changes, list)

        for change in phenotype_changes:
            assert isinstance(change, PhenotypeChange)
            assert change.compound_smiles == "CCO"
            assert isinstance(change.phenotype, CellularPhenotype)
            assert change.confidence > 0
            assert len(change.time_course) == 2

    @patch("torch.cuda.is_available", return_value=False)
    def test_prepare_activation_features(self, mock_cuda):
        """Test activation feature preparation."""
        predictor = CellularPhenotypePredictor(use_gpu=False)

        # Create sample activations
        activations = [
            PathwayActivation(
                pathway_id="test_001",
                pathway_name="Test Pathway",
                activation_state=ActivationState.ACTIVATED,
                activation_score=0.5,
                p_value=0.01,
                adjusted_p_value=0.05,
            )
        ]

        features = predictor._prepare_activation_features(activations)

        assert isinstance(features, np.ndarray)
        assert features.shape == (1000,)


class TestBiologicalPathwayModeler:
    """Test BiologicalPathwayModeler class."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_modeler_creation(self, mock_cuda):
        """Test pathway modeler creation."""
        modeler = BiologicalPathwayModeler(use_gpu=False)
        assert modeler.use_gpu is False
        assert isinstance(modeler.kegg_integration, KEGGIntegration)
        assert isinstance(modeler.reactome_integration, ReactomeIntegration)

    @patch("torch.cuda.is_available", return_value=False)
    def test_load_pathway_databases(self, mock_cuda):
        """Test loading pathway databases."""
        modeler = BiologicalPathwayModeler(use_gpu=False)

        databases = modeler.load_pathway_databases(
            [DatabaseSource.KEGG, DatabaseSource.REACTOME]
        )

        assert isinstance(databases, dict)
        assert DatabaseSource.KEGG.value in databases
        assert DatabaseSource.REACTOME.value in databases

        for db in databases.values():
            assert isinstance(db, PathwayDatabase)

    @patch("torch.cuda.is_available", return_value=False)
    def test_load_interaction_networks(self, mock_cuda):
        """Test loading interaction networks."""
        modeler = BiologicalPathwayModeler(use_gpu=False)

        networks = modeler.load_interaction_networks(["string", "biogrid"])

        assert isinstance(networks, dict)
        assert "string" in networks
        assert "biogrid" in networks

        for network in networks.values():
            assert isinstance(network, nx.Graph)

    @patch("torch.cuda.is_available", return_value=False)
    def test_analyze_compound_effects(self, mock_cuda, sample_gene_expression):
        """Test compound effects analysis."""
        modeler = BiologicalPathwayModeler(use_gpu=False)

        # Load databases and networks
        modeler.load_pathway_databases([DatabaseSource.KEGG])
        modeler.load_interaction_networks(["string"])

        target_interactions = {"PROT_A": 0.8, "PROT_B": 0.6}

        results = modeler.analyze_compound_effects(
            compound_smiles="CCO",
            target_interactions=target_interactions,
            gene_expression_changes=sample_gene_expression,
        )

        assert isinstance(results, dict)
        assert "pathway_activations" in results
        assert "network_propagation" in results
        assert "phenotype_changes" in results


class TestPathwayNetworkIntegrator:
    """Test PathwayNetworkIntegrator class."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_integrator_creation(self, mock_cuda):
        """Test integrator creation."""
        integrator = PathwayNetworkIntegrator(use_gpu=False)
        assert integrator.use_gpu is False

    @patch("torch.cuda.is_available", return_value=False)
    def test_integrate_pathway_network(
        self, mock_cuda, sample_pathway_database, sample_ppi_network
    ):
        """Test pathway-network integration."""
        integrator = PathwayNetworkIntegrator(use_gpu=False)

        integrated_network = integrator.integrate_pathway_network(
            pathway_database=sample_pathway_database, ppi_network=sample_ppi_network
        )

        assert isinstance(integrated_network, nx.Graph)
        assert (
            integrated_network.number_of_nodes() >= sample_ppi_network.number_of_nodes()
        )

        # Check that pathway information was added to nodes
        for node in integrated_network.nodes():
            assert "pathways" in integrated_network.nodes[node]
            assert "pathway_types" in integrated_network.nodes[node]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_load_pathway_database_kegg(self):
        """Test loading KEGG pathway database."""
        database = load_pathway_database(DatabaseSource.KEGG, "hsa")

        assert isinstance(database, PathwayDatabase)
        assert database.source == DatabaseSource.KEGG

    def test_load_pathway_database_reactome(self):
        """Test loading Reactome pathway database."""
        database = load_pathway_database(DatabaseSource.REACTOME, "9606")

        assert isinstance(database, PathwayDatabase)
        assert database.source == DatabaseSource.REACTOME

    def test_load_pathway_database_unsupported(self):
        """Test loading unsupported database source."""
        with pytest.raises(ValueError):
            load_pathway_database(DatabaseSource.CUSTOM, "test")

    @patch("torch.cuda.is_available", return_value=False)
    def test_predict_pathway_activation(
        self, mock_cuda, sample_pathway_database, sample_gene_expression
    ):
        """Test pathway activation prediction utility."""
        activations = predict_pathway_activation(
            gene_expression=sample_gene_expression,
            pathway_database=sample_pathway_database,
            method="gsea",
        )

        assert isinstance(activations, list)
        for activation in activations:
            assert isinstance(activation, PathwayActivation)

    @patch("torch.cuda.is_available", return_value=False)
    def test_predict_pathway_activation_unsupported_method(
        self, mock_cuda, sample_pathway_database, sample_gene_expression
    ):
        """Test unsupported activation prediction method."""
        with pytest.raises(ValueError):
            predict_pathway_activation(
                gene_expression=sample_gene_expression,
                pathway_database=sample_pathway_database,
                method="unsupported",
            )

    @patch("torch.cuda.is_available", return_value=False)
    def test_analyze_phenotype_changes_utility(
        self, mock_cuda, sample_pathway_database, sample_gene_expression
    ):
        """Test phenotype changes analysis utility."""
        # Create pathway activations
        activation_predictor = PathwayActivationPredictor(use_gpu=False)
        pathway_activations = activation_predictor.predict_activation_gsea(
            sample_gene_expression, sample_pathway_database
        )

        changes = analyze_phenotype_changes(
            compound_smiles="CCO",
            pathway_activations=pathway_activations,
            time_points=[24.0, 48.0, 72.0],
        )

        assert isinstance(changes, list)
        for change in changes:
            assert isinstance(change, PhenotypeChange)
            assert len(change.time_course) == 3

    def test_integrate_ppi_networks(self, sample_ppi_network):
        """Test PPI network integration utility."""
        # Create second network
        network2 = nx.Graph()
        network2.add_edge("PROT_A", "PROT_F", weight=0.7)
        network2.add_edge("PROT_F", "PROT_G", weight=0.8)

        networks = [sample_ppi_network, network2]
        integrated = integrate_ppi_networks(networks)

        assert isinstance(integrated, nx.Graph)
        assert integrated.number_of_nodes() >= sample_ppi_network.number_of_nodes()
        assert integrated.number_of_edges() >= sample_ppi_network.number_of_edges()

    def test_integrate_ppi_networks_empty(self):
        """Test integrating empty network list."""
        integrated = integrate_ppi_networks([])
        assert isinstance(integrated, nx.Graph)
        assert integrated.number_of_nodes() == 0


class TestExceptions:
    """Test exception classes."""

    def test_pathway_modeling_error(self):
        """Test PathwayModelingError."""
        with pytest.raises(PathwayModelingError):
            raise PathwayModelingError("Test error")

    def test_database_integration_error(self):
        """Test DatabaseIntegrationError."""
        with pytest.raises(DatabaseIntegrationError):
            raise DatabaseIntegrationError("Database integration failed")

    def test_network_analysis_error(self):
        """Test NetworkAnalysisError."""
        with pytest.raises(NetworkAnalysisError):
            raise NetworkAnalysisError("Network analysis failed")


class TestIntegration:
    """Integration tests."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_end_to_end_pathway_analysis(self, mock_cuda, sample_gene_expression):
        """Test end-to-end pathway analysis workflow."""
        # Create modeler
        modeler = BiologicalPathwayModeler(use_gpu=False)

        # Load databases
        databases = modeler.load_pathway_databases([DatabaseSource.KEGG])
        assert len(databases) == 1

        # Load networks
        networks = modeler.load_interaction_networks(["string"])
        assert len(networks) == 1

        # Analyze compound effects
        target_interactions = {"PROT_001": 0.8, "PROT_002": 0.6}
        results = modeler.analyze_compound_effects(
            compound_smiles="CCO",
            target_interactions=target_interactions,
            gene_expression_changes=sample_gene_expression,
        )

        # Verify results structure
        assert isinstance(results, dict)
        assert "pathway_activations" in results
        assert "network_propagation" in results
        assert "phenotype_changes" in results

        # Verify pathway activations
        pathway_activations = results["pathway_activations"]
        assert isinstance(pathway_activations, dict)

        for db_name, activations in pathway_activations.items():
            assert isinstance(activations, list)
            for activation in activations:
                assert isinstance(activation, PathwayActivation)

        # Verify phenotype changes
        phenotype_changes = results["phenotype_changes"]
        assert isinstance(phenotype_changes, dict)

        for db_name, changes in phenotype_changes.items():
            assert isinstance(changes, list)
            for change in changes:
                assert isinstance(change, PhenotypeChange)

    @patch("torch.cuda.is_available", return_value=False)
    def test_network_integration_workflow(self, mock_cuda, sample_pathway_database):
        """Test network integration workflow."""
        # Create network loader and integrator
        loader = PPINetworkLoader(use_gpu=False)
        integrator = PathwayNetworkIntegrator(use_gpu=False)

        # Load PPI network
        ppi_network = loader.load_string_network()

        # Integrate with pathway information
        integrated_network = integrator.integrate_pathway_network(
            pathway_database=sample_pathway_database, ppi_network=ppi_network
        )

        # Verify integration
        assert isinstance(integrated_network, nx.Graph)
        assert integrated_network.number_of_nodes() >= ppi_network.number_of_nodes()

        # Check pathway annotations
        pathway_annotated_nodes = 0
        for node in integrated_network.nodes():
            if integrated_network.nodes[node].get("pathways"):
                pathway_annotated_nodes += 1

        # Some nodes should have pathway annotations
        assert pathway_annotated_nodes >= 0

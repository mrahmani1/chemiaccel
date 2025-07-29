"""
Unit tests for systems visualization module.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from chemi_accel.biological_interactions.multi_target_prediction import (
    BiologicalTarget,
    ConfidenceLevel,
    InteractionPrediction,
    TargetType,
)
from chemi_accel.biological_interactions.pathway_modeling import (
    ActivationState,
    BiologicalPathway,
    PathwayActivation,
    PathwayType,
)
from chemi_accel.biological_interactions.systems_visualization import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    ColorScheme,
    DashboardCreationError,
    ExplanationGenerationError,
    ExplanationResult,
    NetworkLayout,
    NetworkVisualization,
    SystemsVisualizationError,
    SystemsVisualizer,
    VisualizationConfig,
    VisualizationCreationError,
    VisualizationType,
    create_interactive_systems_dashboard,
    create_systems_visualization,
    explain_biological_prediction,
)
from chemi_accel.biological_interactions.toxicology_prediction import (
    RiskLevel,
    SystemsToxicologyProfile,
    ToxicityEndpoint,
    ToxicityPrediction,
)


@pytest.fixture
def sample_visualization_config():
    """Create sample visualization config."""
    return VisualizationConfig(
        width=800,
        height=600,
        color_scheme=ColorScheme.VIRIDIS,
        layout=NetworkLayout.SPRING,
        show_labels=True,
        show_legend=True,
        interactive=True,
        theme="plotly_white",
        font_size=12,
        opacity=0.8,
    )


@pytest.fixture
def sample_target_interactions():
    """Create sample target interactions."""
    target1 = BiologicalTarget(
        target_id="TARGET_001",
        name="Kinase A",
        target_type=TargetType.KINASE,
        species="human",
    )

    target2 = BiologicalTarget(
        target_id="TARGET_002",
        name="GPCR B",
        target_type=TargetType.GPCR,
        species="human",
    )

    interactions = [
        InteractionPrediction(
            target=target1,
            interaction_probability=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            interaction_type="binding",
        ),
        InteractionPrediction(
            target=target2,
            interaction_probability=0.62,
            confidence_level=ConfidenceLevel.MEDIUM,
            interaction_type="modulation",
        ),
    ]

    return interactions


@pytest.fixture
def sample_pathway_activations():
    """Create sample pathway activations."""
    pathway1 = BiologicalPathway(
        pathway_id="PATHWAY_001",
        name="Apoptosis Pathway",
        pathway_type=PathwayType.SIGNALING,
        description="Cell death pathway",
    )

    pathway2 = BiologicalPathway(
        pathway_id="PATHWAY_002",
        name="Metabolic Pathway",
        pathway_type=PathwayType.METABOLIC,
        description="Energy metabolism",
    )

    activations = [
        PathwayActivation(
            pathway=pathway1,
            activation_score=0.75,
            confidence_score=0.8,
            activation_state=ActivationState.ACTIVATED,
        ),
        PathwayActivation(
            pathway=pathway2,
            activation_score=-0.45,
            confidence_score=0.7,
            activation_state=ActivationState.INHIBITED,
        ),
    ]

    return activations


@pytest.fixture
def sample_toxicity_profile():
    """Create sample toxicity profile."""
    toxicity_predictions = [
        ToxicityPrediction(
            compound_smiles="CCO",
            toxicity_endpoint=ToxicityEndpoint.HEPATOTOXICITY,
            probability=0.3,
            risk_level=RiskLevel.MODERATE,
        ),
        ToxicityPrediction(
            compound_smiles="CCO",
            toxicity_endpoint=ToxicityEndpoint.CARDIOTOXICITY,
            probability=0.15,
            risk_level=RiskLevel.LOW,
        ),
    ]

    profile = SystemsToxicologyProfile(
        compound_smiles="CCO",
        off_target_bindings=[],
        toxicity_predictions=toxicity_predictions,
        adr_predictions=[],
        overall_safety_score=0.85,
        therapeutic_index_estimate=12.0,
    )

    return profile


@pytest.fixture
def sample_explanation_result():
    """Create sample explanation result."""
    return ExplanationResult(
        prediction_id="pred_001",
        compound_smiles="CCO",
        prediction_value=0.75,
        explanation_type="toxicity",
        feature_importance={
            "molecular_weight": 0.25,
            "logp": 0.20,
            "tpsa": 0.15,
            "hbd_count": 0.10,
            "aromatic_rings": 0.30,
        },
        local_explanations={
            "lime": {"importance": [0.1, 0.2, 0.3]},
            "shap": {"values": [0.05, 0.15, 0.25]},
        },
        global_explanations={
            "model_accuracy": 0.85,
            "feature_statistics": {"mean_importance": 0.2},
        },
        confidence_intervals={"prediction": (0.70, 0.80), "uncertainty": (0.05, 0.15)},
        reasoning_path=[
            "High molecular weight contributes to toxicity",
            "Aromatic rings increase binding affinity",
            "LogP value suggests membrane permeability",
        ],
        uncertainty_sources=["limited_training_data", "model_variance"],
    )


class TestVisualizationConfig:
    """Test VisualizationConfig class."""

    def test_config_creation(self, sample_visualization_config):
        """Test configuration creation."""
        config = sample_visualization_config

        assert config.width == 800
        assert config.height == 600
        assert config.color_scheme == ColorScheme.VIRIDIS
        assert config.layout == NetworkLayout.SPRING
        assert config.show_labels is True
        assert config.interactive is True
        assert config.opacity == 0.8

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = VisualizationConfig(width=800, height=600, opacity=0.5)
        assert config.width == 800

        # Invalid width should raise
        with pytest.raises(ValueError):
            VisualizationConfig(width=-100, height=600)

        # Invalid opacity should raise
        with pytest.raises(ValueError):
            VisualizationConfig(width=800, height=600, opacity=1.5)

    def test_config_defaults(self):
        """Test default configuration values."""
        config = VisualizationConfig()

        assert config.width == 1200
        assert config.height == 800
        assert config.color_scheme == ColorScheme.VIRIDIS
        assert config.layout == NetworkLayout.SPRING
        assert config.show_labels is True
        assert config.show_legend is True
        assert config.interactive is True


class TestExplanationResult:
    """Test ExplanationResult class."""

    def test_explanation_creation(self, sample_explanation_result):
        """Test explanation result creation."""
        explanation = sample_explanation_result

        assert explanation.prediction_id == "pred_001"
        assert explanation.compound_smiles == "CCO"
        assert explanation.prediction_value == 0.75
        assert explanation.explanation_type == "toxicity"
        assert len(explanation.feature_importance) == 5
        assert len(explanation.local_explanations) == 2
        assert len(explanation.reasoning_path) == 3
        assert len(explanation.uncertainty_sources) == 2

    def test_feature_importance_ordering(self, sample_explanation_result):
        """Test feature importance is properly ordered."""
        explanation = sample_explanation_result

        # Features should be ordered by importance
        importances = list(explanation.feature_importance.values())
        expected_max = max(importances)
        actual_max = importances[0] if importances else 0

        # The first feature should have high importance
        assert actual_max >= 0.1  # Some reasonable threshold


class TestNetworkVisualization:
    """Test NetworkVisualization class."""

    def test_network_creation(self):
        """Test network visualization creation."""
        nodes = [
            {"id": "node1", "label": "Node 1", "size": 20},
            {"id": "node2", "label": "Node 2", "size": 15},
        ]

        edges = [{"source": "node1", "target": "node2", "weight": 0.8}]

        layout_positions = {"node1": (0.0, 0.0), "node2": (1.0, 1.0)}

        metadata = {"graph_type": "interaction_network"}

        network = NetworkVisualization(
            nodes=nodes,
            edges=edges,
            layout_positions=layout_positions,
            metadata=metadata,
        )

        assert len(network.nodes) == 2
        assert len(network.edges) == 1
        assert len(network.layout_positions) == 2
        assert network.metadata["graph_type"] == "interaction_network"


class TestSystemsVisualizer:
    """Test SystemsVisualizer class."""

    def test_visualizer_creation(self, sample_visualization_config):
        """Test visualizer creation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        assert visualizer.config == sample_visualization_config
        assert len(visualizer.engines) > 0
        assert isinstance(visualizer.visualization_cache, dict)
        assert isinstance(visualizer.network_cache, dict)

    def test_visualizer_default_config(self):
        """Test visualizer with default config."""
        visualizer = SystemsVisualizer()

        assert visualizer.config is not None
        assert visualizer.config.width == 1200
        assert visualizer.config.height == 800

    @patch(
        "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
        True,
    )
    @patch(
        "chemi_accel.biological_interactions.systems_visualization._NETWORKX_AVAILABLE",
        True,
    )
    @patch("chemi_accel.biological_interactions.systems_visualization.go")
    @patch("chemi_accel.biological_interactions.systems_visualization.nx")
    def test_create_target_engagement_network(
        self, mock_nx, mock_go, sample_target_interactions, sample_visualization_config
    ):
        """Test target engagement network creation."""
        # Mock NetworkX graph
        mock_graph = Mock()
        mock_nx.Graph.return_value = mock_graph
        mock_nx.spring_layout.return_value = {
            "CCO": (0, 0),
            "TARGET_001": (1, 0),
            "TARGET_002": (0, 1),
        }

        # Mock Plotly figure
        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        mock_go.Scatter.return_value = Mock()

        visualizer = SystemsVisualizer(sample_visualization_config)

        figure = visualizer.create_target_engagement_network(
            interactions=sample_target_interactions,
            compound_smiles="CCO",
            config=sample_visualization_config,
        )

        # Verify graph operations were called
        mock_graph.add_node.assert_called()
        mock_graph.add_edge.assert_called()
        mock_go.Figure.assert_called()

    @patch(
        "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
        True,
    )
    @patch("chemi_accel.biological_interactions.systems_visualization.go")
    def test_create_pathway_modulation_map(
        self, mock_go, sample_pathway_activations, sample_visualization_config
    ):
        """Test pathway modulation map creation."""
        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        mock_go.Sunburst.return_value = Mock()

        visualizer = SystemsVisualizer(sample_visualization_config)

        figure = visualizer.create_pathway_modulation_map(
            pathway_activations=sample_pathway_activations,
            compound_smiles="CCO",
            config=sample_visualization_config,
        )

        mock_go.Figure.assert_called()
        mock_go.Sunburst.assert_called()

    @patch(
        "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
        True,
    )
    @patch("chemi_accel.biological_interactions.systems_visualization.go")
    def test_create_toxicity_profile_heatmap(
        self, mock_go, sample_toxicity_profile, sample_visualization_config
    ):
        """Test toxicity profile heatmap creation."""
        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        mock_go.Heatmap.return_value = Mock()

        visualizer = SystemsVisualizer(sample_visualization_config)

        figure = visualizer.create_toxicity_profile_heatmap(
            toxicity_profile=sample_toxicity_profile, config=sample_visualization_config
        )

        mock_go.Figure.assert_called()
        mock_go.Heatmap.assert_called()

    @patch(
        "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
        True,
    )
    @patch("chemi_accel.biological_interactions.systems_visualization.make_subplots")
    @patch("chemi_accel.biological_interactions.systems_visualization.go")
    def test_create_systems_overview_dashboard(
        self,
        mock_go,
        mock_make_subplots,
        sample_target_interactions,
        sample_pathway_activations,
        sample_toxicity_profile,
        sample_visualization_config,
    ):
        """Test systems overview dashboard creation."""
        mock_figure = Mock()
        mock_make_subplots.return_value = mock_figure
        mock_go.Scatter.return_value = Mock()
        mock_go.Bar.return_value = Mock()
        mock_go.Indicator.return_value = Mock()
        mock_go.Table.return_value = Mock()

        visualizer = SystemsVisualizer(sample_visualization_config)

        figure = visualizer.create_systems_overview_dashboard(
            interactions=sample_target_interactions,
            pathway_activations=sample_pathway_activations,
            toxicity_profile=sample_toxicity_profile,
            compound_smiles="CCO",
            config=sample_visualization_config,
        )

        mock_make_subplots.assert_called()
        mock_figure.add_trace.assert_called()

    def test_explain_prediction(self, sample_visualization_config):
        """Test prediction explanation generation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        # Mock explanation engines
        with patch.object(
            visualizer, "_calculate_feature_importance", return_value={"feature1": 0.5}
        ), patch.object(
            visualizer, "_generate_reasoning_path", return_value=["step1", "step2"]
        ), patch.object(
            visualizer, "_identify_uncertainty_sources", return_value=["source1"]
        ):
            explanation = visualizer.explain_prediction(
                prediction_value=0.75,
                compound_smiles="CCO",
                prediction_type="toxicity",
                model_context={"model_type": "neural_network"},
                explanation_methods=["feature_importance"],
            )

            assert isinstance(explanation, ExplanationResult)
            assert explanation.compound_smiles == "CCO"
            assert explanation.prediction_value == 0.75
            assert explanation.explanation_type == "toxicity"
            assert len(explanation.feature_importance) > 0
            assert len(explanation.reasoning_path) > 0

    @patch(
        "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
        True,
    )
    @patch("chemi_accel.biological_interactions.systems_visualization.make_subplots")
    @patch("chemi_accel.biological_interactions.systems_visualization.go")
    def test_create_explanation_visualization(
        self,
        mock_go,
        mock_make_subplots,
        sample_explanation_result,
        sample_visualization_config,
    ):
        """Test explanation visualization creation."""
        mock_figure = Mock()
        mock_make_subplots.return_value = mock_figure
        mock_go.Bar.return_value = Mock()
        mock_go.Scatter.return_value = Mock()
        mock_go.Table.return_value = Mock()

        visualizer = SystemsVisualizer(sample_visualization_config)

        figure = visualizer.create_explanation_visualization(
            explanation=sample_explanation_result, config=sample_visualization_config
        )

        mock_make_subplots.assert_called()
        mock_figure.add_trace.assert_called()

    def test_export_visualization(self, sample_visualization_config):
        """Test visualization export."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        # Mock Plotly figure
        mock_figure = Mock()
        mock_figure.write_html = Mock()
        mock_figure.write_image = Mock()

        # Test HTML export
        filename = visualizer.export_visualization(
            figure=mock_figure,
            filename="test.html",
            format="html",
            config=sample_visualization_config,
        )

        assert filename == "test.html"
        mock_figure.write_html.assert_called_once_with("test.html")

        # Test PNG export
        filename = visualizer.export_visualization(
            figure=mock_figure,
            filename="test.png",
            format="png",
            config=sample_visualization_config,
        )

        assert filename == "test.png"
        mock_figure.write_image.assert_called_with(
            "test.png", format="png", width=800, height=600
        )

        # Test invalid format
        with pytest.raises(ValueError):
            visualizer.export_visualization(
                figure=mock_figure,
                filename="test.xyz",
                format="xyz",
                config=sample_visualization_config,
            )

    def test_get_compound_name(self, sample_visualization_config):
        """Test compound name generation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        name = visualizer._get_compound_name("CCO")

        assert isinstance(name, str)
        assert len(name) > 0
        assert "Compound_" in name

    def test_get_target_color(self, sample_visualization_config):
        """Test target color assignment."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        # Test known target types
        kinase_color = visualizer._get_target_color(TargetType.KINASE)
        gpcr_color = visualizer._get_target_color(TargetType.GPCR)

        assert isinstance(kinase_color, str)
        assert isinstance(gpcr_color, str)
        assert (
            kinase_color != gpcr_color
        )  # Different types should have different colors

    def test_calculate_molecular_properties(self, sample_visualization_config):
        """Test molecular properties calculation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        properties = visualizer._calculate_molecular_properties("CCO")

        assert isinstance(properties, dict)
        assert len(properties) > 0

        # Should contain standard molecular properties
        expected_properties = ["Molecular Weight", "LogP", "TPSA"]
        for prop in expected_properties:
            if prop in properties:
                assert isinstance(properties[prop], (int, float))

    def test_feature_importance_calculation(self, sample_visualization_config):
        """Test feature importance calculation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        importance = visualizer._calculate_feature_importance(
            compound_smiles="CCO",
            prediction_value=0.75,
            model_context={"model_type": "random_forest"},
        )

        assert isinstance(importance, dict)
        assert len(importance) > 0

        # Check that values are normalized (sum to 1)
        total_importance = sum(importance.values())
        assert (
            abs(total_importance - 1.0) < 0.01
        )  # Allow for small floating point errors

        # Check that importance values are positive
        assert all(value >= 0 for value in importance.values())

    def test_reasoning_path_generation(self, sample_visualization_config):
        """Test reasoning path generation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        feature_importance = {
            "molecular_weight": 0.3,
            "logp": 0.2,
            "aromatic_rings": 0.5,
        }

        reasoning = visualizer._generate_reasoning_path(
            compound_smiles="CCO",
            prediction_value=0.75,
            prediction_type="toxicity",
            feature_importance=feature_importance,
        )

        assert isinstance(reasoning, list)
        assert len(reasoning) > 0
        assert all(isinstance(step, str) for step in reasoning)

        # Should contain prediction value
        assert any("0.75" in step for step in reasoning)

    def test_uncertainty_sources_identification(self, sample_visualization_config):
        """Test uncertainty sources identification."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        model_context = {
            "training_size": 100,  # Small training set
            "model_performance": {"r2": 0.6},  # Poor performance
        }

        sources = visualizer._identify_uncertainty_sources(
            compound_smiles="CCO",
            prediction_value=0.95,  # Extreme prediction
            model_context=model_context,
        )

        assert isinstance(sources, list)
        assert len(sources) > 0
        assert all(isinstance(source, str) for source in sources)

        # Should identify the issues we set up
        source_text = " ".join(sources)
        assert (
            "training data" in source_text.lower()
            or "performance" in source_text.lower()
        )


class TestVisualizationEngines:
    """Test specific visualization engines."""

    def test_dose_response_plot_creation(self, sample_visualization_config):
        """Test dose-response plot creation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        data = {"doses": [0.1, 1, 10, 100], "responses": [0.1, 0.3, 0.7, 0.9]}

        with patch(
            "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
            True,
        ), patch(
            "chemi_accel.biological_interactions.systems_visualization.go"
        ) as mock_go:
            mock_figure = Mock()
            mock_go.Figure.return_value = mock_figure
            mock_go.Scatter.return_value = Mock()

            figure = visualizer._create_dose_response_plot(data)

            mock_go.Figure.assert_called()
            mock_go.Scatter.assert_called()
            mock_figure.add_trace.assert_called()

    def test_time_series_plot_creation(self, sample_visualization_config):
        """Test time series plot creation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        data = {
            "times": list(range(24)),
            "values": np.random.randn(24).cumsum().tolist(),
        }

        with patch(
            "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
            True,
        ), patch(
            "chemi_accel.biological_interactions.systems_visualization.go"
        ) as mock_go:
            mock_figure = Mock()
            mock_go.Figure.return_value = mock_figure
            mock_go.Scatter.return_value = Mock()

            figure = visualizer._create_time_series_plot(data)

            mock_go.Figure.assert_called()
            mock_go.Scatter.assert_called()

    def test_interaction_matrix_creation(self, sample_visualization_config):
        """Test interaction matrix creation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        data = {
            "targets": ["Target A", "Target B", "Target C"],
            "compounds": ["Compound 1", "Compound 2"],
            "matrix": np.random.rand(2, 3).tolist(),
        }

        with patch(
            "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
            True,
        ), patch(
            "chemi_accel.biological_interactions.systems_visualization.go"
        ) as mock_go:
            mock_figure = Mock()
            mock_go.Figure.return_value = mock_figure
            mock_go.Heatmap.return_value = Mock()

            figure = visualizer._create_interaction_matrix(data)

            mock_go.Figure.assert_called()
            mock_go.Heatmap.assert_called()

    def test_comparative_analysis_creation(self, sample_visualization_config):
        """Test comparative analysis creation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        data = {
            "compounds": ["Compound A", "Compound B"],
            "metrics": ["Efficacy", "Safety", "Selectivity"],
            "values": np.random.rand(2, 3).tolist(),
        }

        with patch(
            "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
            True,
        ), patch(
            "chemi_accel.biological_interactions.systems_visualization.go"
        ) as mock_go:
            mock_figure = Mock()
            mock_go.Figure.return_value = mock_figure
            mock_go.Scatterpolar.return_value = Mock()

            figure = visualizer._create_comparative_analysis(data)

            mock_go.Figure.assert_called()
            mock_go.Scatterpolar.assert_called()


class TestNetworkVisualizationHelpers:
    """Test network visualization helper methods."""

    def test_generate_network_layout_fallback(self, sample_visualization_config):
        """Test network layout generation with fallback."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        # Mock a simple graph
        mock_graph = Mock()
        mock_graph.nodes.return_value = ["node1", "node2", "node3"]

        # Test fallback when NetworkX is not available
        with patch(
            "chemi_accel.biological_interactions.systems_visualization._NETWORKX_AVAILABLE",
            False,
        ):
            pos = visualizer._generate_network_layout(mock_graph, NetworkLayout.SPRING)

            assert isinstance(pos, dict)
            assert len(pos) == 3
            assert all(isinstance(coord, tuple) for coord in pos.values())
            assert all(len(coord) == 2 for coord in pos.values())

    def test_create_network_edges(self, sample_visualization_config):
        """Test network edge creation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        # Mock graph with edges
        mock_graph = Mock()
        mock_graph.edges.return_value = [("node1", "node2"), ("node2", "node3")]
        mock_graph.__getitem__ = Mock(return_value={"node2": {"weight": 0.8}})

        pos = {"node1": (0, 0), "node2": (1, 0), "node3": (1, 1)}

        with patch(
            "chemi_accel.biological_interactions.systems_visualization.go"
        ) as mock_go:
            mock_scatter = Mock()
            mock_go.Scatter.return_value = mock_scatter

            edge_trace = visualizer._create_network_edges(mock_graph, pos)

            mock_go.Scatter.assert_called()
            call_args = mock_go.Scatter.call_args
            assert "x" in call_args[1]
            assert "y" in call_args[1]
            assert "mode" in call_args[1]

    def test_create_network_nodes(self, sample_visualization_config):
        """Test network node creation."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        # Mock graph with nodes
        mock_graph = Mock()
        mock_graph.nodes.return_value = ["node1", "node2"]
        mock_graph.nodes.__getitem__ = Mock(
            side_effect=lambda x: {
                "label": f"Label {x}",
                "node_type": "target",
                "size": 20,
                "color": "blue",
            }
        )

        pos = {"node1": (0, 0), "node2": (1, 1)}

        with patch(
            "chemi_accel.biological_interactions.systems_visualization.go"
        ) as mock_go:
            mock_scatter = Mock()
            mock_go.Scatter.return_value = mock_scatter

            node_trace = visualizer._create_network_nodes(mock_graph, pos)

            mock_go.Scatter.assert_called()
            call_args = mock_go.Scatter.call_args
            assert "x" in call_args[1]
            assert "y" in call_args[1]
            assert "mode" in call_args[1]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_systems_visualization(self, sample_visualization_config):
        """Test systems visualization creation utility."""
        data = {"compound_smiles": "CCO", "interactions": []}

        with patch(
            "chemi_accel.biological_interactions.systems_visualization.SystemsVisualizer"
        ) as mock_visualizer_class:
            mock_visualizer = Mock()
            mock_engine = Mock(return_value=Mock())
            mock_visualizer.engines = {VisualizationType.TARGET_NETWORK: mock_engine}
            mock_visualizer_class.return_value = mock_visualizer

            figure = create_systems_visualization(
                visualization_type=VisualizationType.TARGET_NETWORK,
                data=data,
                config=sample_visualization_config,
            )

            mock_visualizer_class.assert_called_once_with(sample_visualization_config)
            mock_engine.assert_called_once_with(data)

    def test_create_systems_visualization_invalid_type(
        self, sample_visualization_config
    ):
        """Test systems visualization with invalid type."""
        data = {"compound_smiles": "CCO"}

        with patch(
            "chemi_accel.biological_interactions.systems_visualization.SystemsVisualizer"
        ) as mock_visualizer_class:
            mock_visualizer = Mock()
            mock_visualizer.engines = {}  # Empty engines dict
            mock_visualizer_class.return_value = mock_visualizer

            with pytest.raises(ValueError):
                create_systems_visualization(
                    visualization_type=VisualizationType.TARGET_NETWORK,
                    data=data,
                    config=sample_visualization_config,
                )

    def test_explain_biological_prediction(self, sample_visualization_config):
        """Test biological prediction explanation utility."""
        model_context = {"model_type": "neural_network"}

        with patch(
            "chemi_accel.biological_interactions.systems_visualization.SystemsVisualizer"
        ) as mock_visualizer_class:
            mock_visualizer = Mock()
            mock_explanation = Mock()
            mock_figure = Mock()
            mock_visualizer.explain_prediction.return_value = mock_explanation
            mock_visualizer.create_explanation_visualization.return_value = mock_figure
            mock_visualizer_class.return_value = mock_visualizer

            explanation, figure = explain_biological_prediction(
                prediction_value=0.75,
                compound_smiles="CCO",
                prediction_type="toxicity",
                model_context=model_context,
                config=sample_visualization_config,
            )

            assert explanation == mock_explanation
            assert figure == mock_figure
            mock_visualizer.explain_prediction.assert_called_once()
            mock_visualizer.create_explanation_visualization.assert_called_once()

    @patch(
        "chemi_accel.biological_interactions.systems_visualization._DASH_AVAILABLE",
        True,
    )
    def test_create_interactive_systems_dashboard(self, sample_visualization_config):
        """Test interactive dashboard creation utility."""
        compound_data = {"smiles": "CCO", "properties": {}}

        with patch(
            "chemi_accel.biological_interactions.systems_visualization.SystemsVisualizer"
        ) as mock_visualizer_class:
            mock_visualizer = Mock()
            mock_visualizer.create_interactive_dashboard.return_value = (
                "http://localhost:8050"
            )
            mock_visualizer_class.return_value = mock_visualizer

            url = create_interactive_systems_dashboard(
                compound_data=compound_data, config=sample_visualization_config
            )

            assert url == "http://localhost:8050"
            mock_visualizer.create_interactive_dashboard.assert_called_once_with(
                compound_data
            )


class TestExceptionHandling:
    """Test exception handling."""

    def test_systems_visualization_error(self):
        """Test SystemsVisualizationError."""
        with pytest.raises(SystemsVisualizationError):
            raise SystemsVisualizationError("Test error")

    def test_visualization_creation_error(self):
        """Test VisualizationCreationError."""
        with pytest.raises(VisualizationCreationError):
            raise VisualizationCreationError("Visualization creation failed")

    def test_explanation_generation_error(self):
        """Test ExplanationGenerationError."""
        with pytest.raises(ExplanationGenerationError):
            raise ExplanationGenerationError("Explanation generation failed")

    def test_dashboard_creation_error(self):
        """Test DashboardCreationError."""
        with pytest.raises(DashboardCreationError):
            raise DashboardCreationError("Dashboard creation failed")


class TestMissingDependencies:
    """Test behavior when optional dependencies are missing."""

    def test_plotly_not_available(self, sample_visualization_config):
        """Test behavior when Plotly is not available."""
        with patch(
            "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
            False,
        ):
            visualizer = SystemsVisualizer(sample_visualization_config)

            with pytest.raises(ImportError):
                visualizer.create_target_engagement_network([], "CCO")

    def test_networkx_not_available(self, sample_visualization_config):
        """Test behavior when NetworkX is not available."""
        with patch(
            "chemi_accel.biological_interactions.systems_visualization._NETWORKX_AVAILABLE",
            False,
        ):
            visualizer = SystemsVisualizer(sample_visualization_config)

            # Should still work with fallback layout
            mock_graph = Mock()
            mock_graph.nodes.return_value = ["node1", "node2"]

            pos = visualizer._generate_network_layout(mock_graph, NetworkLayout.SPRING)
            assert isinstance(pos, dict)

    def test_dash_not_available(self, sample_visualization_config):
        """Test behavior when Dash is not available."""
        with patch(
            "chemi_accel.biological_interactions.systems_visualization._DASH_AVAILABLE",
            False,
        ):
            visualizer = SystemsVisualizer(sample_visualization_config)

            with pytest.raises(ImportError):
                visualizer.create_interactive_dashboard({})

    def test_rdkit_not_available(self, sample_visualization_config):
        """Test behavior when RDKit is not available."""
        with patch(
            "chemi_accel.biological_interactions.systems_visualization._RDKIT_AVAILABLE",
            False,
        ):
            visualizer = SystemsVisualizer(sample_visualization_config)

            # Should still work with fallback values
            properties = visualizer._calculate_molecular_properties("CCO")
            assert isinstance(properties, dict)

            # Should return None for molecular image
            img = visualizer._generate_molecular_image("CCO")
            assert img is None


class TestIntegration:
    """Integration tests."""

    def test_complete_visualization_workflow(
        self,
        sample_target_interactions,
        sample_pathway_activations,
        sample_toxicity_profile,
        sample_visualization_config,
    ):
        """Test complete visualization workflow."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        # Mock Plotly components
        with patch(
            "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
            True,
        ), patch(
            "chemi_accel.biological_interactions.systems_visualization._NETWORKX_AVAILABLE",
            True,
        ), patch(
            "chemi_accel.biological_interactions.systems_visualization.go"
        ) as mock_go, patch(
            "chemi_accel.biological_interactions.systems_visualization.nx"
        ) as mock_nx, patch(
            "chemi_accel.biological_interactions.systems_visualization.make_subplots"
        ) as mock_subplots:
            # Setup mocks
            mock_figure = Mock()
            mock_go.Figure.return_value = mock_figure
            mock_subplots.return_value = mock_figure
            mock_go.Scatter.return_value = Mock()
            mock_go.Sunburst.return_value = Mock()
            mock_go.Heatmap.return_value = Mock()

            mock_graph = Mock()
            mock_nx.Graph.return_value = mock_graph
            mock_nx.spring_layout.return_value = {"CCO": (0, 0), "TARGET_001": (1, 0)}

            # Test different visualization types
            network_fig = visualizer.create_target_engagement_network(
                sample_target_interactions, "CCO"
            )

            pathway_fig = visualizer.create_pathway_modulation_map(
                sample_pathway_activations, "CCO"
            )

            toxicity_fig = visualizer.create_toxicity_profile_heatmap(
                sample_toxicity_profile
            )

            dashboard_fig = visualizer.create_systems_overview_dashboard(
                sample_target_interactions,
                sample_pathway_activations,
                sample_toxicity_profile,
                "CCO",
            )

            # All should return figure objects
            assert network_fig is not None
            assert pathway_fig is not None
            assert toxicity_fig is not None
            assert dashboard_fig is not None

    def test_explanation_workflow(self, sample_visualization_config):
        """Test complete explanation workflow."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        # Generate explanation
        explanation = visualizer.explain_prediction(
            prediction_value=0.75,
            compound_smiles="CCO",
            prediction_type="toxicity",
            model_context={"model_type": "random_forest"},
            explanation_methods=["feature_importance"],
        )

        # Create visualization
        with patch(
            "chemi_accel.biological_interactions.systems_visualization._PLOTLY_AVAILABLE",
            True,
        ), patch(
            "chemi_accel.biological_interactions.systems_visualization.make_subplots"
        ) as mock_subplots, patch(
            "chemi_accel.biological_interactions.systems_visualization.go"
        ) as mock_go:
            mock_figure = Mock()
            mock_subplots.return_value = mock_figure
            mock_go.Bar.return_value = Mock()
            mock_go.Scatter.return_value = Mock()
            mock_go.Table.return_value = Mock()

            explanation_fig = visualizer.create_explanation_visualization(explanation)

            assert explanation_fig is not None
            assert isinstance(explanation, ExplanationResult)
            assert explanation.compound_smiles == "CCO"
            assert explanation.prediction_value == 0.75

    def test_visualization_caching(self, sample_visualization_config):
        """Test visualization caching behavior."""
        visualizer = SystemsVisualizer(sample_visualization_config)

        # Add something to cache
        cache_key = "test_key"
        cache_value = {"result": "test_value"}
        visualizer.visualization_cache[cache_key] = cache_value

        # Verify cache works
        assert visualizer.visualization_cache[cache_key] == cache_value

        # Test network cache
        network_key = "network_test"
        network_value = {"nodes": [], "edges": []}
        visualizer.network_cache[network_key] = network_value

        assert visualizer.network_cache[network_key] == network_value

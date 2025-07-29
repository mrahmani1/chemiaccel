"""
Interactive Systems Visualization and Explanation Module for ChemiAccel.

This module provides comprehensive tools for visualizing biological systems
interactions, target engagement networks, pathway modulation, and system-level
impacts. It includes interactive visualizations and AI prediction explanation
tools to help users understand complex biological interactions.

Features:
- Interactive network visualizations for target engagement
- Pathway modulation visualization and analysis
- System-level impact assessment and visualization
- AI prediction explanation and interpretation tools
- Real-time interactive dashboards for biological systems
- Multi-scale visualization from molecular to systems level
- Export capabilities for publications and reports
- Integration with biological databases and knowledge graphs
"""

import base64
import json
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualizations will be limited.")

try:
    import networkx as nx

    _NETWORKX_AVAILABLE = True
except ImportError:
    _NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Network visualizations will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import FancyBboxPatch

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Static visualizations will be limited.")

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Molecular visualizations will be limited.")

try:
    import dash
    import dash_bootstrap_components as dbc
    import dash_cytoscape as cyto
    from dash import Input, Output, callback, dcc, html

    _DASH_AVAILABLE = True
except ImportError:
    _DASH_AVAILABLE = False
    warnings.warn("Dash not available. Interactive dashboards will be limited.")

from .multi_target_prediction import (
    BiologicalTarget,
    ConfidenceLevel,
    InteractionPrediction,
    TargetType,
)
from .pathway_modeling import (
    BiologicalPathway,
    CellularPhenotype,
    NetworkEdge,
    NetworkNode,
    PathwayActivation,
)
from .toxicology_prediction import (
    AdverseDrugReaction,
    OffTargetBinding,
    SystemsToxicologyProfile,
    ToxicityPrediction,
)


class VisualizationType(Enum):
    """Types of biological system visualizations."""

    TARGET_NETWORK = "target_network"
    PATHWAY_MAP = "pathway_map"
    TOXICITY_HEATMAP = "toxicity_heatmap"
    DOSE_RESPONSE = "dose_response"
    TIME_SERIES = "time_series"
    MOLECULAR_STRUCTURE = "molecular_structure"
    SYSTEMS_OVERVIEW = "systems_overview"
    INTERACTION_MATRIX = "interaction_matrix"
    SAFETY_PROFILE = "safety_profile"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


class NetworkLayout(Enum):
    """Network layout algorithms."""

    SPRING = "spring"
    CIRCULAR = "circular"
    HIERARCHICAL = "hierarchical"
    FORCE_DIRECTED = "force_directed"
    BIPARTITE = "bipartite"
    RADIAL = "radial"
    GRID = "grid"
    CUSTOM = "custom"


class ColorScheme(Enum):
    """Color schemes for visualizations."""

    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    BLUES = "blues"
    REDS = "reds"
    GREENS = "greens"
    RdYlBu = "RdYlBu"
    CUSTOM = "custom"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""

    width: int = 1200
    height: int = 800
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    layout: NetworkLayout = NetworkLayout.SPRING
    show_labels: bool = True
    show_legend: bool = True
    interactive: bool = True
    export_format: str = "html"
    theme: str = "plotly_white"
    font_size: int = 12
    node_size_range: Tuple[int, int] = (10, 50)
    edge_width_range: Tuple[int, int] = (1, 10)
    opacity: float = 0.8
    animation_duration: int = 500
    custom_colors: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        if not 0 <= self.opacity <= 1:
            raise ValueError("Opacity must be between 0 and 1")


@dataclass
class ExplanationResult:
    """Result of AI prediction explanation."""

    prediction_id: str
    compound_smiles: str
    prediction_value: float
    explanation_type: str
    feature_importance: Dict[str, float]
    local_explanations: Dict[str, Any]
    global_explanations: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]
    counterfactual_examples: List[Dict[str, Any]] = field(default_factory=list)
    similar_compounds: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    uncertainty_sources: List[str] = field(default_factory=list)


@dataclass
class NetworkVisualization:
    """Network visualization data structure."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    layout_positions: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any]
    interactive_elements: Dict[str, Any] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)


class SystemsVisualizer:
    """
    Comprehensive biological systems visualization engine.

    This class provides tools for creating interactive visualizations of
    biological systems, including target networks, pathway maps, toxicity
    profiles, and AI prediction explanations.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the systems visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()

        # Initialize visualization engines
        self._setup_visualization_engines()
        self._setup_color_schemes()
        self._setup_layout_algorithms()

        # Initialize explanation tools
        self._setup_explanation_engines()

        # Cache for generated visualizations
        self.visualization_cache = {}
        self.network_cache = {}

        # Dashboard components
        self.dashboard_app = None
        self.dashboard_components = {}

    def _setup_visualization_engines(self):
        """Setup visualization engines."""
        self.engines = {
            VisualizationType.TARGET_NETWORK: self._create_target_network,
            VisualizationType.PATHWAY_MAP: self._create_pathway_map,
            VisualizationType.TOXICITY_HEATMAP: self._create_toxicity_heatmap,
            VisualizationType.DOSE_RESPONSE: self._create_dose_response_plot,
            VisualizationType.TIME_SERIES: self._create_time_series_plot,
            VisualizationType.MOLECULAR_STRUCTURE: self._create_molecular_structure,
            VisualizationType.SYSTEMS_OVERVIEW: self._create_systems_overview,
            VisualizationType.INTERACTION_MATRIX: self._create_interaction_matrix,
            VisualizationType.SAFETY_PROFILE: self._create_safety_profile,
            VisualizationType.COMPARATIVE_ANALYSIS: self._create_comparative_analysis,
        }

    def _setup_color_schemes(self):
        """Setup color schemes."""
        self.color_schemes = {
            ColorScheme.VIRIDIS: px.colors.sequential.Viridis,
            ColorScheme.PLASMA: px.colors.sequential.Plasma,
            ColorScheme.INFERNO: px.colors.sequential.Inferno,
            ColorScheme.MAGMA: px.colors.sequential.Magma,
            ColorScheme.BLUES: px.colors.sequential.Blues,
            ColorScheme.REDS: px.colors.sequential.Reds,
            ColorScheme.GREENS: px.colors.sequential.Greens,
        }

    def _setup_layout_algorithms(self):
        """Setup network layout algorithms."""
        if _NETWORKX_AVAILABLE:
            self.layout_algorithms = {
                NetworkLayout.SPRING: nx.spring_layout,
                NetworkLayout.CIRCULAR: nx.circular_layout,
                NetworkLayout.HIERARCHICAL: nx.nx_agraph.graphviz_layout,
                NetworkLayout.FORCE_DIRECTED: nx.kamada_kawai_layout,
                NetworkLayout.RADIAL: nx.shell_layout,
            }
        else:
            self.layout_algorithms = {}

    def _setup_explanation_engines(self):
        """Setup AI explanation engines."""
        self.explanation_engines = {
            "feature_importance": self._calculate_feature_importance,
            "lime_explanation": self._generate_lime_explanation,
            "shap_explanation": self._generate_shap_explanation,
            "counterfactual": self._generate_counterfactual_explanations,
            "similarity_analysis": self._analyze_molecular_similarity,
        }

    def create_target_engagement_network(
        self,
        interactions: List[InteractionPrediction],
        compound_smiles: str,
        config: Optional[VisualizationConfig] = None,
    ) -> go.Figure:
        """
        Create interactive target engagement network visualization.

        Args:
            interactions: List of target interactions
            compound_smiles: Compound SMILES string
            config: Optional visualization configuration

        Returns:
            Plotly figure object
        """
        if not _PLOTLY_AVAILABLE or not _NETWORKX_AVAILABLE:
            raise ImportError("Plotly and NetworkX required for network visualization")

        config = config or self.config

        # Create network graph
        G = nx.Graph()

        # Add compound node
        G.add_node(
            compound_smiles,
            node_type="compound",
            label=self._get_compound_name(compound_smiles),
            size=30,
            color="red",
        )

        # Add target nodes and edges
        for interaction in interactions:
            target_id = interaction.target.target_id
            target_name = interaction.target.name or target_id

            G.add_node(
                target_id,
                node_type="target",
                label=target_name,
                target_type=interaction.target.target_type.value,
                size=max(10, int(interaction.interaction_probability * 40)),
                color=self._get_target_color(interaction.target.target_type),
            )

            # Add edge with interaction strength
            G.add_edge(
                compound_smiles,
                target_id,
                weight=interaction.interaction_probability,
                confidence=interaction.confidence_level.value,
                width=max(1, int(interaction.interaction_probability * 10)),
            )

        # Generate layout
        pos = self._generate_network_layout(G, config.layout)

        # Create traces
        edge_trace = self._create_network_edges(G, pos)
        node_trace = self._create_network_nodes(G, pos)

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"Target Engagement Network: {self._get_compound_name(compound_smiles)}",
                titlefont_size=16,
                showlegend=True,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Node size represents interaction probability",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=config.width,
                height=config.height,
                template=config.theme,
            ),
        )

        return fig

    def create_pathway_modulation_map(
        self,
        pathway_activations: List[PathwayActivation],
        compound_smiles: str,
        config: Optional[VisualizationConfig] = None,
    ) -> go.Figure:
        """
        Create pathway modulation visualization.

        Args:
            pathway_activations: List of pathway activations
            compound_smiles: Compound SMILES string
            config: Optional visualization configuration

        Returns:
            Plotly figure object
        """
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for pathway visualization")

        config = config or self.config

        # Prepare data
        pathway_names = [pa.pathway.name for pa in pathway_activations]
        activation_scores = [pa.activation_score for pa in pathway_activations]
        confidence_scores = [pa.confidence_score for pa in pathway_activations]
        pathway_types = [pa.pathway.pathway_type.value for pa in pathway_activations]

        # Create sunburst chart for hierarchical pathway organization
        fig = go.Figure(
            go.Sunburst(
                labels=pathway_names,
                values=activation_scores,
                parents=[""] * len(pathway_names),  # All pathways at root level
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>"
                + "Activation: %{value:.3f}<br>"
                + "<extra></extra>",
                maxdepth=2,
            )
        )

        fig.update_layout(
            title=f"Pathway Modulation Map: {self._get_compound_name(compound_smiles)}",
            width=config.width,
            height=config.height,
            template=config.theme,
        )

        return fig

    def create_toxicity_profile_heatmap(
        self,
        toxicity_profile: SystemsToxicologyProfile,
        config: Optional[VisualizationConfig] = None,
    ) -> go.Figure:
        """
        Create toxicity profile heatmap.

        Args:
            toxicity_profile: Systems toxicology profile
            config: Optional visualization configuration

        Returns:
            Plotly figure object
        """
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for heatmap visualization")

        config = config or self.config

        # Prepare toxicity data
        endpoints = []
        probabilities = []
        risk_levels = []

        for tox_pred in toxicity_profile.toxicity_predictions:
            endpoints.append(tox_pred.toxicity_endpoint.value.replace("_", " ").title())
            probabilities.append(tox_pred.probability)
            risk_levels.append(tox_pred.risk_level.value)

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=[probabilities],
                x=endpoints,
                y=["Toxicity Probability"],
                colorscale="Reds",
                zmin=0,
                zmax=1,
                hovertemplate="<b>%{x}</b><br>"
                + "Probability: %{z:.3f}<br>"
                + "<extra></extra>",
                colorbar=dict(title="Probability", titleside="right"),
            )
        )

        fig.update_layout(
            title=f"Toxicity Profile: {self._get_compound_name(toxicity_profile.compound_smiles)}",
            xaxis_title="Toxicity Endpoints",
            yaxis_title="",
            width=config.width,
            height=config.height // 2,  # Shorter for heatmap
            template=config.theme,
        )

        return fig

    def create_systems_overview_dashboard(
        self,
        interactions: List[InteractionPrediction],
        pathway_activations: List[PathwayActivation],
        toxicity_profile: SystemsToxicologyProfile,
        compound_smiles: str,
        config: Optional[VisualizationConfig] = None,
    ) -> go.Figure:
        """
        Create comprehensive systems overview dashboard.

        Args:
            interactions: Target interactions
            pathway_activations: Pathway activations
            toxicity_profile: Toxicity profile
            compound_smiles: Compound SMILES string
            config: Optional configuration

        Returns:
            Multi-panel dashboard figure
        """
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for dashboard creation")

        config = config or self.config

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Target Interactions",
                "Pathway Modulation",
                "Safety Profile",
                "Molecular Properties",
            ],
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "indicator"}, {"type": "table"}],
            ],
        )

        # Target interactions scatter plot
        target_names = [i.target.name or i.target.target_id for i in interactions]
        interaction_probs = [i.interaction_probability for i in interactions]
        confidence_levels = [i.confidence_level.value for i in interactions]

        fig.add_trace(
            go.Scatter(
                x=target_names[:10],  # Limit to top 10
                y=interaction_probs[:10],
                mode="markers+text",
                text=confidence_levels[:10],
                textposition="top center",
                marker=dict(
                    size=[p * 50 + 10 for p in interaction_probs[:10]],
                    color=interaction_probs[:10],
                    colorscale="Viridis",
                    showscale=False,
                ),
                name="Target Interactions",
            ),
            row=1,
            col=1,
        )

        # Pathway modulation bar chart
        pathway_names = [pa.pathway.name for pa in pathway_activations[:10]]
        activation_scores = [pa.activation_score for pa in pathway_activations[:10]]

        fig.add_trace(
            go.Bar(
                x=pathway_names,
                y=activation_scores,
                name="Pathway Activation",
                marker_color="lightblue",
            ),
            row=1,
            col=2,
        )

        # Safety score indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=toxicity_profile.overall_safety_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Safety Score"},
                delta={"reference": 0.8},
                gauge={
                    "axis": {"range": [None, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 0.5], "color": "lightgray"},
                        {"range": [0.5, 0.8], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 0.8,
                    },
                },
            ),
            row=2,
            col=1,
        )

        # Molecular properties table
        properties = self._calculate_molecular_properties(compound_smiles)
        fig.add_trace(
            go.Table(
                header=dict(values=["Property", "Value"]),
                cells=dict(
                    values=[
                        list(properties.keys()),
                        [
                            f"{v:.2f}" if isinstance(v, float) else str(v)
                            for v in properties.values()
                        ],
                    ]
                ),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=f"Systems Overview: {self._get_compound_name(compound_smiles)}",
            width=config.width,
            height=config.height,
            template=config.theme,
            showlegend=False,
        )

        return fig

    def explain_prediction(
        self,
        prediction_value: float,
        compound_smiles: str,
        prediction_type: str,
        model_context: Dict[str, Any],
        explanation_methods: Optional[List[str]] = None,
    ) -> ExplanationResult:
        """
        Generate comprehensive explanation for AI predictions.

        Args:
            prediction_value: The prediction to explain
            compound_smiles: Compound SMILES string
            prediction_type: Type of prediction (toxicity, interaction, etc.)
            model_context: Model context and parameters
            explanation_methods: Methods to use for explanation

        Returns:
            Comprehensive explanation result
        """
        explanation_methods = explanation_methods or [
            "feature_importance",
            "similarity_analysis",
        ]

        explanation_id = (
            f"{compound_smiles}_{prediction_type}_{hash(str(model_context))}"
        )

        # Initialize explanation result
        explanation = ExplanationResult(
            prediction_id=explanation_id,
            compound_smiles=compound_smiles,
            prediction_value=prediction_value,
            explanation_type=prediction_type,
            feature_importance={},
            local_explanations={},
            global_explanations={},
        )

        # Generate explanations using requested methods
        for method in explanation_methods:
            if method in self.explanation_engines:
                try:
                    result = self.explanation_engines[method](
                        compound_smiles, prediction_value, model_context
                    )
                    explanation.local_explanations[method] = result
                except Exception as e:
                    explanation.local_explanations[method] = {"error": str(e)}

        # Calculate feature importance
        explanation.feature_importance = (
            self._calculate_comprehensive_feature_importance(
                compound_smiles, prediction_type, model_context
            )
        )

        # Generate reasoning path
        explanation.reasoning_path = self._generate_reasoning_path(
            compound_smiles,
            prediction_value,
            prediction_type,
            explanation.feature_importance,
        )

        # Identify uncertainty sources
        explanation.uncertainty_sources = self._identify_uncertainty_sources(
            compound_smiles, prediction_value, model_context
        )

        return explanation

    def create_explanation_visualization(
        self,
        explanation: ExplanationResult,
        config: Optional[VisualizationConfig] = None,
    ) -> go.Figure:
        """
        Create visualization for AI prediction explanation.

        Args:
            explanation: Explanation result
            config: Optional visualization configuration

        Returns:
            Explanation visualization figure
        """
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for explanation visualization")

        config = config or self.config

        # Create subplots for different explanation components
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Feature Importance",
                "Molecular Structure",
                "Similar Compounds",
                "Reasoning Path",
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
        )

        # Feature importance bar chart
        features = list(explanation.feature_importance.keys())[:10]
        importance_values = list(explanation.feature_importance.values())[:10]

        fig.add_trace(
            go.Bar(
                x=importance_values,
                y=features,
                orientation="h",
                name="Feature Importance",
                marker_color="lightcoral",
            ),
            row=1,
            col=1,
        )

        # Molecular structure (placeholder)
        if _RDKIT_AVAILABLE:
            mol_img = self._generate_molecular_image(explanation.compound_smiles)
            if mol_img:
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode="markers",
                        marker=dict(size=1, opacity=0),
                        name="Molecule",
                    ),
                    row=1,
                    col=2,
                )

        # Similar compounds
        if explanation.similar_compounds:
            similarity_scores = [
                comp["similarity"] for comp in explanation.similar_compounds[:10]
            ]
            compound_names = [
                self._get_compound_name(comp["smiles"])
                for comp in explanation.similar_compounds[:10]
            ]

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(similarity_scores))),
                    y=similarity_scores,
                    mode="markers",
                    text=compound_names,
                    name="Similar Compounds",
                    marker=dict(size=10, color="lightblue"),
                ),
                row=2,
                col=1,
            )

        # Reasoning path table
        if explanation.reasoning_path:
            fig.add_trace(
                go.Table(
                    header=dict(values=["Step", "Reasoning"]),
                    cells=dict(
                        values=[
                            list(range(1, len(explanation.reasoning_path) + 1)),
                            explanation.reasoning_path,
                        ]
                    ),
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title=f"Prediction Explanation: {explanation.prediction_type}",
            width=config.width,
            height=config.height,
            template=config.theme,
            showlegend=False,
        )

        return fig

    def create_interactive_dashboard(
        self,
        compound_data: Dict[str, Any],
        config: Optional[VisualizationConfig] = None,
    ) -> str:
        """
        Create interactive dashboard using Dash.

        Args:
            compound_data: Comprehensive compound data
            config: Optional visualization configuration

        Returns:
            URL of the dashboard
        """
        if not _DASH_AVAILABLE:
            raise ImportError("Dash required for interactive dashboard")

        config = config or self.config

        # Initialize Dash app
        self.dashboard_app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
        )

        # Create layout
        self.dashboard_app.layout = self._create_dashboard_layout(compound_data, config)

        # Setup callbacks
        self._setup_dashboard_callbacks()

        # Run dashboard
        port = 8050
        host = "127.0.0.1"

        # Start dashboard in separate thread or process
        import threading

        dashboard_thread = threading.Thread(
            target=lambda: self.dashboard_app.run_server(
                host=host, port=port, debug=False
            )
        )
        dashboard_thread.daemon = True
        dashboard_thread.start()

        return f"http://{host}:{port}"

    def export_visualization(
        self,
        figure: go.Figure,
        filename: str,
        format: str = "html",
        config: Optional[VisualizationConfig] = None,
    ) -> str:
        """
        Export visualization to file.

        Args:
            figure: Plotly figure to export
            filename: Output filename
            format: Export format (html, png, pdf, svg)
            config: Optional configuration

        Returns:
            Path to exported file
        """
        config = config or self.config

        if format == "html":
            figure.write_html(filename)
        elif format == "png":
            figure.write_image(
                filename, format="png", width=config.width, height=config.height
            )
        elif format == "pdf":
            figure.write_image(
                filename, format="pdf", width=config.width, height=config.height
            )
        elif format == "svg":
            figure.write_image(
                filename, format="svg", width=config.width, height=config.height
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")

        return filename

    # Helper methods for visualization creation
    def _create_target_network(self, data: Dict[str, Any]) -> go.Figure:
        """Create target network visualization."""
        return self.create_target_engagement_network(
            data["interactions"], data["compound_smiles"]
        )

    def _create_pathway_map(self, data: Dict[str, Any]) -> go.Figure:
        """Create pathway map visualization."""
        return self.create_pathway_modulation_map(
            data["pathway_activations"], data["compound_smiles"]
        )

    def _create_toxicity_heatmap(self, data: Dict[str, Any]) -> go.Figure:
        """Create toxicity heatmap visualization."""
        return self.create_toxicity_profile_heatmap(data["toxicity_profile"])

    def _create_dose_response_plot(self, data: Dict[str, Any]) -> go.Figure:
        """Create dose-response plot."""
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Plotly required")

        # Placeholder implementation
        doses = data.get("doses", [0.1, 1, 10, 100])
        responses = data.get("responses", [0.1, 0.3, 0.7, 0.9])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=doses, y=responses, mode="lines+markers", name="Dose-Response")
        )

        fig.update_layout(
            title="Dose-Response Curve",
            xaxis_title="Dose (μM)",
            yaxis_title="Response",
            xaxis_type="log",
        )

        return fig

    def _create_time_series_plot(self, data: Dict[str, Any]) -> go.Figure:
        """Create time series plot."""
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Plotly required")

        # Placeholder implementation
        times = data.get("times", list(range(24)))
        values = data.get("values", np.random.randn(24).cumsum())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=values, mode="lines", name="Time Series"))

        fig.update_layout(
            title="Time Series Analysis",
            xaxis_title="Time (hours)",
            yaxis_title="Response",
        )

        return fig

    def _create_molecular_structure(self, data: Dict[str, Any]) -> go.Figure:
        """Create molecular structure visualization."""
        if not _RDKIT_AVAILABLE or not _PLOTLY_AVAILABLE:
            raise ImportError("RDKit and Plotly required")

        smiles = data["compound_smiles"]
        mol_img = self._generate_molecular_image(smiles)

        fig = go.Figure()
        if mol_img:
            fig.add_layout_image(
                dict(
                    source=mol_img,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1,
                    sizex=1,
                    sizey=1,
                    xanchor="left",
                    yanchor="top",
                )
            )

        fig.update_layout(
            title=f"Molecular Structure: {self._get_compound_name(smiles)}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        return fig

    def _create_systems_overview(self, data: Dict[str, Any]) -> go.Figure:
        """Create systems overview visualization."""
        return self.create_systems_overview_dashboard(
            data["interactions"],
            data["pathway_activations"],
            data["toxicity_profile"],
            data["compound_smiles"],
        )

    def _create_interaction_matrix(self, data: Dict[str, Any]) -> go.Figure:
        """Create interaction matrix visualization."""
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Plotly required")

        # Placeholder implementation
        targets = data.get("targets", ["Target A", "Target B", "Target C"])
        compounds = data.get("compounds", ["Compound 1", "Compound 2"])
        matrix = data.get("matrix", np.random.rand(len(compounds), len(targets)))

        fig = go.Figure(
            data=go.Heatmap(z=matrix, x=targets, y=compounds, colorscale="Viridis")
        )

        fig.update_layout(
            title="Compound-Target Interaction Matrix",
            xaxis_title="Targets",
            yaxis_title="Compounds",
        )

        return fig

    def _create_safety_profile(self, data: Dict[str, Any]) -> go.Figure:
        """Create safety profile visualization."""
        return self.create_toxicity_profile_heatmap(data["toxicity_profile"])

    def _create_comparative_analysis(self, data: Dict[str, Any]) -> go.Figure:
        """Create comparative analysis visualization."""
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Plotly required")

        # Placeholder implementation
        compounds = data.get("compounds", ["Compound A", "Compound B"])
        metrics = data.get("metrics", ["Efficacy", "Safety", "Selectivity"])
        values = data.get("values", np.random.rand(len(compounds), len(metrics)))

        fig = go.Figure()

        for i, compound in enumerate(compounds):
            fig.add_trace(
                go.Scatterpolar(
                    r=values[i], theta=metrics, fill="toself", name=compound
                )
            )

        fig.update_layout(
            title="Comparative Analysis",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        )

        return fig

    # Network visualization helpers
    def _generate_network_layout(
        self, G, layout_type: NetworkLayout
    ) -> Dict[str, Tuple[float, float]]:
        """Generate network layout positions."""
        if not _NETWORKX_AVAILABLE:
            # Simple circular layout as fallback
            nodes = list(G.nodes())
            n = len(nodes)
            pos = {}
            for i, node in enumerate(nodes):
                angle = 2 * np.pi * i / n
                pos[node] = (np.cos(angle), np.sin(angle))
            return pos

        layout_func = self.layout_algorithms.get(layout_type, nx.spring_layout)
        return layout_func(G)

    def _create_network_edges(self, G, pos) -> go.Scatter:
        """Create network edge trace."""
        edge_x = []
        edge_y = []
        edge_info = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            edge_data = G[edge[0]][edge[1]]
            edge_info.append(f"Weight: {edge_data.get('weight', 'N/A')}")

        return go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

    def _create_network_nodes(self, G, pos) -> go.Scatter:
        """Create network node trace."""
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        node_sizes = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            node_data = G.nodes[node]
            node_info.append(
                f"Label: {node_data.get('label', node)}<br>"
                + f"Type: {node_data.get('node_type', 'unknown')}<br>"
                + f"Size: {node_data.get('size', 10)}"
            )
            node_colors.append(node_data.get("color", "blue"))
            node_sizes.append(node_data.get("size", 10))

        return go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[G.nodes[node].get("label", node) for node in G.nodes()],
            hovertext=node_info,
            marker=dict(
                showscale=True, color=node_colors, size=node_sizes, line=dict(width=2)
            ),
        )

    # AI Explanation helpers
    def _calculate_feature_importance(
        self,
        compound_smiles: str,
        prediction_value: float,
        model_context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate feature importance for prediction."""
        # Placeholder implementation
        features = [
            "molecular_weight",
            "logp",
            "tpsa",
            "hbd_count",
            "hba_count",
            "aromatic_rings",
            "rotatable_bonds",
            "formal_charge",
        ]

        # Generate mock importance scores
        importance = {}
        for feature in features:
            importance[feature] = np.random.random()

        # Normalize to sum to 1
        total = sum(importance.values())
        for feature in importance:
            importance[feature] /= total

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def _generate_lime_explanation(
        self,
        compound_smiles: str,
        prediction_value: float,
        model_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate LIME explanation."""
        # Placeholder implementation
        return {
            "method": "LIME",
            "local_importance": self._calculate_feature_importance(
                compound_smiles, prediction_value, model_context
            ),
            "explanation_text": f"LIME analysis for prediction {prediction_value:.3f}",
        }

    def _generate_shap_explanation(
        self,
        compound_smiles: str,
        prediction_value: float,
        model_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate SHAP explanation."""
        # Placeholder implementation
        return {
            "method": "SHAP",
            "shap_values": np.random.randn(10).tolist(),
            "feature_names": ["feature_" + str(i) for i in range(10)],
            "explanation_text": f"SHAP analysis for prediction {prediction_value:.3f}",
        }

    def _generate_counterfactual_explanations(
        self,
        compound_smiles: str,
        prediction_value: float,
        model_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations."""
        # Placeholder implementation
        counterfactuals = []
        for i in range(3):
            counterfactuals.append(
                {
                    "modified_smiles": compound_smiles + "C",  # Simple modification
                    "predicted_value": prediction_value + np.random.normal(0, 0.1),
                    "modification_description": f"Added methyl group at position {i+1}",
                    "distance": np.random.random(),
                }
            )

        return counterfactuals

    def _analyze_molecular_similarity(
        self,
        compound_smiles: str,
        prediction_value: float,
        model_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze molecular similarity for explanation."""
        # Placeholder implementation
        similar_compounds = []
        for i in range(5):
            similar_compounds.append(
                {
                    "smiles": f"CC{compound_smiles}C{i}",  # Mock similar compound
                    "similarity": 0.9 - i * 0.1,
                    "prediction": prediction_value + np.random.normal(0, 0.05),
                    "name": f"Similar Compound {i+1}",
                }
            )

        return {
            "similar_compounds": similar_compounds,
            "similarity_method": "Tanimoto coefficient",
            "explanation_text": "Compounds with similar structures and predictions",
        }

    def _calculate_comprehensive_feature_importance(
        self, compound_smiles: str, prediction_type: str, model_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive feature importance."""
        # Combine multiple importance calculation methods
        importance_methods = ["structural", "physicochemical", "topological"]
        combined_importance = {}

        for method in importance_methods:
            method_importance = self._calculate_method_specific_importance(
                compound_smiles, method, model_context
            )

            for feature, score in method_importance.items():
                if feature in combined_importance:
                    combined_importance[feature] += score / len(importance_methods)
                else:
                    combined_importance[feature] = score / len(importance_methods)

        return dict(
            sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        )

    def _calculate_method_specific_importance(
        self, compound_smiles: str, method: str, model_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate importance for specific method."""
        # Placeholder implementation
        if method == "structural":
            features = [
                "aromatic_rings",
                "heteroatoms",
                "ring_systems",
                "functional_groups",
            ]
        elif method == "physicochemical":
            features = ["molecular_weight", "logp", "tpsa", "solubility"]
        else:  # topological
            features = ["connectivity", "branching", "cycles", "path_length"]

        return {feature: np.random.random() for feature in features}

    def _generate_reasoning_path(
        self,
        compound_smiles: str,
        prediction_value: float,
        prediction_type: str,
        feature_importance: Dict[str, float],
    ) -> List[str]:
        """Generate reasoning path for prediction."""
        reasoning = []

        # Start with the prediction
        reasoning.append(f"Prediction: {prediction_type} = {prediction_value:.3f}")

        # Add top contributing features
        top_features = list(feature_importance.items())[:3]
        for feature, importance in top_features:
            reasoning.append(
                f"{feature.replace('_', ' ').title()} contributes {importance:.1%} to the prediction"
            )

        # Add structural insights
        if _RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(compound_smiles)
            if mol:
                mw = Chem.Descriptors.MolWt(mol)
                if mw > 500:
                    reasoning.append("High molecular weight may affect bioavailability")
                elif mw < 200:
                    reasoning.append(
                        "Low molecular weight suggests good oral absorption"
                    )

        # Add confidence assessment
        reasoning.append("Prediction confidence based on model training data coverage")

        return reasoning

    def _identify_uncertainty_sources(
        self,
        compound_smiles: str,
        prediction_value: float,
        model_context: Dict[str, Any],
    ) -> List[str]:
        """Identify sources of prediction uncertainty."""
        uncertainty_sources = []

        # Model-related uncertainties
        if model_context.get("training_size", 1000) < 500:
            uncertainty_sources.append("Limited training data size")

        if model_context.get("model_performance", {}).get("r2", 0.8) < 0.7:
            uncertainty_sources.append("Model performance limitations")

        # Compound-related uncertainties
        if _RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(compound_smiles)
            if mol:
                # Check for unusual structural features
                mw = Chem.Descriptors.MolWt(mol)
                if mw > 800 or mw < 100:
                    uncertainty_sources.append(
                        "Molecular weight outside typical drug range"
                    )

                logp = Chem.Descriptors.MolLogP(mol)
                if logp > 5 or logp < -2:
                    uncertainty_sources.append("LogP value outside typical range")

        # Prediction-related uncertainties
        if abs(prediction_value - 0.5) > 0.45:  # Very high or very low predictions
            uncertainty_sources.append("Prediction at extreme end of scale")

        return uncertainty_sources

    # Utility methods
    def _get_compound_name(self, smiles: str) -> str:
        """Get compound name from SMILES."""
        # Placeholder - would use chemical databases in real implementation
        return f"Compound_{hash(smiles) % 10000}"

    def _get_target_color(self, target_type: TargetType) -> str:
        """Get color for target type."""
        colors = {
            TargetType.KINASE: "red",
            TargetType.GPCR: "blue",
            TargetType.ION_CHANNEL: "green",
            TargetType.ENZYME: "orange",
            TargetType.TRANSPORTER: "purple",
            TargetType.NUCLEAR_RECEPTOR: "brown",
        }
        return colors.get(target_type, "gray")

    def _generate_molecular_image(self, smiles: str) -> Optional[str]:
        """Generate molecular image as base64 string."""
        if not _RDKIT_AVAILABLE:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Generate 2D coordinates
            rdDepictor.Compute2DCoords(mol)

            # Create image
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()

            # Convert to base64
            img_data = drawer.GetDrawingText()
            img_b64 = base64.b64encode(img_data).decode()
            return f"data:image/png;base64,{img_b64}"

        except Exception:
            return None

    def _calculate_molecular_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular properties."""
        if not _RDKIT_AVAILABLE:
            return {
                "Molecular Weight": 300.0,
                "LogP": 2.5,
                "TPSA": 60.0,
                "HBD Count": 2,
                "HBA Count": 3,
            }

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            return {
                "Molecular Weight": Chem.Descriptors.MolWt(mol),
                "LogP": Chem.Descriptors.MolLogP(mol),
                "TPSA": Chem.Descriptors.TPSA(mol),
                "HBD Count": Chem.Descriptors.NumHDonors(mol),
                "HBA Count": Chem.Descriptors.NumHAcceptors(mol),
                "Rotatable Bonds": Chem.Descriptors.NumRotatableBonds(mol),
                "Aromatic Rings": Chem.Descriptors.NumAromaticRings(mol),
            }
        except Exception:
            return {}

    def _create_dashboard_layout(
        self, compound_data: Dict[str, Any], config: VisualizationConfig
    ):
        """Create dashboard layout."""
        # Placeholder implementation for Dash layout
        return html.Div(
            [html.H1("Biological Systems Dashboard"), html.Div(id="content")]
        )

    def _setup_dashboard_callbacks(self):
        """Setup Dash callbacks."""
        # Placeholder for callback setup
        pass


# Utility functions
def create_systems_visualization(
    visualization_type: VisualizationType,
    data: Dict[str, Any],
    config: Optional[VisualizationConfig] = None,
) -> go.Figure:
    """
    Create biological systems visualization.

    Args:
        visualization_type: Type of visualization to create
        data: Data for visualization
        config: Optional configuration

    Returns:
        Plotly figure object
    """
    visualizer = SystemsVisualizer(config)
    engine = visualizer.engines.get(visualization_type)

    if not engine:
        raise ValueError(f"Unsupported visualization type: {visualization_type}")

    return engine(data)


def explain_biological_prediction(
    prediction_value: float,
    compound_smiles: str,
    prediction_type: str,
    model_context: Dict[str, Any],
    config: Optional[VisualizationConfig] = None,
) -> Tuple[ExplanationResult, go.Figure]:
    """
    Explain biological prediction with visualization.

    Args:
        prediction_value: Prediction to explain
        compound_smiles: Compound SMILES
        prediction_type: Type of prediction
        model_context: Model context
        config: Optional configuration

    Returns:
        Explanation result and visualization figure
    """
    visualizer = SystemsVisualizer(config)

    explanation = visualizer.explain_prediction(
        prediction_value, compound_smiles, prediction_type, model_context
    )

    explanation_viz = visualizer.create_explanation_visualization(explanation)

    return explanation, explanation_viz


def create_interactive_systems_dashboard(
    compound_data: Dict[str, Any], config: Optional[VisualizationConfig] = None
) -> str:
    """
    Create interactive systems dashboard.

    Args:
        compound_data: Comprehensive compound data
        config: Optional configuration

    Returns:
        Dashboard URL
    """
    visualizer = SystemsVisualizer(config)
    return visualizer.create_interactive_dashboard(compound_data)


# Exceptions
class SystemsVisualizationError(Exception):
    """Base exception for systems visualization errors."""

    pass


class VisualizationCreationError(SystemsVisualizationError):
    """Exception raised during visualization creation."""

    pass


class ExplanationGenerationError(SystemsVisualizationError):
    """Exception raised during explanation generation."""

    pass


class DashboardCreationError(SystemsVisualizationError):
    """Exception raised during dashboard creation."""

    pass

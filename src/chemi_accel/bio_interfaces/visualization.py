"""
Interactive Visualization of Bio-Interfaces

This module creates interactive visualizations of molecular interactions
at bio-interfaces, predicted cellular responses, and surface properties
driving specific biological outcomes.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import plotly.colors as colors
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, State, callback, dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from .biological_response import BiologicalResponseCharacterizer
from .integration import IntegratedRiskAssessment, ToxicologyLifecycleIntegrator
from .inverse_design import BioOptimizedSurfaceDesigner, OptimizedSurface
from .simulation import (
    BioInterfaceSimulator,
    CellAdhesionResult,
    ProteinSurfaceInteraction,
)

logger = logging.getLogger(__name__)


class VisualizationType(Enum):
    """Types of bio-interface visualizations."""

    MOLECULAR_INTERACTIONS = "molecular_interactions"
    CELLULAR_RESPONSE = "cellular_response"
    SURFACE_PROPERTIES = "surface_properties"
    BIOCOMPATIBILITY_HEATMAP = "biocompatibility_heatmap"
    DEGRADATION_TIMELINE = "degradation_timeline"
    OPTIMIZATION_TRADEOFFS = "optimization_tradeoffs"
    RISK_ASSESSMENT = "risk_assessment"
    PROTEIN_ADSORPTION = "protein_adsorption"
    CELL_ADHESION = "cell_adhesion"
    BIOFILM_FORMATION = "biofilm_formation"


@dataclass
class VisualizationConfig:
    """Configuration for bio-interface visualizations."""

    visualization_type: VisualizationType
    interactive: bool = True
    animation: bool = False
    color_scheme: str = "viridis"
    plot_style: str = "plotly"  # plotly, matplotlib, seaborn
    output_format: str = "html"  # html, svg, png, pdf
    resolution: int = 300  # DPI for static images
    width: int = 800
    height: int = 600


@dataclass
class InteractiveFeatures:
    """Interactive features configuration."""

    hover_tooltips: bool = True
    zoom_pan: bool = True
    selection_tools: bool = True
    animation_controls: bool = False
    parameter_sliders: bool = True
    dropdown_filters: bool = True
    cross_filtering: bool = True
    export_data: bool = True


class BioInterfaceVisualizer:
    """
    Interactive visualizer for bio-interface analysis and design.

    Creates comprehensive visualizations of molecular interactions,
    cellular responses, surface properties, and optimization results.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        interactive_features: Optional[InteractiveFeatures] = None,
    ):
        """
        Initialize bio-interface visualizer.

        Args:
            config: Visualization configuration
            interactive_features: Interactive features configuration
        """
        self.config = config or VisualizationConfig(
            visualization_type=VisualizationType.MOLECULAR_INTERACTIONS
        )
        self.interactive_features = interactive_features or InteractiveFeatures()

        # Check library availability
        self._check_dependencies()

        # Initialize visualization components
        self._initialize_color_schemes()
        self._initialize_plot_templates()

        logger.info("Initialized BioInterfaceVisualizer")

    def _check_dependencies(self) -> None:
        """Check availability of visualization libraries."""
        if not PLOTLY_AVAILABLE and self.config.plot_style == "plotly":
            logger.warning("Plotly not available, falling back to matplotlib")
            self.config.plot_style = "matplotlib"

        if not MATPLOTLIB_AVAILABLE and self.config.plot_style == "matplotlib":
            logger.warning("Matplotlib not available, using basic visualization")
            self.config.plot_style = "basic"

        if not DASH_AVAILABLE and self.config.interactive:
            logger.warning("Dash not available, disabling interactive features")
            self.config.interactive = False

    def _initialize_color_schemes(self) -> None:
        """Initialize color schemes for different visualization types."""
        self.color_schemes = {
            "biocompatibility": [
                "#d73027",
                "#f46d43",
                "#fdae61",
                "#fee08b",
                "#e6f598",
                "#abdda4",
                "#66c2a5",
                "#3288bd",
            ],
            "toxicity": [
                "#2c7bb6",
                "#abd9e9",
                "#e0f3f8",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
            ],
            "cellular_response": [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
            ],
            "protein_types": [
                "#e41a1c",
                "#377eb8",
                "#4daf4a",
                "#984ea3",
                "#ff7f00",
                "#ffff33",
            ],
            "surface_properties": [
                "#8dd3c7",
                "#ffffb3",
                "#bebada",
                "#fb8072",
                "#80b1d3",
                "#fdb462",
            ],
        }

    def _initialize_plot_templates(self) -> None:
        """Initialize plot templates and styles."""
        if PLOTLY_AVAILABLE:
            self.plotly_template = {
                "layout": {
                    "font": {"family": "Arial, sans-serif", "size": 12},
                    "colorway": self.color_schemes["biocompatibility"],
                    "paper_bgcolor": "white",
                    "plot_bgcolor": "white",
                    "margin": {"l": 60, "r": 60, "t": 60, "b": 60},
                }
            }

    def visualize_molecular_interactions(
        self,
        interaction_data: Union[ProteinSurfaceInteraction, Dict[str, Any]],
        include_animation: bool = True,
    ) -> Union[go.Figure, str]:
        """
        Visualize molecular interactions at bio-interfaces.

        Args:
            interaction_data: Protein-surface interaction data
            include_animation: Whether to include animation

        Returns:
            Interactive visualization figure or HTML string
        """
        logger.info("Creating molecular interactions visualization")

        if isinstance(interaction_data, ProteinSurfaceInteraction):
            data = self._extract_interaction_data(interaction_data)
        else:
            data = interaction_data

        if self.config.plot_style == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_molecular_interactions(data, include_animation)
        elif self.config.plot_style == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_molecular_interactions(
                data, include_animation
            )
        else:
            return self._create_basic_molecular_interactions(data)

    def visualize_cellular_response(
        self,
        response_data: Union[CellAdhesionResult, Dict[str, Any]],
        time_series: bool = True,
    ) -> Union[go.Figure, str]:
        """
        Visualize cellular response to bio-interfaces.

        Args:
            response_data: Cell adhesion or response data
            time_series: Whether to show time series data

        Returns:
            Interactive visualization figure or HTML string
        """
        logger.info("Creating cellular response visualization")

        if isinstance(response_data, CellAdhesionResult):
            data = self._extract_cellular_data(response_data)
        else:
            data = response_data

        if self.config.plot_style == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_cellular_response(data, time_series)
        elif self.config.plot_style == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_cellular_response(data, time_series)
        else:
            return self._create_basic_cellular_response(data)

    def visualize_surface_properties(
        self, surface_data: Dict[str, Any], property_correlations: bool = True
    ) -> Union[go.Figure, str]:
        """
        Visualize surface properties and their biological impact.

        Args:
            surface_data: Surface properties and biological outcomes
            property_correlations: Whether to show property correlations

        Returns:
            Interactive visualization figure or HTML string
        """
        logger.info("Creating surface properties visualization")

        if self.config.plot_style == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_surface_properties(
                surface_data, property_correlations
            )
        elif self.config.plot_style == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_surface_properties(
                surface_data, property_correlations
            )
        else:
            return self._create_basic_surface_properties(surface_data)

    def create_biocompatibility_heatmap(
        self, materials_data: List[Dict[str, Any]], biological_endpoints: List[str]
    ) -> Union[go.Figure, str]:
        """
        Create biocompatibility heatmap for multiple materials.

        Args:
            materials_data: List of material biocompatibility data
            biological_endpoints: Biological endpoints to display

        Returns:
            Interactive heatmap figure or HTML string
        """
        logger.info("Creating biocompatibility heatmap")

        # Prepare data matrix
        heatmap_data = self._prepare_heatmap_data(materials_data, biological_endpoints)

        if self.config.plot_style == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_heatmap(heatmap_data, biological_endpoints)
        elif self.config.plot_style == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_heatmap(heatmap_data, biological_endpoints)
        else:
            return self._create_basic_heatmap(heatmap_data, biological_endpoints)

    def visualize_optimization_tradeoffs(
        self, optimization_results: List[OptimizedSurface], objectives: List[str]
    ) -> Union[go.Figure, str]:
        """
        Visualize optimization tradeoffs for bio-optimized surfaces.

        Args:
            optimization_results: List of optimized surface designs
            objectives: Biological objectives to compare

        Returns:
            Interactive tradeoff visualization or HTML string
        """
        logger.info("Creating optimization tradeoffs visualization")

        # Prepare tradeoff data
        tradeoff_data = self._prepare_tradeoff_data(optimization_results, objectives)

        if self.config.plot_style == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_tradeoffs(tradeoff_data, objectives)
        elif self.config.plot_style == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_tradeoffs(tradeoff_data, objectives)
        else:
            return self._create_basic_tradeoffs(tradeoff_data, objectives)

    def visualize_risk_assessment(
        self, risk_data: IntegratedRiskAssessment
    ) -> Union[go.Figure, str]:
        """
        Visualize integrated risk assessment results.

        Args:
            risk_data: Integrated risk assessment data

        Returns:
            Interactive risk visualization or HTML string
        """
        logger.info("Creating risk assessment visualization")

        # Extract risk components
        risk_components = self._extract_risk_data(risk_data)

        if self.config.plot_style == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_risk_assessment(risk_components)
        elif self.config.plot_style == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_risk_assessment(risk_components)
        else:
            return self._create_basic_risk_assessment(risk_components)

    def create_interactive_dashboard(
        self, data_sources: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create comprehensive interactive dashboard.

        Args:
            data_sources: Dictionary of data sources for dashboard

        Returns:
            Dashboard URL if successful, None otherwise
        """
        if not DASH_AVAILABLE:
            logger.warning("Dash not available, cannot create interactive dashboard")
            return None

        logger.info("Creating interactive bio-interface dashboard")

        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Define dashboard layout
        app.layout = self._create_dashboard_layout(data_sources)

        # Register callbacks
        self._register_dashboard_callbacks(app, data_sources)

        # Run dashboard
        port = 8050
        app.run_server(debug=False, port=port, host="127.0.0.1")

        return f"http://127.0.0.1:{port}"

    # Plotly implementations
    def _create_plotly_molecular_interactions(
        self, data: Dict[str, Any], include_animation: bool
    ) -> go.Figure:
        """Create Plotly molecular interactions visualization."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "3D Molecular Structure",
                "Binding Energy Profile",
                "Contact Area Evolution",
                "Orientation Analysis",
            ),
            specs=[
                [{"type": "scatter3d"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
        )

        # 3D molecular structure
        if "coordinates" in data:
            coords = data["coordinates"]
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers+lines",
                    marker=dict(size=5, color=data.get("atom_types", "blue")),
                    name="Protein Structure",
                ),
                row=1,
                col=1,
            )

        # Binding energy profile
        if "energy_profile" in data:
            time_points = data["energy_profile"]["time"]
            energies = data["energy_profile"]["energy"]

            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=energies,
                    mode="lines+markers",
                    name="Binding Energy",
                    line=dict(color="red", width=2),
                ),
                row=1,
                col=2,
            )

        # Contact area evolution
        if "contact_area" in data:
            time_points = data["contact_area"]["time"]
            areas = data["contact_area"]["area"]

            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=areas,
                    mode="lines+markers",
                    name="Contact Area",
                    line=dict(color="green", width=2),
                    fill="tonexty",
                ),
                row=2,
                col=1,
            )

        # Orientation analysis
        if "orientation_angles" in data:
            angles = data["orientation_angles"]
            angle_names = ["Theta", "Phi", "Psi"]

            fig.add_trace(
                go.Bar(
                    x=angle_names,
                    y=angles,
                    name="Orientation Angles",
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title="Molecular Interactions at Bio-Interface",
            showlegend=True,
            height=800,
            **self.plotly_template["layout"],
        )

        return fig

    def _create_plotly_cellular_response(
        self, data: Dict[str, Any], time_series: bool
    ) -> go.Figure:
        """Create Plotly cellular response visualization."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Cell Adhesion Kinetics",
                "Spreading Area",
                "Viability Profile",
                "Proliferation Rate",
            ),
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
        )

        # Cell adhesion kinetics
        if "adhesion_kinetics" in data and time_series:
            time_points = data["adhesion_kinetics"]["time"]
            adhesion_strength = data["adhesion_kinetics"]["strength"]

            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=adhesion_strength,
                    mode="lines+markers",
                    name="Adhesion Strength",
                    line=dict(color="blue", width=2),
                ),
                row=1,
                col=1,
            )

        # Spreading area
        if "spreading_kinetics" in data:
            time_points = data["spreading_kinetics"]["time_points"]
            areas = data["spreading_kinetics"]["areas"]

            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=areas,
                    mode="lines+markers",
                    name="Spreading Area",
                    line=dict(color="orange", width=2),
                    fill="tozeroy",
                ),
                row=1,
                col=2,
            )

        # Viability profile
        if "viability_profile" in data:
            time_points = list(range(len(data["viability_profile"])))
            viability = data["viability_profile"]

            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=viability,
                    mode="lines+markers",
                    name="Cell Viability",
                    line=dict(color="green", width=2),
                ),
                row=2,
                col=1,
            )

        # Proliferation rate
        if "proliferation_metrics" in data:
            metrics = data["proliferation_metrics"]
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())

            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name="Proliferation Metrics",
                    marker_color=self.color_schemes["cellular_response"][
                        : len(metric_names)
                    ],
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Cellular Response to Bio-Interface",
            showlegend=True,
            height=800,
            **self.plotly_template["layout"],
        )

        return fig

    def _create_plotly_surface_properties(
        self, surface_data: Dict[str, Any], property_correlations: bool
    ) -> go.Figure:
        """Create Plotly surface properties visualization."""
        if property_correlations and "correlation_matrix" in surface_data:
            # Create correlation heatmap
            corr_matrix = surface_data["correlation_matrix"]
            property_names = surface_data["property_names"]

            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix,
                    x=property_names,
                    y=property_names,
                    colorscale="RdBu",
                    zmid=0,
                    text=np.round(corr_matrix, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z}<extra></extra>",
                )
            )

            fig.update_layout(
                title="Surface Property Correlations with Biological Response",
                xaxis_title="Surface Properties",
                yaxis_title="Surface Properties",
                **self.plotly_template["layout"],
            )
        else:
            # Create property impact visualization
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Property Impact on Biocompatibility",
                    "Surface Energy Distribution",
                    "Wettability vs Cell Adhesion",
                    "Roughness vs Protein Adsorption",
                ),
            )

            # Property impact
            if "property_impact" in surface_data:
                properties = list(surface_data["property_impact"].keys())
                impacts = list(surface_data["property_impact"].values())

                fig.add_trace(
                    go.Bar(
                        x=properties,
                        y=impacts,
                        name="Property Impact",
                        marker_color=self.color_schemes["surface_properties"][
                            : len(properties)
                        ],
                    ),
                    row=1,
                    col=1,
                )

        return fig

    def _create_plotly_heatmap(
        self, heatmap_data: np.ndarray, endpoints: List[str]
    ) -> go.Figure:
        """Create Plotly biocompatibility heatmap."""
        material_names = [f"Material_{i+1}" for i in range(heatmap_data.shape[0])]

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=endpoints,
                y=material_names,
                colorscale=self.color_schemes["biocompatibility"],
                colorbar=dict(title="Biocompatibility Score"),
                text=np.round(heatmap_data, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Score: %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Material Biocompatibility Heatmap",
            xaxis_title="Biological Endpoints",
            yaxis_title="Materials",
            height=600,
            **self.plotly_template["layout"],
        )

        return fig

    def _create_plotly_tradeoffs(
        self, tradeoff_data: Dict[str, Any], objectives: List[str]
    ) -> go.Figure:
        """Create Plotly optimization tradeoffs visualization."""
        if len(objectives) >= 3:
            # 3D scatter plot for multi-objective tradeoffs
            fig = go.Figure(
                data=go.Scatter3d(
                    x=tradeoff_data[objectives[0]],
                    y=tradeoff_data[objectives[1]],
                    z=tradeoff_data[objectives[2]],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=tradeoff_data.get(
                            "pareto_rank", [1] * len(tradeoff_data[objectives[0]])
                        ),
                        colorscale="Viridis",
                        colorbar=dict(title="Pareto Rank"),
                        line=dict(width=1, color="black"),
                    ),
                    text=tradeoff_data.get("surface_ids", []),
                    hovertemplate="<b>%{text}</b><br>"
                    + f"{objectives[0]}: %{{x}}<br>"
                    + f"{objectives[1]}: %{{y}}<br>"
                    + f"{objectives[2]}: %{{z}}<extra></extra>",
                )
            )

            fig.update_layout(
                title="Multi-Objective Optimization Tradeoffs",
                scene=dict(
                    xaxis_title=objectives[0],
                    yaxis_title=objectives[1],
                    zaxis_title=objectives[2],
                ),
                **self.plotly_template["layout"],
            )
        else:
            # 2D scatter plot for two objectives
            fig = go.Figure(
                data=go.Scatter(
                    x=tradeoff_data[objectives[0]],
                    y=tradeoff_data[objectives[1]],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=tradeoff_data.get(
                            "pareto_rank", [1] * len(tradeoff_data[objectives[0]])
                        ),
                        colorscale="Viridis",
                        colorbar=dict(title="Pareto Rank"),
                        line=dict(width=1, color="black"),
                    ),
                    text=tradeoff_data.get("surface_ids", []),
                    hovertemplate="<b>%{text}</b><br>"
                    + f"{objectives[0]}: %{{x}}<br>"
                    + f"{objectives[1]}: %{{y}}<extra></extra>",
                )
            )

            fig.update_layout(
                title="Bi-Objective Optimization Tradeoffs",
                xaxis_title=objectives[0],
                yaxis_title=objectives[1],
                **self.plotly_template["layout"],
            )

        return fig

    def _create_plotly_risk_assessment(
        self, risk_components: Dict[str, Any]
    ) -> go.Figure:
        """Create Plotly risk assessment visualization."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Risk by Endpoint",
                "Exposure Scenarios",
                "Safety Margins",
                "Regulatory Compliance",
            ),
            specs=[
                [{"type": "bar"}, {"type": "sunburst"}],
                [{"type": "scatter"}, {"type": "indicator"}],
            ],
        )

        # Risk by endpoint
        if "endpoint_risks" in risk_components:
            endpoints = list(risk_components["endpoint_risks"].keys())
            risks = list(risk_components["endpoint_risks"].values())

            fig.add_trace(
                go.Bar(
                    x=endpoints,
                    y=risks,
                    name="Risk Scores",
                    marker_color=self.color_schemes["toxicity"][: len(endpoints)],
                ),
                row=1,
                col=1,
            )

        # Safety margins
        if "safety_margins" in risk_components:
            endpoints = list(risk_components["safety_margins"].keys())
            margins = list(risk_components["safety_margins"].values())

            fig.add_trace(
                go.Scatter(
                    x=endpoints,
                    y=margins,
                    mode="markers",
                    marker=dict(
                        size=[max(5, min(20, m / 10)) for m in margins],
                        color="green",
                        line=dict(width=1, color="black"),
                    ),
                    name="Safety Margins",
                ),
                row=2,
                col=1,
            )

        # Regulatory compliance indicator
        if "compliance_score" in risk_components:
            compliance = risk_components["compliance_score"]

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=compliance * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Regulatory Compliance (%)"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "gray"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Integrated Risk Assessment",
            showlegend=True,
            height=800,
            **self.plotly_template["layout"],
        )

        return fig

    # Data extraction and preparation methods
    def _extract_interaction_data(
        self, interaction: ProteinSurfaceInteraction
    ) -> Dict[str, Any]:
        """Extract data from ProteinSurfaceInteraction for visualization."""
        # Generate synthetic coordinate data for visualization
        n_atoms = 1000
        coordinates = np.random.rand(n_atoms, 3) * 20

        # Generate energy profile
        time_points = np.linspace(0, 100, 50)
        energy_profile = {
            "time": time_points.tolist(),
            "energy": [
                interaction.binding_energy * (1 - np.exp(-t / 20)) for t in time_points
            ],
        }

        # Generate contact area evolution
        contact_area = {
            "time": time_points.tolist(),
            "area": [
                interaction.contact_area * (1 - np.exp(-t / 15)) for t in time_points
            ],
        }

        return {
            "coordinates": coordinates,
            "atom_types": ["C", "N", "O"] * (n_atoms // 3),
            "energy_profile": energy_profile,
            "contact_area": contact_area,
            "orientation_angles": list(interaction.orientation_angles),
            "binding_energy": interaction.binding_energy,
            "interaction_residues": interaction.interaction_residues,
        }

    def _extract_cellular_data(self, response: CellAdhesionResult) -> Dict[str, Any]:
        """Extract data from CellAdhesionResult for visualization."""
        # Generate time series data
        time_points = np.linspace(0, 24, 25)  # 24 hours

        adhesion_kinetics = {
            "time": time_points.tolist(),
            "strength": [
                response.adhesion_strength * (1 - np.exp(-t / 4)) for t in time_points
            ],
        }

        return {
            "adhesion_kinetics": adhesion_kinetics,
            "spreading_kinetics": response.spreading_kinetics,
            "viability_profile": response.viability_over_time,
            "proliferation_metrics": {
                "Focal Adhesions": response.focal_adhesion_count,
                "Migration Velocity": response.migration_velocity,
                "Viability Score": response.viability_score,
            },
            "cell_type": response.cell_type,
            "surface_properties": response.surface_properties,
        }

    def _prepare_heatmap_data(
        self, materials_data: List[Dict[str, Any]], endpoints: List[str]
    ) -> np.ndarray:
        """Prepare data matrix for biocompatibility heatmap."""
        n_materials = len(materials_data)
        n_endpoints = len(endpoints)

        heatmap_matrix = np.zeros((n_materials, n_endpoints))

        for i, material in enumerate(materials_data):
            for j, endpoint in enumerate(endpoints):
                # Extract biocompatibility score for endpoint
                score = material.get("biocompatibility", {}).get(endpoint, 0.5)
                heatmap_matrix[i, j] = score

        return heatmap_matrix

    def _prepare_tradeoff_data(
        self, optimization_results: List[OptimizedSurface], objectives: List[str]
    ) -> Dict[str, List[float]]:
        """Prepare data for tradeoff visualization."""
        tradeoff_data = {obj: [] for obj in objectives}
        tradeoff_data["surface_ids"] = []
        tradeoff_data["pareto_rank"] = []

        for i, surface in enumerate(optimization_results):
            for obj in objectives:
                score = surface.predicted_performance.get(obj, 0.0)
                tradeoff_data[obj].append(score)

            tradeoff_data["surface_ids"].append(surface.surface_id)
            tradeoff_data["pareto_rank"].append(i + 1)  # Simplified ranking

        return tradeoff_data

    def _extract_risk_data(
        self, risk_assessment: IntegratedRiskAssessment
    ) -> Dict[str, Any]:
        """Extract risk data for visualization."""
        # Aggregate endpoint risks from all assessments
        endpoint_risks = {}
        safety_margins = {}

        for assessment in risk_assessment.toxicity_assessments:
            for endpoint, risk in assessment.endpoint_predictions.items():
                if endpoint.value not in endpoint_risks:
                    endpoint_risks[endpoint.value] = []
                    safety_margins[endpoint.value] = []

                endpoint_risks[endpoint.value].append(risk)
                safety_margins[endpoint.value].append(
                    assessment.safety_margins[endpoint]
                )

        # Average risks and margins
        avg_endpoint_risks = {
            endpoint: np.mean(risks) for endpoint, risks in endpoint_risks.items()
        }
        avg_safety_margins = {
            endpoint: np.mean(margins) for endpoint, margins in safety_margins.items()
        }

        # Calculate compliance score
        compliance_scores = list(risk_assessment.regulatory_compliance.values())
        compliance_score = np.mean(compliance_scores) if compliance_scores else 0.5

        return {
            "endpoint_risks": avg_endpoint_risks,
            "safety_margins": avg_safety_margins,
            "compliance_score": compliance_score,
            "cumulative_risk": risk_assessment.cumulative_risk_score,
            "critical_pathways": risk_assessment.critical_exposure_pathways,
        }

    def _create_dashboard_layout(self, data_sources: Dict[str, Any]) -> html.Div:
        """Create dashboard layout."""
        return html.Div(
            [
                dbc.NavbarSimple(
                    brand="Bio-Interface Analysis Dashboard",
                    brand_href="#",
                    color="primary",
                    dark=True,
                    className="mb-4",
                ),
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H4("Material Selection"),
                                        dcc.Dropdown(
                                            id="material-dropdown",
                                            options=[
                                                {"label": f"Material {i+1}", "value": i}
                                                for i in range(
                                                    len(
                                                        data_sources.get(
                                                            "materials", [{}]
                                                        )
                                                    )
                                                )
                                            ],
                                            value=0,
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.H4("Analysis Type"),
                                        dcc.Dropdown(
                                            id="analysis-dropdown",
                                            options=[
                                                {
                                                    "label": "Molecular Interactions",
                                                    "value": "molecular",
                                                },
                                                {
                                                    "label": "Cellular Response",
                                                    "value": "cellular",
                                                },
                                                {
                                                    "label": "Risk Assessment",
                                                    "value": "risk",
                                                },
                                                {
                                                    "label": "Optimization Tradeoffs",
                                                    "value": "tradeoffs",
                                                },
                                            ],
                                            value="molecular",
                                        ),
                                    ],
                                    width=6,
                                ),
                            ],
                            className="mb-4",
                        ),
                        dbc.Row(
                            [
                                dbc.Col([dcc.Graph(id="main-visualization")], width=8),
                                dbc.Col(
                                    [
                                        html.H5("Analysis Parameters"),
                                        html.Hr(),
                                        html.Div(id="parameter-controls"),
                                        html.Hr(),
                                        html.H5("Key Metrics"),
                                        html.Div(id="metrics-display"),
                                    ],
                                    width=4,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H5("Data Export"),
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(
                                                    "Export CSV",
                                                    id="export-csv",
                                                    color="info",
                                                ),
                                                dbc.Button(
                                                    "Export JSON",
                                                    id="export-json",
                                                    color="info",
                                                ),
                                                dbc.Button(
                                                    "Export Image",
                                                    id="export-image",
                                                    color="info",
                                                ),
                                            ]
                                        ),
                                    ],
                                    width=12,
                                )
                            ],
                            className="mt-4",
                        ),
                    ]
                ),
            ]
        )

    def _register_dashboard_callbacks(
        self, app: dash.Dash, data_sources: Dict[str, Any]
    ) -> None:
        """Register dashboard callbacks."""

        @app.callback(
            Output("main-visualization", "figure"),
            [Input("material-dropdown", "value"), Input("analysis-dropdown", "value")],
        )
        def update_main_visualization(material_idx, analysis_type):
            # Generate visualization based on selections
            if analysis_type == "molecular":
                # Create molecular interactions plot
                return self._create_sample_molecular_plot()
            elif analysis_type == "cellular":
                # Create cellular response plot
                return self._create_sample_cellular_plot()
            elif analysis_type == "risk":
                # Create risk assessment plot
                return self._create_sample_risk_plot()
            else:
                # Create tradeoffs plot
                return self._create_sample_tradeoffs_plot()

        @app.callback(
            Output("parameter-controls", "children"),
            [Input("analysis-dropdown", "value")],
        )
        def update_parameter_controls(analysis_type):
            # Create parameter controls based on analysis type
            if analysis_type == "molecular":
                return [
                    html.Label("Simulation Time (ns)"),
                    dcc.Slider(id="sim-time", min=1, max=100, value=10, step=1),
                    html.Label("Temperature (K)"),
                    dcc.Slider(id="temperature", min=250, max=400, value=310, step=5),
                ]
            elif analysis_type == "cellular":
                return [
                    html.Label("Analysis Duration (hours)"),
                    dcc.Slider(id="duration", min=1, max=168, value=24, step=1),
                    html.Label("Cell Density"),
                    dcc.Slider(
                        id="cell-density", min=1000, max=100000, value=10000, step=1000
                    ),
                ]
            else:
                return [html.P("No parameters available for this analysis type.")]

    # Sample plot creation methods for dashboard
    def _create_sample_molecular_plot(self) -> go.Figure:
        """Create sample molecular interactions plot."""
        x = np.linspace(0, 100, 50)
        y = np.exp(-x / 20) * np.sin(x / 5)

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode="lines+markers"))
        fig.update_layout(
            title="Molecular Interactions",
            xaxis_title="Time (ns)",
            yaxis_title="Energy (kJ/mol)",
        )
        return fig

    def _create_sample_cellular_plot(self) -> go.Figure:
        """Create sample cellular response plot."""
        x = np.linspace(0, 24, 25)
        y = 1 - np.exp(-x / 6)

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode="lines+markers"))
        fig.update_layout(
            title="Cell Adhesion",
            xaxis_title="Time (hours)",
            yaxis_title="Adhesion Strength",
        )
        return fig

    def _create_sample_risk_plot(self) -> go.Figure:
        """Create sample risk assessment plot."""
        categories = ["Cytotoxicity", "Genotoxicity", "Immunotoxicity"]
        values = [0.2, 0.1, 0.3]

        fig = go.Figure(data=go.Bar(x=categories, y=values))
        fig.update_layout(title="Risk Assessment", yaxis_title="Risk Score")
        return fig

    def _create_sample_tradeoffs_plot(self) -> go.Figure:
        """Create sample tradeoffs plot."""
        x = np.random.rand(20)
        y = np.random.rand(20)

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode="markers"))
        fig.update_layout(
            title="Optimization Tradeoffs",
            xaxis_title="Objective 1",
            yaxis_title="Objective 2",
        )
        return fig

    # Fallback implementations for when advanced libraries are not available
    def _create_basic_molecular_interactions(self, data: Dict[str, Any]) -> str:
        """Create basic molecular interactions visualization."""
        html_content = f"""
        <html>
        <head><title>Molecular Interactions</title></head>
        <body>
            <h1>Molecular Interactions at Bio-Interface</h1>
            <h2>Key Metrics:</h2>
            <ul>
                <li>Binding Energy: {data.get('binding_energy', 'N/A')} kJ/mol</li>
                <li>Contact Area: {data.get('contact_area', {}).get('area', ['N/A'])[-1] if data.get('contact_area') else 'N/A'} nm²</li>
                <li>Interaction Residues: {', '.join(data.get('interaction_residues', []))}</li>
            </ul>
        </body>
        </html>
        """
        return html_content

    def _create_basic_cellular_response(self, data: Dict[str, Any]) -> str:
        """Create basic cellular response visualization."""
        html_content = f"""
        <html>
        <head><title>Cellular Response</title></head>
        <body>
            <h1>Cellular Response to Bio-Interface</h1>
            <h2>Cell Type: {data.get('cell_type', 'Unknown')}</h2>
            <h2>Key Metrics:</h2>
            <ul>
                <li>Final Viability: {data.get('viability_profile', [0])[-1]:.2f}</li>
                <li>Proliferation Metrics: {data.get('proliferation_metrics', {})}</li>
            </ul>
        </body>
        </html>
        """
        return html_content

    def _create_basic_surface_properties(self, surface_data: Dict[str, Any]) -> str:
        """Create basic surface properties visualization."""
        html_content = """
        <html>
        <head><title>Surface Properties</title></head>
        <body>
            <h1>Surface Properties Analysis</h1>
            <h2>Property Impact on Biocompatibility:</h2>
            <ul>
        """

        for prop, impact in surface_data.get("property_impact", {}).items():
            html_content += f"<li>{prop}: {impact:.3f}</li>"

        html_content += """
            </ul>
        </body>
        </html>
        """
        return html_content

    def _create_basic_heatmap(
        self, heatmap_data: np.ndarray, endpoints: List[str]
    ) -> str:
        """Create basic heatmap visualization."""
        html_content = """
        <html>
        <head><title>Biocompatibility Heatmap</title></head>
        <body>
            <h1>Material Biocompatibility Heatmap</h1>
            <table border="1" style="border-collapse: collapse;">
                <tr><th>Material</th>
        """

        for endpoint in endpoints:
            html_content += f"<th>{endpoint}</th>"
        html_content += "</tr>"

        for i in range(heatmap_data.shape[0]):
            html_content += f"<tr><td>Material_{i+1}</td>"
            for j in range(heatmap_data.shape[1]):
                value = heatmap_data[i, j]
                color = f"rgb({int(255*(1-value))}, {int(255*value)}, 0)"
                html_content += f'<td style="background-color: {color}; padding: 5px;">{value:.2f}</td>'
            html_content += "</tr>"

        html_content += """
            </table>
        </body>
        </html>
        """
        return html_content

    def _create_basic_tradeoffs(
        self, tradeoff_data: Dict[str, Any], objectives: List[str]
    ) -> str:
        """Create basic tradeoffs visualization."""
        html_content = """
        <html>
        <head><title>Optimization Tradeoffs</title></head>
        <body>
            <h1>Multi-Objective Optimization Tradeoffs</h1>
            <h2>Pareto Optimal Solutions:</h2>
            <table border="1" style="border-collapse: collapse;">
                <tr><th>Surface ID</th>
        """

        for obj in objectives:
            html_content += f"<th>{obj}</th>"
        html_content += "<th>Pareto Rank</th></tr>"

        n_solutions = len(tradeoff_data["surface_ids"])
        for i in range(min(10, n_solutions)):  # Show top 10
            html_content += f"<tr><td>{tradeoff_data['surface_ids'][i]}</td>"
            for obj in objectives:
                html_content += f"<td>{tradeoff_data[obj][i]:.3f}</td>"
            html_content += f"<td>{tradeoff_data['pareto_rank'][i]}</td></tr>"

        html_content += """
            </table>
        </body>
        </html>
        """
        return html_content

    def _create_basic_risk_assessment(self, risk_components: Dict[str, Any]) -> str:
        """Create basic risk assessment visualization."""
        html_content = """
        <html>
        <head><title>Risk Assessment</title></head>
        <body>
            <h1>Integrated Risk Assessment</h1>
            <h2>Risk by Endpoint:</h2>
            <ul>
        """

        for endpoint, risk in risk_components.get("endpoint_risks", {}).items():
            risk_level = "Low" if risk < 0.3 else "Moderate" if risk < 0.7 else "High"
            html_content += f"<li>{endpoint}: {risk:.3f} ({risk_level})</li>"

        html_content += f"""
            </ul>
            <h2>Overall Risk Score: {risk_components.get('cumulative_risk', 'N/A')}</h2>
            <h2>Regulatory Compliance: {risk_components.get('compliance_score', 'N/A'):.1%}</h2>
        </body>
        </html>
        """
        return html_content

    # Matplotlib implementations (simplified)
    def _create_matplotlib_molecular_interactions(
        self, data: Dict[str, Any], include_animation: bool
    ):
        """Create matplotlib molecular interactions visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_basic_molecular_interactions(data)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 3D structure (simplified to 2D projection)
        if "coordinates" in data:
            coords = data["coordinates"]
            ax1.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
            ax1.set_title("Molecular Structure (2D Projection)")
            ax1.set_xlabel("X coordinate")
            ax1.set_ylabel("Y coordinate")

        # Energy profile
        if "energy_profile" in data:
            time_points = data["energy_profile"]["time"]
            energies = data["energy_profile"]["energy"]
            ax2.plot(time_points, energies, "r-", linewidth=2)
            ax2.set_title("Binding Energy Profile")
            ax2.set_xlabel("Time (ns)")
            ax2.set_ylabel("Energy (kJ/mol)")

        # Contact area
        if "contact_area" in data:
            time_points = data["contact_area"]["time"]
            areas = data["contact_area"]["area"]
            ax3.fill_between(time_points, areas, alpha=0.5, color="green")
            ax3.set_title("Contact Area Evolution")
            ax3.set_xlabel("Time (ns)")
            ax3.set_ylabel("Area (nm²)")

        # Orientation angles
        if "orientation_angles" in data:
            angles = data["orientation_angles"]
            angle_names = ["Theta", "Phi", "Psi"]
            ax4.bar(angle_names, angles, color=["blue", "orange", "green"])
            ax4.set_title("Orientation Angles")
            ax4.set_ylabel("Angle (degrees)")

        plt.tight_layout()

        # Save to file and return path
        output_path = "molecular_interactions.png"
        plt.savefig(output_path, dpi=self.config.resolution, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_matplotlib_cellular_response(
        self, data: Dict[str, Any], time_series: bool
    ):
        """Create matplotlib cellular response visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_basic_cellular_response(data)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Adhesion kinetics
        if "adhesion_kinetics" in data:
            time_points = data["adhesion_kinetics"]["time"]
            strength = data["adhesion_kinetics"]["strength"]
            ax1.plot(time_points, strength, "b-", linewidth=2, marker="o")
            ax1.set_title("Cell Adhesion Kinetics")
            ax1.set_xlabel("Time (hours)")
            ax1.set_ylabel("Adhesion Strength")

        # Spreading kinetics
        if "spreading_kinetics" in data:
            time_points = data["spreading_kinetics"]["time_points"]
            areas = data["spreading_kinetics"]["areas"]
            ax2.fill_between(time_points, areas, alpha=0.6, color="orange")
            ax2.set_title("Cell Spreading Area")
            ax2.set_xlabel("Time (hours)")
            ax2.set_ylabel("Area (μm²)")

        # Viability profile
        if "viability_profile" in data:
            time_points = list(range(len(data["viability_profile"])))
            viability = data["viability_profile"]
            ax3.plot(time_points, viability, "g-", linewidth=2, marker="s")
            ax3.set_title("Cell Viability")
            ax3.set_xlabel("Time (hours)")
            ax3.set_ylabel("Viability Score")

        # Proliferation metrics
        if "proliferation_metrics" in data:
            metrics = data["proliferation_metrics"]
            names = list(metrics.keys())
            values = list(metrics.values())
            ax4.bar(names, values, color=["red", "blue", "green"])
            ax4.set_title("Proliferation Metrics")
            ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save to file and return path
        output_path = "cellular_response.png"
        plt.savefig(output_path, dpi=self.config.resolution, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_matplotlib_surface_properties(
        self, surface_data: Dict[str, Any], property_correlations: bool
    ):
        """Create matplotlib surface properties visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_basic_surface_properties(surface_data)

        if property_correlations and "correlation_matrix" in surface_data:
            # Correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            corr_matrix = surface_data["correlation_matrix"]
            property_names = surface_data["property_names"]

            im = ax.imshow(corr_matrix, cmap="RdBu", vmin=-1, vmax=1)
            ax.set_xticks(range(len(property_names)))
            ax.set_yticks(range(len(property_names)))
            ax.set_xticklabels(property_names, rotation=45)
            ax.set_yticklabels(property_names)

            # Add correlation values
            for i in range(len(property_names)):
                for j in range(len(property_names)):
                    ax.text(
                        j,
                        i,
                        f"{corr_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

            plt.colorbar(im, ax=ax, label="Correlation")
            plt.title("Surface Property Correlations")
            plt.tight_layout()

            output_path = "surface_correlations.png"
            plt.savefig(output_path, dpi=self.config.resolution, bbox_inches="tight")
            plt.close()

            return output_path
        else:
            # Property impact visualization
            fig, ax = plt.subplots(figsize=(10, 6))

            if "property_impact" in surface_data:
                properties = list(surface_data["property_impact"].keys())
                impacts = list(surface_data["property_impact"].values())

                bars = ax.bar(
                    properties,
                    impacts,
                    color=self.color_schemes["surface_properties"][: len(properties)],
                )
                ax.set_title("Property Impact on Biocompatibility")
                ax.set_ylabel("Impact Score")
                ax.tick_params(axis="x", rotation=45)

                plt.tight_layout()

                output_path = "surface_properties.png"
                plt.savefig(
                    output_path, dpi=self.config.resolution, bbox_inches="tight"
                )
                plt.close()

                return output_path

    def _create_matplotlib_heatmap(
        self, heatmap_data: np.ndarray, endpoints: List[str]
    ):
        """Create matplotlib biocompatibility heatmap."""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_basic_heatmap(heatmap_data, endpoints)

        fig, ax = plt.subplots(figsize=(12, 8))

        material_names = [f"Material_{i+1}" for i in range(heatmap_data.shape[0])]

        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(endpoints)))
        ax.set_yticks(range(len(material_names)))
        ax.set_xticklabels(endpoints, rotation=45)
        ax.set_yticklabels(material_names)

        # Add values to heatmap
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{heatmap_data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        plt.colorbar(im, ax=ax, label="Biocompatibility Score")
        plt.title("Material Biocompatibility Heatmap")
        plt.tight_layout()

        output_path = "biocompatibility_heatmap.png"
        plt.savefig(output_path, dpi=self.config.resolution, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_matplotlib_tradeoffs(
        self, tradeoff_data: Dict[str, Any], objectives: List[str]
    ):
        """Create matplotlib optimization tradeoffs visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_basic_tradeoffs(tradeoff_data, objectives)

        if len(objectives) >= 3:
            # 3D scatter plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            scatter = ax.scatter(
                tradeoff_data[objectives[0]],
                tradeoff_data[objectives[1]],
                tradeoff_data[objectives[2]],
                c=tradeoff_data.get(
                    "pareto_rank", [1] * len(tradeoff_data[objectives[0]])
                ),
                cmap="viridis",
                s=60,
            )

            ax.set_xlabel(objectives[0])
            ax.set_ylabel(objectives[1])
            ax.set_zlabel(objectives[2])
            ax.set_title("Multi-Objective Optimization Tradeoffs")

            plt.colorbar(scatter, ax=ax, label="Pareto Rank")
        else:
            # 2D scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))

            scatter = ax.scatter(
                tradeoff_data[objectives[0]],
                tradeoff_data[objectives[1]],
                c=tradeoff_data.get(
                    "pareto_rank", [1] * len(tradeoff_data[objectives[0]])
                ),
                cmap="viridis",
                s=60,
            )

            ax.set_xlabel(objectives[0])
            ax.set_ylabel(objectives[1])
            ax.set_title("Bi-Objective Optimization Tradeoffs")

            plt.colorbar(scatter, ax=ax, label="Pareto Rank")

        plt.tight_layout()

        output_path = "optimization_tradeoffs.png"
        plt.savefig(output_path, dpi=self.config.resolution, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_matplotlib_risk_assessment(self, risk_components: Dict[str, Any]):
        """Create matplotlib risk assessment visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_basic_risk_assessment(risk_components)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Risk by endpoint
        if "endpoint_risks" in risk_components:
            endpoints = list(risk_components["endpoint_risks"].keys())
            risks = list(risk_components["endpoint_risks"].values())

            bars = ax1.bar(
                endpoints, risks, color=self.color_schemes["toxicity"][: len(endpoints)]
            )
            ax1.set_title("Risk by Endpoint")
            ax1.set_ylabel("Risk Score")
            ax1.tick_params(axis="x", rotation=45)

        # Safety margins
        if "safety_margins" in risk_components:
            endpoints = list(risk_components["safety_margins"].keys())
            margins = list(risk_components["safety_margins"].values())

            ax2.scatter(
                endpoints,
                margins,
                s=[max(20, min(200, m * 10)) for m in margins],
                color="green",
                alpha=0.6,
            )
            ax2.set_title("Safety Margins")
            ax2.set_ylabel("Safety Margin")
            ax2.tick_params(axis="x", rotation=45)

        # Cumulative risk gauge (simplified)
        cumulative_risk = risk_components.get("cumulative_risk", 0.5)
        risk_colors = ["green", "yellow", "orange", "red"]
        risk_levels = [0.25, 0.5, 0.75, 1.0]

        for i, (level, color) in enumerate(zip(risk_levels, risk_colors)):
            if cumulative_risk <= level:
                ax3.pie(
                    [cumulative_risk, 1 - cumulative_risk],
                    colors=[color, "lightgray"],
                    startangle=90,
                )
                break
        ax3.set_title(f"Cumulative Risk: {cumulative_risk:.2f}")

        # Compliance score
        compliance = risk_components.get("compliance_score", 0.5)
        ax4.bar(["Compliance"], [compliance * 100], color="blue", alpha=0.7)
        ax4.axhline(y=90, color="red", linestyle="--", label="Regulatory Threshold")
        ax4.set_ylabel("Compliance (%)")
        ax4.set_title("Regulatory Compliance")
        ax4.legend()

        plt.tight_layout()

        output_path = "risk_assessment.png"
        plt.savefig(output_path, dpi=self.config.resolution, bbox_inches="tight")
        plt.close()

        return output_path

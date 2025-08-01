"""
Interactive Transport Pathway Visualization

This module provides interactive visualization capabilities for molecular transport
pathways, concentration gradients, and dynamic interactions at interfaces using
modern visualization libraries and GPU acceleration.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.widgets import Button, Slider

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, callback, dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import HTML, display

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from .biological_transport import BiologicalTransportResult
from .simulation_engine import TransportResult
from .transport_analyzer import PartitioningAnalysis, TransportKinetics


@dataclass
class VisualizationConfig:
    """Configuration for transport visualization."""

    backend: str = "plotly"  # 'matplotlib', 'plotly', 'dash'
    theme: str = "default"  # 'default', 'dark', 'scientific'
    interactive: bool = True
    animation_fps: int = 30
    color_scheme: str = "plasma"
    figure_size: Tuple[int, int] = (12, 8)
    resolution: int = 300  # DPI for static plots


class TransportVisualizationBase(ABC):
    """Abstract base class for transport visualizations."""

    @abstractmethod
    def create_concentration_profile_plot(
        self, transport_result: TransportResult
    ) -> Any:
        """Create concentration profile visualization."""
        pass

    @abstractmethod
    def create_flux_animation(self, transport_result: TransportResult) -> Any:
        """Create animated flux visualization."""
        pass

    @abstractmethod
    def create_pathway_diagram(self, transport_result: TransportResult) -> Any:
        """Create transport pathway diagram."""
        pass


class MatplotlibVisualizer(TransportVisualizationBase):
    """Matplotlib-based transport visualization."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not MATPLOTLIB_AVAILABLE:
            raise ChemiAccelError("Matplotlib not available for visualization")

        # Set style
        plt.style.use(
            "seaborn-v0_8" if hasattr(plt.style, "seaborn-v0_8") else "default"
        )

    def create_concentration_profile_plot(
        self, transport_result: TransportResult
    ) -> plt.Figure:
        """Create static concentration profile plot."""
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle("Transport Analysis Results", fontsize=16, fontweight="bold")

        # Plot 1: Concentration profiles over space
        ax1 = axes[0, 0]
        if transport_result.concentration_profiles:
            for frame_name, profile in transport_result.concentration_profiles.items():
                z_positions = np.linspace(-10, 10, len(profile))
                ax1.plot(z_positions, profile, label=frame_name, linewidth=2)

        ax1.set_xlabel("Position (nm)")
        ax1.set_ylabel("Concentration (molecules/nm³)")
        ax1.set_title("Concentration Profiles")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Flux over time
        ax2 = axes[0, 1]
        if len(transport_result.flux_data) > 0:
            time_points = np.arange(len(transport_result.flux_data))
            ax2.plot(
                time_points, transport_result.flux_data, "o-", color="red", linewidth=2
            )
            ax2.fill_between(
                time_points, transport_result.flux_data, alpha=0.3, color="red"
            )

        ax2.set_xlabel("Time Point")
        ax2.set_ylabel("Flux (molecules/time)")
        ax2.set_title("Transport Flux")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Molecular trajectories (sample)
        ax3 = axes[1, 0]
        if transport_result.trajectories.shape[0] > 0:
            n_sample = min(5, transport_result.trajectories.shape[0])
            for i in range(n_sample):
                z_traj = transport_result.trajectories[i, :, 2]
                time_points = np.arange(len(z_traj))
                ax3.plot(time_points, z_traj, alpha=0.7, linewidth=1.5)

        ax3.set_xlabel("Time Point")
        ax3.set_ylabel("Z Position (nm)")
        ax3.set_title("Sample Molecular Trajectories")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Transport properties summary
        ax4 = axes[1, 1]
        properties = {
            "Permeability": transport_result.permeability_coefficient
            * 1e6,  # Convert units
            "Avg Diffusion": np.mean(
                list(transport_result.diffusion_coefficients.values())
            )
            * 1e9,
            "Max Partition": max(transport_result.partition_coefficients.values())
            if transport_result.partition_coefficients
            else 0,
        }

        bars = ax4.bar(
            properties.keys(),
            properties.values(),
            color=["skyblue", "lightgreen", "orange"],
        )
        ax4.set_ylabel("Property Value")
        ax4.set_title("Transport Properties Summary")
        ax4.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, properties.values()):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        return fig

    def create_flux_animation(
        self, transport_result: TransportResult
    ) -> animation.FuncAnimation:
        """Create animated visualization of molecular flux."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Setup the plot
        ax.set_xlim(-12, 12)
        ax.set_ylim(-2, 12)
        ax.set_xlabel("X Position (nm)")
        ax.set_ylabel("Z Position (nm)")
        ax.set_title("Molecular Transport Animation")

        # Add membrane representation
        membrane_x = np.linspace(-12, 12, 100)
        membrane_y1 = np.full_like(membrane_x, -1)
        membrane_y2 = np.full_like(membrane_x, 1)
        ax.fill_between(
            membrane_x,
            membrane_y1,
            membrane_y2,
            alpha=0.3,
            color="gray",
            label="Membrane",
        )

        # Initialize scatter plot for molecules
        scat = ax.scatter([], [], s=50, c=[], cmap=self.config.color_scheme, alpha=0.7)

        # Animation function
        def animate(frame):
            if frame < transport_result.trajectories.shape[1]:
                x_positions = transport_result.trajectories[:, frame, 0]
                z_positions = transport_result.trajectories[:, frame, 2]

                # Color by molecule index
                colors = np.arange(len(x_positions))

                scat.set_offsets(np.column_stack([x_positions, z_positions]))
                scat.set_array(colors)

            return (scat,)

        # Create animation
        n_frames = transport_result.trajectories.shape[1]
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=n_frames,
            interval=1000 // self.config.animation_fps,
            blit=True,
            repeat=True,
        )

        return anim

    def create_pathway_diagram(self, transport_result: TransportResult) -> plt.Figure:
        """Create transport pathway diagram."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Create simplified pathway diagram
        phases = ["Aqueous", "Interface", "Membrane", "Interface", "Aqueous"]
        positions = np.arange(len(phases))

        # Plot phase regions
        colors = ["lightblue", "yellow", "gray", "yellow", "lightblue"]
        for i, (phase, color) in enumerate(zip(phases, colors)):
            ax.barh(
                0,
                1,
                left=i,
                height=0.5,
                color=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=2,
            )
            ax.text(
                i + 0.5,
                0,
                phase,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=10,
            )

        # Add transport arrows
        arrow_props = dict(arrowstyle="->", lw=3, color="red")
        for i in range(len(phases) - 1):
            ax.annotate(
                "", xy=(i + 1, 0.3), xytext=(i + 0.8, 0.3), arrowprops=arrow_props
            )

        # Add property annotations
        if transport_result.permeability_coefficient:
            ax.text(
                2.5,
                -0.8,
                f"Permeability: {transport_result.permeability_coefficient:.2e} cm/s",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white"),
            )

        ax.set_xlim(-0.5, len(phases) - 0.5)
        ax.set_ylim(-1.2, 1)
        ax.set_title("Transport Pathway Schematic", fontsize=16, fontweight="bold")
        ax.axis("off")

        return fig


class PlotlyVisualizer(TransportVisualizationBase):
    """Plotly-based interactive transport visualization."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not PLOTLY_AVAILABLE:
            raise ChemiAccelError("Plotly not available for visualization")

    def create_concentration_profile_plot(
        self, transport_result: TransportResult
    ) -> go.Figure:
        """Create interactive concentration profile plot."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Concentration Profiles",
                "Transport Flux",
                "Molecular Trajectories",
                "Properties Summary",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "bar"}],
            ],
        )

        # Plot 1: Concentration profiles
        if transport_result.concentration_profiles:
            for frame_name, profile in transport_result.concentration_profiles.items():
                z_positions = np.linspace(-10, 10, len(profile))
                fig.add_trace(
                    go.Scatter(
                        x=z_positions,
                        y=profile,
                        mode="lines+markers",
                        name=frame_name,
                        line=dict(width=3),
                    ),
                    row=1,
                    col=1,
                )

        # Plot 2: Flux over time
        if len(transport_result.flux_data) > 0:
            time_points = np.arange(len(transport_result.flux_data))
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=transport_result.flux_data,
                    mode="lines+markers",
                    name="Flux",
                    line=dict(color="red", width=3),
                    fill="tonexty",
                ),
                row=1,
                col=2,
            )

        # Plot 3: Sample trajectories
        if transport_result.trajectories.shape[0] > 0:
            n_sample = min(5, transport_result.trajectories.shape[0])
            for i in range(n_sample):
                z_traj = transport_result.trajectories[i, :, 2]
                time_points = np.arange(len(z_traj))
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=z_traj,
                        mode="lines",
                        name=f"Molecule {i+1}",
                        opacity=0.7,
                    ),
                    row=2,
                    col=1,
                )

        # Plot 4: Properties bar chart
        properties = {
            "Permeability (×10⁻⁶)": transport_result.permeability_coefficient * 1e6,
            "Avg Diffusion (×10⁻⁹)": np.mean(
                list(transport_result.diffusion_coefficients.values())
            )
            * 1e9,
            "Max Partition": max(transport_result.partition_coefficients.values())
            if transport_result.partition_coefficients
            else 0,
        }

        fig.add_trace(
            go.Bar(
                x=list(properties.keys()),
                y=list(properties.values()),
                marker=dict(color=["skyblue", "lightgreen", "orange"]),
                text=[f"{v:.2f}" for v in properties.values()],
                textposition="auto",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title_text="Transport Analysis Dashboard",
            title_x=0.5,
            showlegend=True,
            height=800,
            template="plotly_white"
            if self.config.theme == "default"
            else "plotly_dark",
        )

        # Update axis labels
        fig.update_xaxes(title_text="Position (nm)", row=1, col=1)
        fig.update_yaxes(title_text="Concentration", row=1, col=1)
        fig.update_xaxes(title_text="Time Point", row=1, col=2)
        fig.update_yaxes(title_text="Flux", row=1, col=2)
        fig.update_xaxes(title_text="Time Point", row=2, col=1)
        fig.update_yaxes(title_text="Z Position (nm)", row=2, col=1)
        fig.update_yaxes(title_text="Property Value", row=2, col=2)

        return fig

    def create_flux_animation(self, transport_result: TransportResult) -> go.Figure:
        """Create interactive 3D animation of molecular transport."""

        # Prepare data for animation
        n_molecules, n_frames, _ = transport_result.trajectories.shape

        # Create frames for animation
        frames = []
        for frame_idx in range(n_frames):
            x_pos = transport_result.trajectories[:, frame_idx, 0]
            y_pos = transport_result.trajectories[:, frame_idx, 1]
            z_pos = transport_result.trajectories[:, frame_idx, 2]

            frame_data = go.Scatter3d(
                x=x_pos,
                y=y_pos,
                z=z_pos,
                mode="markers",
                marker=dict(
                    size=8,
                    color=np.arange(n_molecules),
                    colorscale=self.config.color_scheme,
                    opacity=0.8,
                ),
                name=f"Frame {frame_idx}",
            )

            frames.append(go.Frame(data=[frame_data], name=str(frame_idx)))

        # Create initial figure
        initial_data = go.Scatter3d(
            x=transport_result.trajectories[:, 0, 0],
            y=transport_result.trajectories[:, 0, 1],
            z=transport_result.trajectories[:, 0, 2],
            mode="markers",
            marker=dict(
                size=8,
                color=np.arange(n_molecules),
                colorscale=self.config.color_scheme,
                opacity=0.8,
            ),
        )

        # Add membrane representation
        membrane_x, membrane_y = np.meshgrid(
            np.linspace(-10, 10, 20), np.linspace(-10, 10, 20)
        )
        membrane_z1 = np.full_like(membrane_x, -1)
        membrane_z2 = np.full_like(membrane_x, 1)

        membrane1 = go.Surface(
            x=membrane_x,
            y=membrane_y,
            z=membrane_z1,
            opacity=0.3,
            colorscale="Greys",
            showscale=False,
        )
        membrane2 = go.Surface(
            x=membrane_x,
            y=membrane_y,
            z=membrane_z2,
            opacity=0.3,
            colorscale="Greys",
            showscale=False,
        )

        fig = go.Figure(data=[initial_data, membrane1, membrane2], frames=frames)

        # Add animation controls
        fig.update_layout(
            title="3D Molecular Transport Animation",
            scene=dict(
                xaxis_title="X Position (nm)",
                yaxis_title="Y Position (nm)",
                zaxis_title="Z Position (nm)",
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {
                                        "duration": 1000 // self.config.animation_fps,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top",
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=20),
                        prefix="Frame: ",
                        visible=True,
                        xanchor="right",
                    ),
                    transition=dict(duration=300, easing="cubic-in-out"),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[
                                [k],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                            label=str(k),
                            method="animate",
                        )
                        for k in range(n_frames)
                    ],
                )
            ],
        )

        return fig

    def create_pathway_diagram(self, transport_result: TransportResult) -> go.Figure:
        """Create interactive transport pathway diagram."""

        # Create Sankey diagram for transport pathways
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=[
                            "Donor Phase",
                            "Interface 1",
                            "Membrane",
                            "Interface 2",
                            "Acceptor Phase",
                        ],
                        color=["lightblue", "yellow", "gray", "yellow", "lightgreen"],
                    ),
                    link=dict(
                        source=[0, 1, 2, 3],  # indices correspond to labels
                        target=[1, 2, 3, 4],
                        value=[100, 80, 60, 50],  # flow values (simplified)
                        color=[
                            "rgba(255,0,0,0.5)",
                            "rgba(0,255,0,0.5)",
                            "rgba(0,0,255,0.5)",
                            "rgba(255,255,0,0.5)",
                        ],
                    ),
                )
            ]
        )

        fig.update_layout(
            title_text="Transport Pathway Flow Diagram",
            font_size=12,
            height=600,
            template="plotly_white"
            if self.config.theme == "default"
            else "plotly_dark",
        )

        return fig


class DashboardVisualizer:
    """Dash-based interactive dashboard for transport visualization."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not DASH_AVAILABLE:
            raise ChemiAccelError("Dash not available for dashboard visualization")

        self.app = None

    def create_dashboard(
        self, transport_results: Dict[str, TransportResult]
    ) -> dash.Dash:
        """Create interactive dashboard for multiple transport results."""

        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Define layout
        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Transport Analysis Dashboard",
                                    className="text-center mb-4",
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Select Molecule:"),
                                dcc.Dropdown(
                                    id="molecule-dropdown",
                                    options=[
                                        {"label": mol, "value": mol}
                                        for mol in transport_results.keys()
                                    ],
                                    value=list(transport_results.keys())[0]
                                    if transport_results
                                    else None,
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Visualization Type:"),
                                dcc.Dropdown(
                                    id="viz-type-dropdown",
                                    options=[
                                        {
                                            "label": "Concentration Profiles",
                                            "value": "concentration",
                                        },
                                        {"label": "Flux Animation", "value": "flux"},
                                        {
                                            "label": "Pathway Diagram",
                                            "value": "pathway",
                                        },
                                    ],
                                    value="concentration",
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row([dbc.Col([dcc.Graph(id="main-plot")])]),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("Transport Properties"),
                                html.Div(id="properties-table"),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.H3("Analysis Summary"),
                                html.Div(id="analysis-summary"),
                            ],
                            width=6,
                        ),
                    ],
                    className="mt-4",
                ),
            ],
            fluid=True,
        )

        # Define callbacks
        @self.app.callback(
            [
                Output("main-plot", "figure"),
                Output("properties-table", "children"),
                Output("analysis-summary", "children"),
            ],
            [Input("molecule-dropdown", "value"), Input("viz-type-dropdown", "value")],
        )
        def update_dashboard(selected_molecule, viz_type):
            if selected_molecule and selected_molecule in transport_results:
                result = transport_results[selected_molecule]

                # Create visualization based on type
                plotly_viz = PlotlyVisualizer(self.config)

                if viz_type == "concentration":
                    fig = plotly_viz.create_concentration_profile_plot(result)
                elif viz_type == "flux":
                    fig = plotly_viz.create_flux_animation(result)
                else:  # pathway
                    fig = plotly_viz.create_pathway_diagram(result)

                # Create properties table
                properties_table = dbc.Table(
                    [
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td("Permeability Coefficient"),
                                        html.Td(
                                            f"{result.permeability_coefficient:.2e} cm/s"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Simulation Time"),
                                        html.Td(f"{result.simulation_time:.2f} s"),
                                    ]
                                ),
                                *[
                                    html.Tr(
                                        [
                                            html.Td(f"{phase} Diffusion"),
                                            html.Td(f"{coeff:.2e} cm²/s"),
                                        ]
                                    )
                                    for phase, coeff in result.diffusion_coefficients.items()
                                ],
                                *[
                                    html.Tr(
                                        [
                                            html.Td(f"{partition_type}"),
                                            html.Td(f"{coeff:.2f}"),
                                        ]
                                    )
                                    for partition_type, coeff in result.partition_coefficients.items()
                                ],
                            ]
                        )
                    ],
                    bordered=True,
                    striped=True,
                )

                # Create analysis summary
                analysis_summary = dbc.Alert(
                    [
                        html.H5("Key Findings:", className="alert-heading"),
                        html.P(
                            f"• Transport mechanism: {result.metadata.get('method', 'Unknown')}"
                        ),
                        html.P(
                            f"• Backend used: {result.metadata.get('backend', 'Unknown')}"
                        ),
                        html.P(
                            f"• Number of molecules: {result.metadata.get('n_molecules', 'Unknown')}"
                        ),
                    ],
                    color="info",
                )

                return fig, properties_table, analysis_summary
            else:
                empty_fig = go.Figure()
                empty_fig.add_annotation(
                    text="No data selected",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
                return empty_fig, "No data", "No analysis available"

        return self.app

    def run_dashboard(self, debug=True, port=8050):
        """Run the dashboard server."""
        if self.app:
            self.app.run_server(debug=debug, port=port)
        else:
            raise ChemiAccelError(
                "Dashboard not created. Call create_dashboard() first."
            )


class TransportVisualizer:
    """
    Comprehensive visualizer for transport pathways and analysis results.

    This class provides a unified interface for creating interactive visualizations
    of molecular transport data using various backends.
    """

    def __init__(
        self, backend: str = "plotly", config: Optional[VisualizationConfig] = None
    ):
        """
        Initialize the transport visualizer.

        Args:
            backend: Visualization backend ('matplotlib', 'plotly', 'dash')
            config: Visualization configuration
        """
        self.backend = backend
        self.config = config or VisualizationConfig(backend=backend)
        self.logger = logging.getLogger(__name__)

        # Initialize appropriate visualizer
        self.visualizer = self._create_visualizer()

    def _create_visualizer(self) -> TransportVisualizationBase:
        """Create appropriate visualizer based on backend."""
        if self.backend == "matplotlib":
            return MatplotlibVisualizer(self.config)
        elif self.backend == "plotly":
            return PlotlyVisualizer(self.config)
        elif self.backend == "dash":
            return DashboardVisualizer(self.config)
        else:
            raise ChemiAccelError(f"Unknown visualization backend: {self.backend}")

    def visualize_transport_results(
        self, transport_result: TransportResult, output_file: Optional[str] = None
    ) -> Any:
        """
        Create comprehensive visualization of transport results.

        Args:
            transport_result: Transport simulation results
            output_file: Optional output file path

        Returns:
            Visualization object (figure, widget, or app)
        """
        self.logger.info("Creating transport results visualization")

        if self.backend == "dash":
            # For dashboard, we need multiple results
            dashboard = self.visualizer.create_dashboard({"molecule": transport_result})
            return dashboard
        else:
            # Create concentration profile plot
            viz = self.visualizer.create_concentration_profile_plot(transport_result)

            if output_file and self.backend == "matplotlib":
                viz.savefig(
                    output_file, dpi=self.config.resolution, bbox_inches="tight"
                )
            elif output_file and self.backend == "plotly":
                viz.write_html(output_file)

            return viz

    def animate_transport_pathways(
        self, transport_result: TransportResult, output_file: Optional[str] = None
    ) -> Any:
        """
        Create animated visualization of transport pathways.

        Args:
            transport_result: Transport simulation results
            output_file: Optional output file path

        Returns:
            Animation object or figure
        """
        self.logger.info("Creating transport pathway animation")

        animation = self.visualizer.create_flux_animation(transport_result)

        if output_file:
            if self.backend == "matplotlib":
                animation.save(
                    output_file, writer="pillow", fps=self.config.animation_fps
                )
            elif self.backend == "plotly":
                animation.write_html(output_file)

        return animation

    def create_pathway_diagram(
        self, transport_result: TransportResult, output_file: Optional[str] = None
    ) -> Any:
        """
        Create transport pathway schematic diagram.

        Args:
            transport_result: Transport simulation results
            output_file: Optional output file path

        Returns:
            Diagram figure
        """
        self.logger.info("Creating transport pathway diagram")

        diagram = self.visualizer.create_pathway_diagram(transport_result)

        if output_file:
            if self.backend == "matplotlib":
                diagram.savefig(
                    output_file, dpi=self.config.resolution, bbox_inches="tight"
                )
            elif self.backend == "plotly":
                diagram.write_html(output_file)

        return diagram

    def visualize_biological_transport(
        self, bio_result: BiologicalTransportResult, output_file: Optional[str] = None
    ) -> Any:
        """
        Visualize biological transport and ADME results.

        Args:
            bio_result: Biological transport simulation results
            output_file: Optional output file path

        Returns:
            Biological transport visualization
        """
        self.logger.info("Creating biological transport visualization")

        if self.backend == "plotly":
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Concentration vs Time",
                    "Tissue Distribution",
                    "Transporter Fluxes",
                    "Metabolite Formation",
                ),
                specs=[
                    [{"secondary_y": False}, {"type": "bar"}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
            )

            # Plot concentration profiles
            for compartment, conc_profile in bio_result.concentration_profiles.items():
                time_points = np.arange(len(conc_profile))
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=conc_profile,
                        mode="lines",
                        name=compartment,
                        line=dict(width=2),
                    ),
                    row=1,
                    col=1,
                )

            # Plot tissue distribution
            tissues = list(bio_result.tissue_distribution.keys())
            distribution_values = list(bio_result.tissue_distribution.values())
            fig.add_trace(
                go.Bar(
                    x=tissues,
                    y=distribution_values,
                    name="Distribution",
                    marker=dict(color="lightcoral"),
                ),
                row=1,
                col=2,
            )

            # Plot transporter fluxes
            for transporter, flux_profile in bio_result.transporter_fluxes.items():
                time_points = np.arange(len(flux_profile))
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=flux_profile,
                        mode="lines",
                        name=transporter,
                        opacity=0.7,
                    ),
                    row=2,
                    col=1,
                )

            # Plot metabolite formation
            for (
                metabolite,
                formation_profile,
            ) in bio_result.metabolite_formation.items():
                time_points = np.arange(len(formation_profile))
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=formation_profile,
                        mode="lines",
                        name=metabolite,
                        opacity=0.7,
                    ),
                    row=2,
                    col=2,
                )

            fig.update_layout(
                title_text="Biological Transport Analysis", height=800, showlegend=True
            )

            if output_file:
                fig.write_html(output_file)

            return fig

        else:
            # Simplified matplotlib version
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)

            # Concentration profiles
            ax1 = axes[0, 0]
            for compartment, conc_profile in bio_result.concentration_profiles.items():
                time_points = np.arange(len(conc_profile))
                ax1.plot(time_points, conc_profile, label=compartment, linewidth=2)
            ax1.set_title("Concentration vs Time")
            ax1.legend()

            # Tissue distribution
            ax2 = axes[0, 1]
            tissues = list(bio_result.tissue_distribution.keys())
            distribution_values = list(bio_result.tissue_distribution.values())
            ax2.bar(tissues, distribution_values, color="lightcoral")
            ax2.set_title("Tissue Distribution")
            ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()

            if output_file:
                fig.savefig(
                    output_file, dpi=self.config.resolution, bbox_inches="tight"
                )

            return fig

    def create_comparison_dashboard(
        self, transport_results: Dict[str, TransportResult]
    ) -> Any:
        """
        Create interactive dashboard for comparing multiple transport results.

        Args:
            transport_results: Dictionary mapping molecule names to transport results

        Returns:
            Dashboard application or comparison figure
        """
        self.logger.info(
            f"Creating comparison dashboard for {len(transport_results)} molecules"
        )

        if self.backend == "dash":
            dashboard_viz = DashboardVisualizer(self.config)
            return dashboard_viz.create_dashboard(transport_results)
        else:
            # Create comparison plot
            if self.backend == "plotly":
                fig = go.Figure()

                # Compare permeabilities
                molecules = list(transport_results.keys())
                permeabilities = [
                    result.permeability_coefficient
                    for result in transport_results.values()
                ]

                fig.add_trace(
                    go.Bar(
                        x=molecules,
                        y=permeabilities,
                        name="Permeability",
                        marker=dict(color="steelblue"),
                    )
                )

                fig.update_layout(
                    title="Transport Properties Comparison",
                    xaxis_title="Molecules",
                    yaxis_title="Permeability Coefficient (cm/s)",
                    yaxis_type="log",
                )

                return fig
            else:
                # Matplotlib comparison
                molecules = list(transport_results.keys())
                permeabilities = [
                    result.permeability_coefficient
                    for result in transport_results.values()
                ]

                fig, ax = plt.subplots(figsize=self.config.figure_size)
                bars = ax.bar(molecules, permeabilities, color="steelblue")
                ax.set_yscale("log")
                ax.set_ylabel("Permeability Coefficient (cm/s)")
                ax.set_title("Transport Properties Comparison")
                plt.xticks(rotation=45)
                plt.tight_layout()

                return fig

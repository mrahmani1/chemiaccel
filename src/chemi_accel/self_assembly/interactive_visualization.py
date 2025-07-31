"""
Interactive Self-Assembly Visualization Module for ChemiAccel.

This module provides comprehensive interactive 3D visualization tools for self-assembly
simulations, allowing users to observe structural evolution, phase transitions, and
emergent properties in real-time. It includes dashboards for tracking properties over
time and across different design parameters.

Features:
- Interactive 3D visualization of self-assembly simulations
- Real-time tracking of emergent properties and phase transitions
- Multi-parameter dashboard for design space exploration
- Timeline visualization for structural evolution
- Comparative analysis tools for different simulation conditions
- Export capabilities for high-quality figures and animations

Dependencies:
- plotly: For interactive 3D visualization and dashboards
- dash: For web-based interactive applications
- matplotlib: For static plots and figure generation
- numpy: For numerical computations
- pandas: For data management
"""

import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional imports with graceful fallbacks
try:
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some visualization features will be limited.")

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, State, callback, dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    warnings.warn("Dash not available. Interactive dashboards will be limited.")

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Static visualization will be limited.")

from .characterization import (
    CharacterizationResult,
    LiquidCrystalCharacteristics,
    MicelleCharacteristics,
    PhaseType,
    StructureMetrics,
    StructureType,
    VesicleCharacteristics,
)

# Import from other modules
from .coarse_grained_simulation import (
    Particle,
    ParticleType,
    SimulationBox,
    SimulationState,
    TrajectoryFrame,
)

# =============================================================================
# Enums and Configuration Classes
# =============================================================================


class VisualizationMode(Enum):
    """Visualization modes for different types of analysis."""

    PARTICLES_3D = "particles_3d"
    DENSITY_FIELD = "density_field"
    PHASE_DIAGRAM = "phase_diagram"
    STRUCTURE_EVOLUTION = "structure_evolution"
    PROPERTY_TRACKING = "property_tracking"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


class ColorScheme(Enum):
    """Color schemes for visualization."""

    PARTICLE_TYPE = "particle_type"
    CLUSTER_ID = "cluster_id"
    ENERGY = "energy"
    VELOCITY = "velocity"
    CUSTOM = "custom"


class AnimationSpeed(Enum):
    """Animation speed settings."""

    SLOW = 0.5
    NORMAL = 1.0
    FAST = 2.0
    VERY_FAST = 5.0


class ExportFormat(Enum):
    """Export formats for figures and animations."""

    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    MP4 = "mp4"
    GIF = "gif"


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    # Display settings
    width: int = 1200
    height: int = 800
    background_color: str = "white"
    particle_size_scale: float = 1.0
    bond_width_scale: float = 1.0

    # Color settings
    color_scheme: ColorScheme = ColorScheme.PARTICLE_TYPE
    custom_colors: Optional[Dict[str, str]] = None
    opacity: float = 0.8

    # Animation settings
    animation_speed: AnimationSpeed = AnimationSpeed.NORMAL
    frame_interval: float = 100.0  # milliseconds
    show_trajectory_traces: bool = False
    max_trajectory_length: int = 100

    # Interaction settings
    enable_zoom: bool = True
    enable_rotation: bool = True
    enable_selection: bool = True

    # Export settings
    export_dpi: int = 300
    export_quality: int = 95

    # Performance settings
    max_particles_3d: int = 10000
    downsample_threshold: int = 50000
    use_gpu_acceleration: bool = True


@dataclass
class DashboardConfig:
    """Configuration for interactive dashboards."""

    # Layout settings
    theme: str = "bootstrap"
    fluid_layout: bool = True
    sidebar_width: int = 300

    # Update settings
    update_interval: float = 1000.0  # milliseconds
    real_time_updates: bool = True
    buffer_size: int = 1000

    # Chart settings
    chart_height: int = 400
    show_legends: bool = True
    enable_crossfilter: bool = True

    # Performance settings
    max_data_points: int = 10000
    downsampling_method: str = "lttb"  # Largest Triangle Three Buckets
    cache_size: int = 100


# =============================================================================
# Data Structures for Visualization
# =============================================================================


@dataclass
class VisualizationFrame:
    """Single frame data for visualization."""

    timestamp: float
    positions: np.ndarray
    particle_types: np.ndarray
    particle_ids: np.ndarray
    velocities: Optional[np.ndarray] = None
    forces: Optional[np.ndarray] = None
    energies: Optional[np.ndarray] = None
    cluster_ids: Optional[np.ndarray] = None
    structure_metrics: Optional[StructureMetrics] = None
    box_dimensions: Optional[Tuple[float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesData:
    """Time series data for property tracking."""

    timestamps: np.ndarray
    properties: Dict[str, np.ndarray]
    labels: Dict[str, str] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    categories: Dict[str, str] = field(default_factory=dict)


@dataclass
class PhaseTransitionEvent:
    """Data structure for phase transition events."""

    timestamp: float
    from_phase: PhaseType
    to_phase: PhaseType
    transition_probability: float
    order_parameter: float
    critical_temperature: Optional[float] = None
    critical_concentration: Optional[float] = None
    transition_duration: Optional[float] = None
    nucleation_sites: Optional[List[Tuple[float, float, float]]] = None


@dataclass
class ParameterSweepData:
    """Data for parameter sweep visualization."""

    parameter_names: List[str]
    parameter_values: np.ndarray
    property_maps: Dict[str, np.ndarray]
    optimal_points: Optional[List[Tuple[float, ...]]] = None
    pareto_front: Optional[np.ndarray] = None


# =============================================================================
# Exceptions
# =============================================================================


class VisualizationError(Exception):
    """Base exception for visualization errors."""

    pass


class RenderingError(VisualizationError):
    """Exception raised when rendering fails."""

    pass


class DashboardError(VisualizationError):
    """Exception raised for dashboard-related errors."""

    pass


class ExportError(VisualizationError):
    """Exception raised when export operations fail."""

    pass


class DataProcessingError(VisualizationError):
    """Exception raised when data processing fails."""

    pass


# =============================================================================
# Core Visualization Engine
# =============================================================================


class InteractiveSelfAssemblyVisualizer:
    """
    Core visualization engine for interactive self-assembly analysis.

    This class provides comprehensive 3D visualization capabilities for
    self-assembly simulations, including real-time rendering, animation,
    and interactive controls.
    """

    def __init__(
        self, config: Optional[VisualizationConfig] = None, use_gpu: bool = True
    ):
        """
        Initialize the visualizer.

        Args:
            config: Visualization configuration
            use_gpu: Whether to use GPU acceleration
        """
        self.config = config or VisualizationConfig()
        self.use_gpu = use_gpu and self.config.use_gpu_acceleration

        # Initialize storage
        self.trajectory_data: List[VisualizationFrame] = []
        self.current_frame: int = 0
        self.animation_running: bool = False

        # Plotly figure
        self.figure: Optional[go.Figure] = None
        self.layout_initialized: bool = False

        # Color maps
        self.color_maps = self._initialize_color_maps()

        # Performance metrics
        self.render_times: List[float] = []
        self.frame_rates: List[float] = []

        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available. 3D visualization will be limited.")

    def _initialize_color_maps(self) -> Dict[str, Dict]:
        """Initialize color maps for different visualization modes."""
        return {
            "particle_type": {
                "hydrophilic": "#3498db",  # Blue
                "hydrophobic": "#e74c3c",  # Red
                "amphiphilic": "#9b59b6",  # Purple
                "solvent": "#95a5a6",  # Gray
                "ion": "#f39c12",  # Orange
                "surfactant_head": "#2ecc71",  # Green
                "surfactant_tail": "#34495e",  # Dark gray
            },
            "energy": px.colors.sequential.Viridis,
            "velocity": px.colors.sequential.Plasma,
            "cluster": px.colors.qualitative.Set3,
            "phase": {
                "liquid": "#3498db",
                "crystal": "#e74c3c",
                "gel": "#9b59b6",
                "micelle": "#2ecc71",
                "vesicle": "#f39c12",
                "bilayer": "#34495e",
            },
        }

    def load_trajectory(
        self,
        trajectory: List[TrajectoryFrame],
        structure_data: Optional[List[CharacterizationResult]] = None,
    ) -> None:
        """
        Load trajectory data for visualization.

        Args:
            trajectory: List of trajectory frames
            structure_data: Optional structure characterization data
        """
        self.trajectory_data = []

        for i, frame in enumerate(trajectory):
            # Extract positions and particle information
            positions = np.array(
                [[p.position.x, p.position.y, p.position.z] for p in frame.particles]
            )

            particle_types = np.array([p.particle_type.name for p in frame.particles])
            particle_ids = np.array([p.particle_id for p in frame.particles])

            # Extract velocities if available
            velocities = None
            if hasattr(frame.particles[0], "velocity") and frame.particles[0].velocity:
                velocities = np.array(
                    [
                        [p.velocity.x, p.velocity.y, p.velocity.z]
                        for p in frame.particles
                    ]
                )

            # Extract energies if available
            energies = None
            if hasattr(frame.particles[0], "energy"):
                energies = np.array(
                    [getattr(p, "energy", 0.0) for p in frame.particles]
                )

            # Get structure metrics if available
            structure_metrics = None
            if structure_data and i < len(structure_data):
                structure_metrics = structure_data[i].structure_metrics

            # Create visualization frame
            vis_frame = VisualizationFrame(
                timestamp=frame.timestamp,
                positions=positions,
                particle_types=particle_types,
                particle_ids=particle_ids,
                velocities=velocities,
                energies=energies,
                structure_metrics=structure_metrics,
                box_dimensions=(frame.box.length, frame.box.width, frame.box.height),
                metadata={
                    "frame_index": i,
                    "total_energy": getattr(frame, "total_energy", None),
                    "temperature": getattr(frame, "temperature", None),
                    "pressure": getattr(frame, "pressure", None),
                },
            )

            self.trajectory_data.append(vis_frame)

        # Reset current frame
        self.current_frame = 0
        self.layout_initialized = False

    def create_3d_visualization(
        self,
        frame_index: Optional[int] = None,
        mode: VisualizationMode = VisualizationMode.PARTICLES_3D,
        color_by: ColorScheme = ColorScheme.PARTICLE_TYPE,
    ) -> go.Figure:
        """
        Create interactive 3D visualization of a simulation frame.

        Args:
            frame_index: Frame index to visualize (None for current)
            mode: Visualization mode
            color_by: Color scheme for particles

        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly not available for 3D visualization")

        if not self.trajectory_data:
            raise VisualizationError("No trajectory data loaded")

        # Get frame data
        if frame_index is None:
            frame_index = self.current_frame

        if frame_index >= len(self.trajectory_data):
            raise VisualizationError(f"Frame index {frame_index} out of range")

        frame = self.trajectory_data[frame_index]

        # Create figure
        fig = go.Figure()

        # Add particles based on visualization mode
        if mode == VisualizationMode.PARTICLES_3D:
            self._add_particles_3d(fig, frame, color_by)
        elif mode == VisualizationMode.DENSITY_FIELD:
            self._add_density_field(fig, frame)
        elif mode == VisualizationMode.STRUCTURE_EVOLUTION:
            self._add_structure_evolution(fig, frame)

        # Update layout
        self._update_3d_layout(fig, frame)

        self.figure = fig
        return fig

    def _add_particles_3d(
        self, fig: go.Figure, frame: VisualizationFrame, color_by: ColorScheme
    ) -> None:
        """Add 3D particles to the figure."""
        positions = frame.positions

        # Handle large datasets by downsampling
        if len(positions) > self.config.max_particles_3d:
            indices = np.random.choice(
                len(positions), self.config.max_particles_3d, replace=False
            )
            positions = positions[indices]
            particle_types = frame.particle_types[indices]
            particle_ids = frame.particle_ids[indices]
        else:
            particle_types = frame.particle_types
            particle_ids = frame.particle_ids

        # Determine colors
        colors = self._get_particle_colors(frame, color_by)
        if len(colors) != len(positions):
            colors = colors[: len(positions)]

        # Determine sizes
        sizes = self._get_particle_sizes(particle_types)

        # Create hover text
        hover_text = self._create_hover_text(frame, particle_ids)

        # Add scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker=dict(
                    size=sizes * self.config.particle_size_scale,
                    color=colors,
                    opacity=self.config.opacity,
                    line=dict(width=0.5, color="DarkSlateGrey"),
                ),
                text=hover_text,
                hovertemplate="<b>Particle %{text}</b><br>"
                + "Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>"
                + "<extra></extra>",
                name="Particles",
            )
        )

        # Add trajectory traces if enabled
        if self.config.show_trajectory_traces:
            self._add_trajectory_traces(fig, frame)

    def _add_density_field(self, fig: go.Figure, frame: VisualizationFrame) -> None:
        """Add density field visualization."""
        if not frame.box_dimensions:
            return

        # Create 3D grid
        nx, ny, nz = 20, 20, 20
        x = np.linspace(0, frame.box_dimensions[0], nx)
        y = np.linspace(0, frame.box_dimensions[1], ny)
        z = np.linspace(0, frame.box_dimensions[2], nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Calculate density field
        density = self._calculate_density_field(frame, X, Y, Z)

        # Add isosurface
        fig.add_trace(
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=density.flatten(),
                isomin=density.min(),
                isomax=density.max(),
                surface_count=5,
                colorscale="Viridis",
                showscale=True,
                name="Density Field",
            )
        )

    def _add_structure_evolution(
        self, fig: go.Figure, frame: VisualizationFrame
    ) -> None:
        """Add structure evolution visualization."""
        if not frame.structure_metrics:
            return

        # Add cluster visualization if cluster data available
        if frame.cluster_ids is not None:
            unique_clusters = np.unique(frame.cluster_ids)

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue

                mask = frame.cluster_ids == cluster_id
                cluster_positions = frame.positions[mask]

                # Add cluster as separate trace
                fig.add_trace(
                    go.Scatter3d(
                        x=cluster_positions[:, 0],
                        y=cluster_positions[:, 1],
                        z=cluster_positions[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=self.color_maps["cluster"][
                                cluster_id % len(self.color_maps["cluster"])
                            ],
                            opacity=0.8,
                        ),
                        name=f"Cluster {cluster_id}",
                    )
                )

    def _get_particle_colors(
        self, frame: VisualizationFrame, color_by: ColorScheme
    ) -> List[str]:
        """Get particle colors based on color scheme."""
        if color_by == ColorScheme.PARTICLE_TYPE:
            return [
                self.color_maps["particle_type"].get(pt, "#888888")
                for pt in frame.particle_types
            ]

        elif color_by == ColorScheme.ENERGY and frame.energies is not None:
            # Normalize energies to [0, 1] range
            energies_norm = (frame.energies - frame.energies.min()) / (
                frame.energies.max() - frame.energies.min()
            )
            return [
                self.color_maps["energy"][int(e * (len(self.color_maps["energy"]) - 1))]
                for e in energies_norm
            ]

        elif color_by == ColorScheme.VELOCITY and frame.velocities is not None:
            # Calculate speed
            speeds = np.linalg.norm(frame.velocities, axis=1)
            speeds_norm = (speeds - speeds.min()) / (speeds.max() - speeds.min())
            return [
                self.color_maps["velocity"][
                    int(s * (len(self.color_maps["velocity"]) - 1))
                ]
                for s in speeds_norm
            ]

        elif color_by == ColorScheme.CLUSTER_ID and frame.cluster_ids is not None:
            return [
                self.color_maps["cluster"][cid % len(self.color_maps["cluster"])]
                for cid in frame.cluster_ids
            ]

        else:
            # Default color
            return ["#888888"] * len(frame.positions)

    def _get_particle_sizes(self, particle_types: np.ndarray) -> np.ndarray:
        """Get particle sizes based on type."""
        size_map = {
            "hydrophilic": 5,
            "hydrophobic": 5,
            "amphiphilic": 6,
            "solvent": 3,
            "ion": 4,
            "surfactant_head": 6,
            "surfactant_tail": 4,
        }

        return np.array([size_map.get(pt, 5) for pt in particle_types])

    def _create_hover_text(
        self, frame: VisualizationFrame, particle_ids: np.ndarray
    ) -> List[str]:
        """Create hover text for particles."""
        hover_texts = []

        for i, pid in enumerate(particle_ids):
            text = f"ID: {pid}<br>Type: {frame.particle_types[i]}"

            if frame.energies is not None:
                text += f"<br>Energy: {frame.energies[i]:.3f}"

            if frame.velocities is not None:
                speed = np.linalg.norm(frame.velocities[i])
                text += f"<br>Speed: {speed:.3f}"

            if frame.cluster_ids is not None:
                text += f"<br>Cluster: {frame.cluster_ids[i]}"

            hover_texts.append(text)

        return hover_texts

    def _calculate_density_field(
        self, frame: VisualizationFrame, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        """Calculate density field on a 3D grid."""
        density = np.zeros_like(X)
        sigma = 2.0  # Gaussian width

        for pos in frame.positions:
            # Calculate Gaussian contribution
            r_squared = (X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2
            density += np.exp(-r_squared / (2 * sigma**2))

        return density

    def _add_trajectory_traces(self, fig: go.Figure, frame: VisualizationFrame) -> None:
        """Add trajectory traces for particle movement."""
        # This would require storing historical positions
        # For now, just add placeholder
        pass

    def _update_3d_layout(self, fig: go.Figure, frame: VisualizationFrame) -> None:
        """Update 3D layout settings."""
        if frame.box_dimensions:
            # Set equal aspect ratio based on simulation box
            max_dim = max(frame.box_dimensions)

            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title="X (nm)",
                        range=[0, frame.box_dimensions[0]],
                        showbackground=True,
                        backgroundcolor="rgba(0,0,0,0.1)",
                    ),
                    yaxis=dict(
                        title="Y (nm)",
                        range=[0, frame.box_dimensions[1]],
                        showbackground=True,
                        backgroundcolor="rgba(0,0,0,0.1)",
                    ),
                    zaxis=dict(
                        title="Z (nm)",
                        range=[0, frame.box_dimensions[2]],
                        showbackground=True,
                        backgroundcolor="rgba(0,0,0,0.1)",
                    ),
                    aspectmode="manual",
                    aspectratio=dict(
                        x=frame.box_dimensions[0] / max_dim,
                        y=frame.box_dimensions[1] / max_dim,
                        z=frame.box_dimensions[2] / max_dim,
                    ),
                ),
                width=self.config.width,
                height=self.config.height,
                title=f"Self-Assembly Simulation - Frame {self.current_frame}",
                showlegend=True,
            )
        else:
            # Default layout
            fig.update_layout(
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                width=self.config.width,
                height=self.config.height,
                title="Self-Assembly Simulation",
                showlegend=True,
            )

    def animate_trajectory(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
        mode: VisualizationMode = VisualizationMode.PARTICLES_3D,
    ) -> go.Figure:
        """
        Create animated visualization of trajectory.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all frames)
            step: Step size between frames
            mode: Visualization mode

        Returns:
            Animated Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly not available for animation")

        if not self.trajectory_data:
            raise VisualizationError("No trajectory data loaded")

        if end_frame is None:
            end_frame = len(self.trajectory_data)

        frames = []
        frame_indices = range(
            start_frame, min(end_frame, len(self.trajectory_data)), step
        )

        for i in frame_indices:
            frame_data = self.trajectory_data[i]

            # Create frame traces
            frame_traces = []

            if mode == VisualizationMode.PARTICLES_3D:
                colors = self._get_particle_colors(
                    frame_data, ColorScheme.PARTICLE_TYPE
                )
                sizes = self._get_particle_sizes(frame_data.particle_types)

                frame_traces.append(
                    go.Scatter3d(
                        x=frame_data.positions[:, 0],
                        y=frame_data.positions[:, 1],
                        z=frame_data.positions[:, 2],
                        mode="markers",
                        marker=dict(
                            size=sizes * self.config.particle_size_scale,
                            color=colors,
                            opacity=self.config.opacity,
                        ),
                        name="Particles",
                    )
                )

            frames.append(
                go.Frame(
                    data=frame_traces,
                    name=str(i),
                    layout=go.Layout(
                        title=f"Frame {i} - Time: {frame_data.timestamp:.2f}"
                    ),
                )
            )

        # Create initial figure
        fig = self.create_3d_visualization(start_frame, mode)

        # Add frames
        fig.frames = frames

        # Add animation controls
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": self.config.frame_interval,
                                        "redraw": True,
                                    },
                                    "transition": {"duration": 50},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                    "x": 0.1,
                    "y": 0,
                    "xanchor": "right",
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [str(i)],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": str(i),
                            "method": "animate",
                        }
                        for i in frame_indices
                    ],
                    "active": 0,
                    "x": 0.1,
                    "len": 0.9,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )

        return fig

    def export_visualization(
        self, filename: str, format: ExportFormat = ExportFormat.PNG, **kwargs
    ) -> None:
        """
        Export current visualization to file.

        Args:
            filename: Output filename
            format: Export format
            **kwargs: Additional export parameters
        """
        if not self.figure:
            raise ExportError("No figure to export")

        try:
            if format == ExportFormat.PNG:
                self.figure.write_image(
                    filename,
                    format="png",
                    width=kwargs.get("width", self.config.width),
                    height=kwargs.get("height", self.config.height),
                    scale=kwargs.get("scale", self.config.export_dpi / 100),
                )

            elif format == ExportFormat.SVG:
                self.figure.write_image(filename, format="svg")

            elif format == ExportFormat.PDF:
                self.figure.write_image(filename, format="pdf")

            elif format == ExportFormat.HTML:
                self.figure.write_html(
                    filename, include_plotlyjs=kwargs.get("include_plotlyjs", True)
                )

            else:
                raise ExportError(f"Unsupported export format: {format}")

        except Exception as e:
            raise ExportError(f"Export failed: {str(e)}")


# =============================================================================
# Property Tracking Dashboard
# =============================================================================


class PropertyTrackingDashboard:
    """
    Interactive dashboard for tracking emergent properties over time
    and across different design parameters.
    """

    def __init__(self, config: Optional[DashboardConfig] = None, port: int = 8050):
        """
        Initialize the dashboard.

        Args:
            config: Dashboard configuration
            port: Port for web server
        """
        self.config = config or DashboardConfig()
        self.port = port

        # Data storage
        self.time_series_data: Dict[str, TimeSeriesData] = {}
        self.parameter_sweep_data: Optional[ParameterSweepData] = None
        self.phase_transitions: List[PhaseTransitionEvent] = []

        # Dashboard app
        self.app = None
        self.server_running = False

        if not DASH_AVAILABLE:
            warnings.warn("Dash not available. Interactive dashboard will be limited.")

    def add_time_series(
        self,
        name: str,
        timestamps: np.ndarray,
        properties: Dict[str, np.ndarray],
        labels: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Add time series data for tracking.

        Args:
            name: Name of the time series
            timestamps: Time points
            properties: Dictionary of property arrays
            labels: Property labels for display
            units: Property units
        """
        self.time_series_data[name] = TimeSeriesData(
            timestamps=timestamps,
            properties=properties,
            labels=labels or {},
            units=units or {},
        )

    def add_parameter_sweep(
        self,
        parameter_names: List[str],
        parameter_values: np.ndarray,
        property_maps: Dict[str, np.ndarray],
    ) -> None:
        """
        Add parameter sweep data.

        Args:
            parameter_names: Names of swept parameters
            parameter_values: Parameter value arrays
            property_maps: Property maps over parameter space
        """
        self.parameter_sweep_data = ParameterSweepData(
            parameter_names=parameter_names,
            parameter_values=parameter_values,
            property_maps=property_maps,
        )

    def add_phase_transition(
        self,
        timestamp: float,
        from_phase: PhaseType,
        to_phase: PhaseType,
        transition_probability: float,
        **kwargs,
    ) -> None:
        """
        Add phase transition event.

        Args:
            timestamp: When transition occurred
            from_phase: Initial phase
            to_phase: Final phase
            transition_probability: Probability of transition
            **kwargs: Additional transition data
        """
        event = PhaseTransitionEvent(
            timestamp=timestamp,
            from_phase=from_phase,
            to_phase=to_phase,
            transition_probability=transition_probability,
            **kwargs,
        )
        self.phase_transitions.append(event)

    def create_dashboard(self) -> None:
        """Create the interactive dashboard."""
        if not DASH_AVAILABLE:
            raise DashboardError("Dash not available for dashboard creation")

        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Create layout
        self.app.layout = self._create_layout()

        # Register callbacks
        self._register_callbacks()

    def _create_layout(self):
        """Create dashboard layout."""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Self-Assembly Property Tracking Dashboard",
                                    className="text-center mb-4",
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                # Control panel
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Controls"),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Time Series:"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="time-series-selector",
                                                                    options=[
                                                                        {
                                                                            "label": name,
                                                                            "value": name,
                                                                        }
                                                                        for name in self.time_series_data.keys()
                                                                    ],
                                                                    value=(
                                                                        list(
                                                                            self.time_series_data.keys()
                                                                        )[0]
                                                                        if self.time_series_data
                                                                        else None
                                                                    ),
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Properties:"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="property-selector",
                                                                    multi=True,
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ]
                                                ),
                                                html.Br(),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Real-time Updates",
                                                                    id="realtime-toggle",
                                                                    color="primary",
                                                                    size="sm",
                                                                    outline=True,
                                                                )
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Export Data",
                                                                    id="export-button",
                                                                    color="secondary",
                                                                    size="sm",
                                                                )
                                                            ],
                                                            width=6,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),
                # Main content area
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                # Time series plots
                                dcc.Graph(
                                    id="time-series-plot",
                                    style={"height": f"{self.config.chart_height}px"},
                                )
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                # Property statistics
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Property Statistics"),
                                        dbc.CardBody(id="property-stats"),
                                    ]
                                )
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                # Phase transition tracking
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Phase Transitions"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="phase-transition-plot",
                                                    style={
                                                        "height": f"{self.config.chart_height}px"
                                                    },
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Parameter Space"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="parameter-space-plot",
                                                    style={
                                                        "height": f"{self.config.chart_height}px"
                                                    },
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
                # Auto-refresh component
                dcc.Interval(
                    id="interval-component",
                    interval=self.config.update_interval,
                    n_intervals=0,
                    disabled=not self.config.real_time_updates,
                ),
                # Hidden div for storing data
                html.Div(id="hidden-div", style={"display": "none"}),
            ],
            fluid=self.config.fluid_layout,
        )

    def _register_callbacks(self):
        """Register dashboard callbacks."""

        @self.app.callback(
            Output("property-selector", "options"),
            Input("time-series-selector", "value"),
        )
        def update_property_options(selected_series):
            if not selected_series or selected_series not in self.time_series_data:
                return []

            properties = self.time_series_data[selected_series].properties
            return [{"label": prop, "value": prop} for prop in properties.keys()]

        @self.app.callback(
            Output("property-selector", "value"), Input("property-selector", "options")
        )
        def set_default_properties(options):
            if not options:
                return []
            return [opt["value"] for opt in options[:3]]  # Default to first 3

        @self.app.callback(
            [
                Output("time-series-plot", "figure"),
                Output("property-stats", "children"),
            ],
            [
                Input("time-series-selector", "value"),
                Input("property-selector", "value"),
                Input("interval-component", "n_intervals"),
            ],
        )
        def update_time_series_plot(selected_series, selected_properties, n_intervals):
            if not selected_series or not selected_properties:
                return {}, "No data selected"

            # Create time series plot
            fig = self._create_time_series_plot(selected_series, selected_properties)

            # Create statistics
            stats = self._create_property_statistics(
                selected_series, selected_properties
            )

            return fig, stats

        @self.app.callback(
            Output("phase-transition-plot", "figure"),
            Input("interval-component", "n_intervals"),
        )
        def update_phase_transition_plot(n_intervals):
            return self._create_phase_transition_plot()

        @self.app.callback(
            Output("parameter-space-plot", "figure"),
            Input("interval-component", "n_intervals"),
        )
        def update_parameter_space_plot(n_intervals):
            return self._create_parameter_space_plot()

    def _create_time_series_plot(
        self, series_name: str, properties: List[str]
    ) -> go.Figure:
        """Create time series plot."""
        if series_name not in self.time_series_data:
            return go.Figure()

        data = self.time_series_data[series_name]

        fig = make_subplots(
            rows=len(properties),
            cols=1,
            shared_xaxes=True,
            subplot_titles=properties,
            vertical_spacing=0.05,
        )

        for i, prop in enumerate(properties, 1):
            if prop not in data.properties:
                continue

            fig.add_trace(
                go.Scatter(
                    x=data.timestamps,
                    y=data.properties[prop],
                    mode="lines+markers",
                    name=data.labels.get(prop, prop),
                    line=dict(width=2),
                    marker=dict(size=4),
                ),
                row=i,
                col=1,
            )

            # Add units to y-axis label
            ylabel = data.labels.get(prop, prop)
            if prop in data.units:
                ylabel += f" ({data.units[prop]})"

            fig.update_yaxes(title_text=ylabel, row=i, col=1)

        fig.update_xaxes(title_text="Time", row=len(properties), col=1)
        fig.update_layout(
            height=self.config.chart_height * len(properties),
            showlegend=True,
            title=f"Property Evolution: {series_name}",
        )

        return fig

    def _create_property_statistics(
        self, series_name: str, properties: List[str]
    ) -> List:
        """Create property statistics summary."""
        if series_name not in self.time_series_data:
            return ["No data available"]

        data = self.time_series_data[series_name]
        stats_components = []

        for prop in properties:
            if prop not in data.properties:
                continue

            values = data.properties[prop]

            stats_components.extend(
                [
                    html.H6(f"{data.labels.get(prop, prop)}", className="mb-2"),
                    html.P(
                        [
                            html.Strong("Mean: "),
                            f"{np.mean(values):.3f}",
                            html.Br(),
                            html.Strong("Std: "),
                            f"{np.std(values):.3f}",
                            html.Br(),
                            html.Strong("Min: "),
                            f"{np.min(values):.3f}",
                            html.Br(),
                            html.Strong("Max: "),
                            f"{np.max(values):.3f}",
                        ],
                        className="mb-3",
                    ),
                ]
            )

        return stats_components

    def _create_phase_transition_plot(self) -> go.Figure:
        """Create phase transition visualization."""
        if not self.phase_transitions:
            return go.Figure().add_annotation(
                text="No phase transitions detected",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Create timeline plot
        fig = go.Figure()

        timestamps = [event.timestamp for event in self.phase_transitions]
        from_phases = [event.from_phase.value for event in self.phase_transitions]
        to_phases = [event.to_phase.value for event in self.phase_transitions]
        probabilities = [
            event.transition_probability for event in self.phase_transitions
        ]

        # Add transition events
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=probabilities,
                mode="markers+text",
                marker=dict(
                    size=15,
                    color=probabilities,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Transition Probability"),
                ),
                text=[f"{fp}→{tp}" for fp, tp in zip(from_phases, to_phases)],
                textposition="top center",
                hovertemplate="<b>Transition</b><br>"
                + "Time: %{x:.2f}<br>"
                + "Probability: %{y:.3f}<br>"
                + "Type: %{text}<br>"
                + "<extra></extra>",
                name="Phase Transitions",
            )
        )

        fig.update_layout(
            title="Phase Transition Timeline",
            xaxis_title="Time",
            yaxis_title="Transition Probability",
            height=self.config.chart_height,
        )

        return fig

    def _create_parameter_space_plot(self) -> go.Figure:
        """Create parameter space visualization."""
        if not self.parameter_sweep_data:
            return go.Figure().add_annotation(
                text="No parameter sweep data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        data = self.parameter_sweep_data

        # Create 2D parameter space plot (first two parameters)
        if len(data.parameter_names) >= 2:
            # Get first property for visualization
            prop_name = list(data.property_maps.keys())[0]
            prop_values = data.property_maps[prop_name]

            fig = go.Figure()

            fig.add_trace(
                go.Heatmap(
                    x=data.parameter_values[0],
                    y=data.parameter_values[1],
                    z=prop_values,
                    colorscale="Viridis",
                    colorbar=dict(title=prop_name),
                )
            )

            fig.update_layout(
                title=f"Parameter Space: {prop_name}",
                xaxis_title=data.parameter_names[0],
                yaxis_title=data.parameter_names[1],
                height=self.config.chart_height,
            )

            return fig

        return go.Figure()

    def run_dashboard(self, debug: bool = False) -> None:
        """
        Run the dashboard server.

        Args:
            debug: Whether to run in debug mode
        """
        if not DASH_AVAILABLE:
            raise DashboardError("Dash not available")

        if not self.app:
            self.create_dashboard()

        try:
            self.server_running = True
            self.app.run_server(debug=debug, port=self.port, host="127.0.0.1")
        except Exception as e:
            self.server_running = False
            raise DashboardError(f"Failed to start dashboard: {str(e)}")

    def stop_dashboard(self) -> None:
        """Stop the dashboard server."""
        self.server_running = False


# =============================================================================
# Phase Transition Analyzer
# =============================================================================


class PhaseTransitionAnalyzer:
    """
    Specialized analyzer for visualizing and tracking phase transitions
    in self-assembly systems.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the analyzer.

        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu

        # Storage for analysis results
        self.transition_data: List[PhaseTransitionEvent] = []
        self.order_parameters: Dict[str, np.ndarray] = {}
        self.phase_diagrams: Dict[str, np.ndarray] = {}

    def analyze_phase_transitions(
        self,
        trajectory: List[TrajectoryFrame],
        structure_data: List[CharacterizationResult],
        temperature_range: Optional[Tuple[float, float]] = None,
        concentration_range: Optional[Tuple[float, float]] = None,
    ) -> List[PhaseTransitionEvent]:
        """
        Analyze phase transitions in trajectory data.

        Args:
            trajectory: Simulation trajectory
            structure_data: Structure characterization results
            temperature_range: Temperature range for analysis
            concentration_range: Concentration range for analysis

        Returns:
            List of detected phase transition events
        """
        transitions = []

        # Track phase changes
        previous_phase = None

        for i, (frame, char_result) in enumerate(zip(trajectory, structure_data)):
            current_phase = self._identify_dominant_phase(char_result)

            if previous_phase and current_phase != previous_phase:
                # Calculate transition probability
                prob = self._calculate_transition_probability(
                    previous_phase, current_phase, char_result
                )

                # Calculate order parameter
                order_param = self._calculate_order_parameter(char_result)

                transition = PhaseTransitionEvent(
                    timestamp=frame.timestamp,
                    from_phase=previous_phase,
                    to_phase=current_phase,
                    transition_probability=prob,
                    order_parameter=order_param,
                )

                transitions.append(transition)

            previous_phase = current_phase

        self.transition_data = transitions
        return transitions

    def _identify_dominant_phase(
        self, char_result: CharacterizationResult
    ) -> PhaseType:
        """Identify the dominant phase from characterization results."""
        # Simple heuristic based on structure metrics
        metrics = char_result.structure_metrics

        if hasattr(metrics, "micelle_count") and metrics.micelle_count > 0:
            return PhaseType.MICELLE
        elif hasattr(metrics, "vesicle_count") and metrics.vesicle_count > 0:
            return PhaseType.VESICLE
        elif hasattr(metrics, "bilayer_thickness") and metrics.bilayer_thickness > 0:
            return PhaseType.BILAYER
        else:
            return PhaseType.LIQUID

    def _calculate_transition_probability(
        self,
        from_phase: PhaseType,
        to_phase: PhaseType,
        char_result: CharacterizationResult,
    ) -> float:
        """Calculate transition probability between phases."""
        # Simplified probability calculation
        metrics = char_result.structure_metrics

        # Base probability on structure stability metrics
        if hasattr(metrics, "cluster_stability"):
            return float(metrics.cluster_stability)
        else:
            return 0.5  # Default probability

    def _calculate_order_parameter(self, char_result: CharacterizationResult) -> float:
        """Calculate order parameter for phase characterization."""
        metrics = char_result.structure_metrics

        # Simple order parameter based on clustering
        if hasattr(metrics, "cluster_compactness"):
            return float(metrics.cluster_compactness)
        else:
            return 0.0

    def create_phase_diagram(
        self,
        parameter1_values: np.ndarray,
        parameter2_values: np.ndarray,
        phase_map: np.ndarray,
        parameter1_name: str = "Temperature",
        parameter2_name: str = "Concentration",
    ) -> go.Figure:
        """
        Create interactive phase diagram.

        Args:
            parameter1_values: First parameter values
            parameter2_values: Second parameter values
            phase_map: 2D array of phase identities
            parameter1_name: Name of first parameter
            parameter2_name: Name of second parameter

        Returns:
            Interactive phase diagram figure
        """
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly not available for phase diagrams")

        fig = go.Figure()

        # Create phase diagram heatmap
        fig.add_trace(
            go.Heatmap(
                x=parameter1_values,
                y=parameter2_values,
                z=phase_map,
                colorscale="viridis",
                showscale=True,
                colorbar=dict(
                    title="Phase",
                    tickmode="array",
                    tickvals=list(range(len(PhaseType))),
                    ticktext=[phase.value for phase in PhaseType],
                ),
            )
        )

        # Add phase transition boundaries
        self._add_phase_boundaries(fig, parameter1_values, parameter2_values, phase_map)

        fig.update_layout(
            title="Phase Diagram",
            xaxis_title=parameter1_name,
            yaxis_title=parameter2_name,
            width=800,
            height=600,
        )

        return fig

    def _add_phase_boundaries(
        self, fig: go.Figure, param1: np.ndarray, param2: np.ndarray, phases: np.ndarray
    ) -> None:
        """Add phase boundary lines to diagram."""
        # Simple boundary detection using gradients
        grad_x = np.gradient(phases, axis=1)
        grad_y = np.gradient(phases, axis=0)

        boundary_mask = (np.abs(grad_x) > 0.1) | (np.abs(grad_y) > 0.1)

        if np.any(boundary_mask):
            y_indices, x_indices = np.where(boundary_mask)

            fig.add_trace(
                go.Scatter(
                    x=param1[x_indices],
                    y=param2[y_indices],
                    mode="markers",
                    marker=dict(size=2, color="red", symbol="square"),
                    name="Phase Boundaries",
                    showlegend=True,
                )
            )

    def create_transition_network(self) -> go.Figure:
        """Create network visualization of phase transitions."""
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly not available for network visualization")

        if not self.transition_data:
            return go.Figure().add_annotation(
                text="No transition data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Count transitions between phases
        transition_counts = {}
        phases = list(PhaseType)

        for event in self.transition_data:
            key = (event.from_phase, event.to_phase)
            transition_counts[key] = transition_counts.get(key, 0) + 1

        # Create network layout
        n_phases = len(phases)
        angles = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)

        # Node positions
        node_x = np.cos(angles)
        node_y = np.sin(angles)

        fig = go.Figure()

        # Add edges (transitions)
        for (from_phase, to_phase), count in transition_counts.items():
            from_idx = phases.index(from_phase)
            to_idx = phases.index(to_phase)

            # Draw arrow from from_phase to to_phase
            fig.add_trace(
                go.Scatter(
                    x=[node_x[from_idx], node_x[to_idx]],
                    y=[node_y[from_idx], node_y[to_idx]],
                    mode="lines",
                    line=dict(width=count * 2, color="gray"),
                    hovertemplate=f"{from_phase.value} → {to_phase.value}<br>"
                    + f"Count: {count}<extra></extra>",
                    showlegend=False,
                )
            )

        # Add nodes (phases)
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=30,
                    color=list(range(n_phases)),
                    colorscale="viridis",
                    line=dict(width=2, color="white"),
                ),
                text=[phase.value for phase in phases],
                textposition="middle center",
                hovertemplate="Phase: %{text}<extra></extra>",
                name="Phases",
            )
        )

        fig.update_layout(
            title="Phase Transition Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=600,
            height=600,
        )

        return fig


# =============================================================================
# Utility Functions
# =============================================================================


def create_comparative_visualization(
    visualizers: List[InteractiveSelfAssemblyVisualizer],
    labels: List[str],
    sync_frames: bool = True,
) -> go.Figure:
    """
    Create comparative visualization of multiple simulations.

    Args:
        visualizers: List of visualizer instances
        labels: Labels for each simulation
        sync_frames: Whether to synchronize frame indices

    Returns:
        Comparative visualization figure
    """
    if not PLOTLY_AVAILABLE:
        raise VisualizationError("Plotly not available")

    n_sims = len(visualizers)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=n_sims,
        specs=[[{"type": "scatter3d"} for _ in range(n_sims)]],
        subplot_titles=labels,
        horizontal_spacing=0.02,
    )

    for i, (viz, label) in enumerate(zip(visualizers, labels)):
        if not viz.trajectory_data:
            continue

        # Get current frame
        frame_idx = viz.current_frame if not sync_frames else 0
        frame = viz.trajectory_data[frame_idx]

        # Add particles
        colors = viz._get_particle_colors(frame, ColorScheme.PARTICLE_TYPE)
        sizes = viz._get_particle_sizes(frame.particle_types)

        fig.add_trace(
            go.Scatter3d(
                x=frame.positions[:, 0],
                y=frame.positions[:, 1],
                z=frame.positions[:, 2],
                mode="markers",
                marker=dict(size=sizes, color=colors, opacity=0.8),
                name=label,
                showlegend=True,
            ),
            row=1,
            col=i + 1,
        )

        # Update subplot layout
        scene_name = f"scene{i+1}" if i > 0 else "scene"
        fig.update_layout(
            {
                scene_name: dict(
                    xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"
                )
            }
        )

    fig.update_layout(title="Comparative Self-Assembly Visualization", height=600)

    return fig


def export_dashboard_data(
    dashboard: PropertyTrackingDashboard, filename: str, format: str = "csv"
) -> None:
    """
    Export dashboard data to file.

    Args:
        dashboard: Dashboard instance
        filename: Output filename
        format: Export format ('csv', 'json', 'hdf5')
    """
    if format == "csv":
        # Export each time series to separate CSV
        for name, data in dashboard.time_series_data.items():
            df = pd.DataFrame({"timestamp": data.timestamps, **data.properties})

            base_name = filename.replace(".csv", "")
            df.to_csv(f"{base_name}_{name}.csv", index=False)

    elif format == "json":
        # Export all data to JSON
        export_data = {
            "time_series": {},
            "phase_transitions": [],
            "parameter_sweep": {},
        }

        # Time series data
        for name, data in dashboard.time_series_data.items():
            export_data["time_series"][name] = {
                "timestamps": data.timestamps.tolist(),
                "properties": {k: v.tolist() for k, v in data.properties.items()},
                "labels": data.labels,
                "units": data.units,
            }

        # Phase transitions
        for event in dashboard.phase_transitions:
            export_data["phase_transitions"].append(
                {
                    "timestamp": event.timestamp,
                    "from_phase": event.from_phase.value,
                    "to_phase": event.to_phase.value,
                    "transition_probability": event.transition_probability,
                    "order_parameter": event.order_parameter,
                }
            )

        # Parameter sweep
        if dashboard.parameter_sweep_data:
            sweep = dashboard.parameter_sweep_data
            export_data["parameter_sweep"] = {
                "parameter_names": sweep.parameter_names,
                "parameter_values": sweep.parameter_values.tolist(),
                "property_maps": {
                    k: v.tolist() for k, v in sweep.property_maps.items()
                },
            }

        import json

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

    else:
        raise ExportError(f"Unsupported export format: {format}")


# =============================================================================
# Example Usage and Demo Functions
# =============================================================================


def create_demo_visualization() -> InteractiveSelfAssemblyVisualizer:
    """Create a demo visualization with synthetic data."""

    # Create synthetic trajectory data
    n_frames = 50
    n_particles = 1000
    box_size = 20.0

    trajectory = []

    for frame_idx in range(n_frames):
        particles = []

        for i in range(n_particles):
            # Create particle with random position
            position = np.random.uniform(0, box_size, 3)

            # Assign particle type
            if i < n_particles // 3:
                ptype = "hydrophilic"
            elif i < 2 * n_particles // 3:
                ptype = "hydrophobic"
            else:
                ptype = "amphiphilic"

            particle = type(
                "Particle",
                (),
                {
                    "position": type(
                        "Position",
                        (),
                        {"x": position[0], "y": position[1], "z": position[2]},
                    )(),
                    "particle_type": type("ParticleType", (), {"name": ptype})(),
                    "particle_id": i,
                    "velocity": (
                        type("Velocity", (), {"x": 0, "y": 0, "z": 0})()
                        if frame_idx > 0
                        else None
                    ),
                    "energy": np.random.uniform(-1, 1),
                },
            )()

            particles.append(particle)

        # Create trajectory frame
        frame = type(
            "TrajectoryFrame",
            (),
            {
                "timestamp": frame_idx * 0.1,
                "particles": particles,
                "box": type(
                    "Box",
                    (),
                    {"length": box_size, "width": box_size, "height": box_size},
                )(),
                "total_energy": np.random.uniform(-100, 100),
                "temperature": 300.0,
                "pressure": 1.0,
            },
        )()

        trajectory.append(frame)

    # Create visualizer and load data
    visualizer = InteractiveSelfAssemblyVisualizer()
    visualizer.load_trajectory(trajectory)

    return visualizer


def run_demo_dashboard(port: int = 8050) -> PropertyTrackingDashboard:
    """Run a demo dashboard with synthetic data."""

    # Create dashboard
    dashboard = PropertyTrackingDashboard(port=port)

    # Generate synthetic time series data
    timestamps = np.linspace(0, 100, 1000)

    # Micelle formation simulation
    dashboard.add_time_series(
        name="Micelle Formation",
        timestamps=timestamps,
        properties={
            "micelle_count": 10
            + 5 * np.sin(timestamps / 10)
            + np.random.normal(0, 1, len(timestamps)),
            "average_size": 20
            + 10 * np.cos(timestamps / 15)
            + np.random.normal(0, 2, len(timestamps)),
            "aggregation_number": 50
            + 20 * np.sin(timestamps / 8)
            + np.random.normal(0, 3, len(timestamps)),
        },
        labels={
            "micelle_count": "Number of Micelles",
            "average_size": "Average Micelle Size",
            "aggregation_number": "Aggregation Number",
        },
        units={
            "micelle_count": "count",
            "average_size": "nm",
            "aggregation_number": "molecules",
        },
    )

    # Vesicle formation simulation
    dashboard.add_time_series(
        name="Vesicle Formation",
        timestamps=timestamps,
        properties={
            "vesicle_count": 3
            + 2 * np.sin(timestamps / 20)
            + np.random.normal(0, 0.5, len(timestamps)),
            "membrane_thickness": 4
            + 0.5 * np.cos(timestamps / 12)
            + np.random.normal(0, 0.1, len(timestamps)),
            "curvature": 0.1
            + 0.05 * np.sin(timestamps / 18)
            + np.random.normal(0, 0.01, len(timestamps)),
        },
        labels={
            "vesicle_count": "Number of Vesicles",
            "membrane_thickness": "Membrane Thickness",
            "curvature": "Mean Curvature",
        },
        units={
            "vesicle_count": "count",
            "membrane_thickness": "nm",
            "curvature": "1/nm",
        },
    )

    # Add synthetic phase transitions
    for i in range(5):
        timestamp = 20 + i * 15 + np.random.uniform(-5, 5)
        phases = list(PhaseType)
        from_phase = np.random.choice(phases)
        to_phase = np.random.choice([p for p in phases if p != from_phase])

        dashboard.add_phase_transition(
            timestamp=timestamp,
            from_phase=from_phase,
            to_phase=to_phase,
            transition_probability=np.random.uniform(0.3, 0.9),
            order_parameter=np.random.uniform(0.1, 0.8),
        )

    # Add parameter sweep data
    temp_range = np.linspace(280, 350, 20)
    conc_range = np.linspace(0.1, 1.0, 20)
    T, C = np.meshgrid(temp_range, conc_range)

    # Synthetic property map (phase stability)
    stability = np.sin(T / 50) * np.cos(C * 5) + np.random.normal(0, 0.1, T.shape)

    dashboard.add_parameter_sweep(
        parameter_names=["Temperature", "Concentration"],
        parameter_values=np.array([temp_range, conc_range]),
        property_maps={"stability": stability},
    )

    return dashboard


if __name__ == "__main__":
    # Demo usage
    print("Creating demo visualization...")
    visualizer = create_demo_visualization()

    # Create 3D visualization
    fig = visualizer.create_3d_visualization()

    if PLOTLY_AVAILABLE:
        fig.show()

    print("Demo visualization created successfully!")

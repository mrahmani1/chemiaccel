"""
Interactive Visualization of Dynamic Material Response

This module creates interactive visualizations that display the material's dynamic
response, damage evolution, and property changes over time under specified extreme conditions.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.animation as animation
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, Slider
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    animation = None
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    pyo = None
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import HTML, display

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    widgets = None
    display = None
    HTML = None
    IPYWIDGETS_AVAILABLE = False

from ..core.exceptions import ChemiAccelError, VisualizationError
from ..utils.logging import get_logger
from .experimental_integration import ExperimentalDataset
from .property_characterization import (
    DynamicProperty,
    FailureAnalysis,
    FailureMechanism,
    PropertyType,
)
from .simulation_engine import ExtremeConditionType, SimulationResults

logger = get_logger(__name__)


class VisualizationType(Enum):
    """Types of visualizations available."""

    TIME_SERIES = "time_series"
    PHASE_DIAGRAM = "phase_diagram"
    STRESS_STRAIN = "stress_strain"
    DAMAGE_EVOLUTION = "damage_evolution"
    FAILURE_MAP = "failure_map"
    PROPERTY_CORRELATION = "property_correlation"
    ANIMATION_3D = "animation_3d"
    INTERACTIVE_DASHBOARD = "interactive_dashboard"
    COMPARISON_PLOT = "comparison_plot"
    STATISTICAL_ANALYSIS = "statistical_analysis"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""

    # Plot appearance
    figure_size: Tuple[int, int] = (12, 8)
    color_scheme: str = "viridis"
    background_color: str = "white"
    grid_enabled: bool = True

    # Animation settings
    animation_fps: int = 30
    animation_duration: float = 10.0  # seconds
    frame_skip: int = 1

    # Interactive features
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_hover: bool = True
    show_controls: bool = True

    # Output settings
    output_format: str = "html"  # html, png, svg, pdf
    output_quality: str = "high"  # low, medium, high
    save_animation: bool = False


class BaseVisualizer:
    """Base class for all visualizers."""

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.logger = get_logger(self.__class__.__name__)

        # Check available backends
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        self.plotly_available = PLOTLY_AVAILABLE
        self.seaborn_available = SEABORN_AVAILABLE

        if not (self.matplotlib_available or self.plotly_available):
            raise VisualizationError("No visualization backend available")

    def _setup_matplotlib_style(self):
        """Setup matplotlib style."""
        if not self.matplotlib_available:
            return

        plt.style.use("default")
        if self.seaborn_available:
            sns.set_palette(self.config.color_scheme)

    def _create_plotly_layout(self, title: str = "") -> Dict[str, Any]:
        """Create standard Plotly layout."""
        return {
            "title": title,
            "showlegend": True,
            "hovermode": "closest" if self.config.enable_hover else False,
            "plot_bgcolor": self.config.background_color,
            "paper_bgcolor": self.config.background_color,
            "font": {"size": 12},
            "width": self.config.figure_size[0] * 80,
            "height": self.config.figure_size[1] * 80,
        }


class DynamicResponseVisualizer(BaseVisualizer):
    """Visualizer for dynamic material response."""

    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)

    def create_time_series_plot(
        self,
        simulation_results: SimulationResults,
        properties: List[str] = None,
        interactive: bool = True,
    ) -> Union[str, Any]:
        """Create time series plot of dynamic properties."""

        self.logger.info("Creating time series visualization")

        if properties is None:
            properties = ["pressure", "temperature", "damage_fraction"]

        # Extract time series data
        time_data = self._extract_time_series(simulation_results, properties)

        if interactive and self.plotly_available:
            return self._create_plotly_time_series(time_data, properties)
        elif self.matplotlib_available:
            return self._create_matplotlib_time_series(time_data, properties)
        else:
            raise VisualizationError("No visualization backend available")

    def create_stress_strain_visualization(
        self, simulation_results: SimulationResults, interactive: bool = True
    ) -> Union[str, Any]:
        """Create stress-strain curve visualization."""

        self.logger.info("Creating stress-strain visualization")

        if (
            simulation_results.stress_tensor.size == 0
            or simulation_results.strain_tensor.size == 0
        ):
            raise VisualizationError("No stress-strain data available")

        # Calculate von Mises stress and equivalent strain
        stresses = []
        strains = []

        for i in range(len(simulation_results.stress_tensor)):
            stress = self._calculate_von_mises_stress(
                simulation_results.stress_tensor[i]
            )
            strain = self._calculate_equivalent_strain(
                simulation_results.strain_tensor[i]
            )
            stresses.append(stress)
            strains.append(strain)

        if interactive and self.plotly_available:
            return self._create_plotly_stress_strain(stresses, strains)
        elif self.matplotlib_available:
            return self._create_matplotlib_stress_strain(stresses, strains)
        else:
            raise VisualizationError("No visualization backend available")

    def create_damage_evolution_animation(
        self, simulation_results: SimulationResults, save_path: str = None
    ) -> Union[str, Any]:
        """Create animated damage evolution visualization."""

        self.logger.info("Creating damage evolution animation")

        if simulation_results.damage_fraction.size == 0:
            raise VisualizationError("No damage data available")

        if self.plotly_available:
            return self._create_plotly_damage_animation(simulation_results, save_path)
        elif self.matplotlib_available:
            return self._create_matplotlib_damage_animation(
                simulation_results, save_path
            )
        else:
            raise VisualizationError("No visualization backend available")

    def create_3d_structure_animation(
        self, simulation_results: SimulationResults, save_path: str = None
    ) -> Union[str, Any]:
        """Create 3D structure evolution animation."""

        self.logger.info("Creating 3D structure animation")

        if simulation_results.positions.size == 0:
            raise VisualizationError("No position data available")

        if self.plotly_available:
            return self._create_plotly_3d_animation(simulation_results, save_path)
        elif self.matplotlib_available:
            return self._create_matplotlib_3d_animation(simulation_results, save_path)
        else:
            raise VisualizationError("No visualization backend available")

    def _extract_time_series(
        self, results: SimulationResults, properties: List[str]
    ) -> Dict[str, np.ndarray]:
        """Extract time series data from simulation results."""

        time_data = {}
        n_steps = 0

        # Determine number of time steps
        if results.temperatures.size > 0:
            n_steps = len(results.temperatures)
        elif results.pressures.size > 0:
            n_steps = len(results.pressures)
        elif results.damage_fraction.size > 0:
            n_steps = len(results.damage_fraction)

        time_array = np.arange(n_steps)  # Placeholder time axis
        time_data["time"] = time_array

        for prop in properties:
            if prop == "pressure" and results.pressures.size > 0:
                time_data[prop] = results.pressures
            elif prop == "temperature" and results.temperatures.size > 0:
                time_data[prop] = results.temperatures
            elif prop == "energy" and results.energies.size > 0:
                time_data[prop] = results.energies
            elif prop == "damage_fraction" and results.damage_fraction.size > 0:
                time_data[prop] = results.damage_fraction
            elif prop == "defect_count" and results.defect_count.size > 0:
                time_data[prop] = results.defect_count
            elif prop == "crack_length" and results.crack_length.size > 0:
                time_data[prop] = results.crack_length
            elif prop == "stress" and results.stress_tensor.size > 0:
                # Calculate von Mises stress
                vm_stresses = []
                for stress_tensor in results.stress_tensor:
                    vm_stress = self._calculate_von_mises_stress(stress_tensor)
                    vm_stresses.append(vm_stress)
                time_data[prop] = np.array(vm_stresses)
            elif prop == "strain" and results.strain_tensor.size > 0:
                # Calculate equivalent strain
                eq_strains = []
                for strain_tensor in results.strain_tensor:
                    eq_strain = self._calculate_equivalent_strain(strain_tensor)
                    eq_strains.append(eq_strain)
                time_data[prop] = np.array(eq_strains)

        return time_data

    def _create_plotly_time_series(
        self, time_data: Dict[str, np.ndarray], properties: List[str]
    ) -> str:
        """Create Plotly time series plot."""

        fig = make_subplots(
            rows=len(properties),
            cols=1,
            subplot_titles=properties,
            vertical_spacing=0.1,
        )

        time_array = time_data.get("time", np.arange(100))

        for i, prop in enumerate(properties, 1):
            if prop in time_data:
                fig.add_trace(
                    go.Scatter(
                        x=time_array,
                        y=time_data[prop],
                        mode="lines",
                        name=prop,
                        line=dict(width=2),
                    ),
                    row=i,
                    col=1,
                )

        fig.update_layout(
            **self._create_plotly_layout("Dynamic Material Response"), showlegend=False
        )

        # Save as HTML
        html_str = pyo.plot(fig, output_type="div", include_plotlyjs="cdn")
        return html_str

    def _create_matplotlib_time_series(
        self, time_data: Dict[str, np.ndarray], properties: List[str]
    ) -> Any:
        """Create matplotlib time series plot."""

        self._setup_matplotlib_style()

        fig, axes = plt.subplots(
            len(properties), 1, figsize=self.config.figure_size, sharex=True
        )

        if len(properties) == 1:
            axes = [axes]

        time_array = time_data.get("time", np.arange(100))

        for i, prop in enumerate(properties):
            ax = axes[i]

            if prop in time_data:
                ax.plot(time_array, time_data[prop], linewidth=2, label=prop)
                ax.set_ylabel(prop.replace("_", " ").title())
                ax.grid(self.config.grid_enabled)

                if self.config.enable_zoom:
                    ax.set_xlim(np.min(time_array), np.max(time_array))

        axes[-1].set_xlabel("Time Step")
        plt.suptitle("Dynamic Material Response")
        plt.tight_layout()

        return fig

    def _create_plotly_stress_strain(
        self, stresses: List[float], strains: List[float]
    ) -> str:
        """Create Plotly stress-strain plot."""

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=strains,
                y=stresses,
                mode="lines+markers",
                name="Stress-Strain Curve",
                line=dict(width=3),
                marker=dict(size=4),
            )
        )

        # Add yield point and ultimate strength markers if detectable
        if len(stresses) > 10:
            max_stress_idx = np.argmax(stresses)
            fig.add_trace(
                go.Scatter(
                    x=[strains[max_stress_idx]],
                    y=[stresses[max_stress_idx]],
                    mode="markers",
                    name="Ultimate Strength",
                    marker=dict(size=10, color="red", symbol="diamond"),
                )
            )

        fig.update_layout(
            **self._create_plotly_layout("Stress-Strain Response"),
            xaxis_title="Equivalent Strain",
            yaxis_title="von Mises Stress (Pa)",
        )

        html_str = pyo.plot(fig, output_type="div", include_plotlyjs="cdn")
        return html_str

    def _create_matplotlib_stress_strain(
        self, stresses: List[float], strains: List[float]
    ) -> Any:
        """Create matplotlib stress-strain plot."""

        self._setup_matplotlib_style()

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        ax.plot(
            strains,
            stresses,
            linewidth=3,
            marker="o",
            markersize=3,
            label="Stress-Strain",
        )

        # Highlight ultimate strength
        if len(stresses) > 10:
            max_stress_idx = np.argmax(stresses)
            ax.plot(
                strains[max_stress_idx],
                stresses[max_stress_idx],
                "ro",
                markersize=10,
                label="Ultimate Strength",
            )

        ax.set_xlabel("Equivalent Strain")
        ax.set_ylabel("von Mises Stress (Pa)")
        ax.set_title("Stress-Strain Response")
        ax.grid(self.config.grid_enabled)
        ax.legend()

        plt.tight_layout()
        return fig

    def _create_plotly_damage_animation(
        self, results: SimulationResults, save_path: str = None
    ) -> str:
        """Create Plotly damage evolution animation."""

        frames = []
        time_steps = np.arange(len(results.damage_fraction))

        for i, damage in enumerate(results.damage_fraction):
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=time_steps[: i + 1],
                        y=results.damage_fraction[: i + 1],
                        mode="lines+markers",
                        name="Damage Evolution",
                        line=dict(color="red", width=3),
                    )
                ],
                name=str(i),
            )
            frames.append(frame)

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[time_steps[0]],
                    y=[results.damage_fraction[0]],
                    mode="lines+markers",
                    name="Damage Evolution",
                )
            ],
            frames=frames,
        )

        # Add animation controls
        fig.update_layout(
            **self._create_plotly_layout("Damage Evolution Animation"),
            xaxis_title="Time Step",
            yaxis_title="Damage Fraction",
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
                                        "duration": 1000 // self.config.animation_fps,
                                        "redraw": True,
                                    },
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
                }
            ],
        )

        html_str = pyo.plot(fig, output_type="div", include_plotlyjs="cdn")

        if save_path:
            with open(save_path, "w") as f:
                f.write(html_str)

        return html_str

    def _create_matplotlib_damage_animation(
        self, results: SimulationResults, save_path: str = None
    ) -> Any:
        """Create matplotlib damage evolution animation."""

        self._setup_matplotlib_style()

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        time_steps = np.arange(len(results.damage_fraction))

        ax.set_xlim(0, len(time_steps))
        ax.set_ylim(0, max(results.damage_fraction) * 1.1)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Damage Fraction")
        ax.set_title("Damage Evolution Animation")
        ax.grid(self.config.grid_enabled)

        (line,) = ax.plot([], [], "r-", linewidth=3, label="Damage")
        (points,) = ax.plot([], [], "ro", markersize=6)
        ax.legend()

        def animate(frame):
            line.set_data(time_steps[: frame + 1], results.damage_fraction[: frame + 1])
            if frame < len(time_steps):
                points.set_data([time_steps[frame]], [results.damage_fraction[frame]])
            return line, points

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(time_steps),
            interval=1000 // self.config.animation_fps,
            blit=True,
            repeat=True,
        )

        if save_path and self.config.save_animation:
            anim.save(save_path, writer="html")

        return anim

    def _create_plotly_3d_animation(
        self, results: SimulationResults, save_path: str = None
    ) -> str:
        """Create Plotly 3D structure animation."""

        if results.positions.size == 0:
            raise VisualizationError("No position data for 3D animation")

        frames = []

        for i, positions in enumerate(results.positions):
            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=positions[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=range(len(positions)),
                            colorscale="viridis",
                            opacity=0.8,
                        ),
                        name=f"Frame {i}",
                    )
                ],
                name=str(i),
            )
            frames.append(frame)

        # Initial frame
        initial_positions = (
            results.positions[0]
            if len(results.positions) > 0
            else np.array([[0, 0, 0]])
        )

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=initial_positions[:, 0],
                    y=initial_positions[:, 1],
                    z=initial_positions[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=range(len(initial_positions)),
                        colorscale="viridis",
                    ),
                )
            ],
            frames=frames,
        )

        fig.update_layout(
            **self._create_plotly_layout("3D Structure Evolution"),
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Z Position",
            ),
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
                                {"frame": {"duration": 200, "redraw": True}},
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {"frame": {"duration": 0, "redraw": False}},
                            ],
                        },
                    ],
                }
            ],
        )

        html_str = pyo.plot(fig, output_type="div", include_plotlyjs="cdn")

        if save_path:
            with open(save_path, "w") as f:
                f.write(html_str)

        return html_str

    def _create_matplotlib_3d_animation(
        self, results: SimulationResults, save_path: str = None
    ) -> Any:
        """Create matplotlib 3D structure animation."""

        self._setup_matplotlib_style()

        fig = plt.figure(figsize=self.config.figure_size)
        ax = fig.add_subplot(111, projection="3d")

        # Set axis limits
        all_positions = results.positions.reshape(-1, 3)
        ax.set_xlim(np.min(all_positions[:, 0]), np.max(all_positions[:, 0]))
        ax.set_ylim(np.min(all_positions[:, 1]), np.max(all_positions[:, 1]))
        ax.set_zlim(np.min(all_positions[:, 2]), np.max(all_positions[:, 2]))

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("3D Structure Evolution")

        scat = ax.scatter([], [], [], s=50, c=[], cmap="viridis", alpha=0.8)

        def animate(frame):
            if frame < len(results.positions):
                positions = results.positions[frame]
                scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
                # Color by atom index
                colors = np.arange(len(positions))
                scat.set_array(colors)
            return (scat,)

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(results.positions),
            interval=1000 // self.config.animation_fps,
            blit=False,
            repeat=True,
        )

        if save_path and self.config.save_animation:
            anim.save(save_path, writer="html")

        return anim

    def _calculate_von_mises_stress(self, stress_tensor: np.ndarray) -> float:
        """Calculate von Mises equivalent stress."""
        if stress_tensor.size == 0:
            return 0.0

        s11, s22, s33 = stress_tensor[0, 0], stress_tensor[1, 1], stress_tensor[2, 2]
        s12, s13, s23 = stress_tensor[0, 1], stress_tensor[0, 2], stress_tensor[1, 2]

        return np.sqrt(
            0.5
            * (
                (s11 - s22) ** 2
                + (s22 - s33) ** 2
                + (s33 - s11) ** 2
                + 6 * (s12**2 + s13**2 + s23**2)
            )
        )

    def _calculate_equivalent_strain(self, strain_tensor: np.ndarray) -> float:
        """Calculate equivalent strain."""
        if strain_tensor.size == 0:
            return 0.0

        e11, e22, e33 = strain_tensor[0, 0], strain_tensor[1, 1], strain_tensor[2, 2]
        e12, e13, e23 = strain_tensor[0, 1], strain_tensor[0, 2], strain_tensor[1, 2]

        return np.sqrt(
            (2 / 3)
            * (e11**2 + e22**2 + e33**2 + 2 * (e12**2 + e13**2 + e23**2))
        )


class MaterialDamageVisualizer(BaseVisualizer):
    """Specialized visualizer for material damage and failure."""

    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)

    def create_failure_mechanism_plot(
        self,
        failure_analysis: FailureAnalysis,
        dynamic_properties: Dict[str, DynamicProperty] = None,
    ) -> Union[str, Any]:
        """Create failure mechanism visualization."""

        self.logger.info("Creating failure mechanism visualization")

        if self.plotly_available:
            return self._create_plotly_failure_plot(
                failure_analysis, dynamic_properties
            )
        elif self.matplotlib_available:
            return self._create_matplotlib_failure_plot(
                failure_analysis, dynamic_properties
            )
        else:
            raise VisualizationError("No visualization backend available")

    def create_damage_map(
        self, simulation_results: SimulationResults, interactive: bool = True
    ) -> Union[str, Any]:
        """Create spatial damage distribution map."""

        self.logger.info("Creating damage map visualization")

        if simulation_results.positions.size == 0:
            raise VisualizationError("No position data for damage mapping")

        # Use final positions and damage data
        final_positions = (
            simulation_results.positions[-1]
            if len(simulation_results.positions) > 0
            else np.array([[0, 0, 0]])
        )
        damage_values = np.random.random(
            len(final_positions)
        )  # Placeholder damage values

        if interactive and self.plotly_available:
            return self._create_plotly_damage_map(final_positions, damage_values)
        elif self.matplotlib_available:
            return self._create_matplotlib_damage_map(final_positions, damage_values)
        else:
            raise VisualizationError("No visualization backend available")

    def create_crack_propagation_plot(
        self, simulation_results: SimulationResults
    ) -> Union[str, Any]:
        """Create crack propagation visualization."""

        self.logger.info("Creating crack propagation visualization")

        if simulation_results.crack_length.size == 0:
            raise VisualizationError("No crack data available")

        time_steps = np.arange(len(simulation_results.crack_length))

        if self.plotly_available:
            return self._create_plotly_crack_plot(
                time_steps, simulation_results.crack_length
            )
        elif self.matplotlib_available:
            return self._create_matplotlib_crack_plot(
                time_steps, simulation_results.crack_length
            )
        else:
            raise VisualizationError("No visualization backend available")

    def _create_plotly_failure_plot(
        self,
        failure_analysis: FailureAnalysis,
        dynamic_properties: Dict[str, DynamicProperty] = None,
    ) -> str:
        """Create Plotly failure mechanism plot."""

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Primary Mechanism",
                "Failure Timeline",
                "Critical Values",
                "Mechanism Scores",
            ),
            specs=[
                [{"type": "indicator"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # Primary mechanism indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=failure_analysis.failure_probability * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={
                    "text": f"{failure_analysis.primary_mechanism.value.replace('_', ' ').title()}<br>Failure Probability (%)"
                },
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "red"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            ),
            row=1,
            col=1,
        )

        # Failure timeline (placeholder)
        timeline_x = [
            0,
            failure_analysis.failure_onset_time,
            failure_analysis.failure_onset_time * 2,
        ]
        timeline_y = [0, 0.5, 1.0]
        fig.add_trace(
            go.Scatter(
                x=timeline_x,
                y=timeline_y,
                mode="lines+markers",
                name="Failure Evolution",
                line=dict(color="red", width=3),
            ),
            row=1,
            col=2,
        )

        # Critical values
        critical_names = ["Critical Stress", "Critical Strain", "Fracture Energy"]
        critical_values = [
            failure_analysis.critical_stress / 1e6,  # Convert to MPa
            failure_analysis.critical_strain * 100,  # Convert to %
            failure_analysis.fracture_energy / 1000,  # Convert to kJ/m²
        ]

        fig.add_trace(
            go.Bar(
                x=critical_names,
                y=critical_values,
                marker_color=["red", "orange", "yellow"],
                name="Critical Values",
            ),
            row=2,
            col=1,
        )

        # Mechanism scores (placeholder)
        mechanisms = ["Brittle", "Ductile", "Fatigue", "Thermal", "Radiation"]
        scores = [0.8, 0.3, 0.1, 0.2, 0.0]  # Placeholder scores

        fig.add_trace(
            go.Bar(
                x=mechanisms,
                y=scores,
                marker_color="blue",
                name="Mechanism Probability",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            **self._create_plotly_layout("Failure Mechanism Analysis"), showlegend=False
        )

        html_str = pyo.plot(fig, output_type="div", include_plotlyjs="cdn")
        return html_str

    def _create_matplotlib_failure_plot(
        self,
        failure_analysis: FailureAnalysis,
        dynamic_properties: Dict[str, DynamicProperty] = None,
    ) -> Any:
        """Create matplotlib failure mechanism plot."""

        self._setup_matplotlib_style()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=self.config.figure_size
        )

        # Primary mechanism pie chart
        mechanisms = [
            failure_analysis.primary_mechanism.value.replace("_", " ").title()
        ]
        secondary = [
            m.value.replace("_", " ").title()
            for m in failure_analysis.secondary_mechanisms
        ]
        all_mechanisms = mechanisms + secondary + ["Other"]

        probabilities = [failure_analysis.failure_probability]
        probabilities.extend([0.1] * len(secondary))  # Placeholder for secondary
        probabilities.append(max(0, 1 - sum(probabilities)))  # Remainder

        ax1.pie(probabilities, labels=all_mechanisms, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Failure Mechanism Distribution")

        # Failure timeline
        timeline_x = [
            0,
            failure_analysis.failure_onset_time,
            failure_analysis.failure_onset_time * 2,
        ]
        timeline_y = [0, 0.5, 1.0]
        ax2.plot(timeline_x, timeline_y, "r-o", linewidth=3, markersize=8)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Failure Progress")
        ax2.set_title("Failure Timeline")
        ax2.grid(True)

        # Critical values
        critical_names = [
            "Critical\nStress\n(MPa)",
            "Critical\nStrain\n(%)",
            "Fracture\nEnergy\n(kJ/m²)",
        ]
        critical_values = [
            failure_analysis.critical_stress / 1e6,
            failure_analysis.critical_strain * 100,
            failure_analysis.fracture_energy / 1000,
        ]

        bars = ax3.bar(
            critical_names, critical_values, color=["red", "orange", "yellow"]
        )
        ax3.set_title("Critical Values")
        ax3.tick_params(axis="x", rotation=0)

        # Add value labels on bars
        for bar, value in zip(bars, critical_values):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Mechanism scores
        mechanisms = ["Brittle", "Ductile", "Fatigue", "Thermal", "Radiation"]
        scores = [0.8, 0.3, 0.1, 0.2, 0.0]  # Placeholder

        ax4.bar(mechanisms, scores, color="blue", alpha=0.7)
        ax4.set_title("Mechanism Probability Scores")
        ax4.set_ylabel("Probability")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig

    def _create_plotly_damage_map(
        self, positions: np.ndarray, damage_values: np.ndarray
    ) -> str:
        """Create Plotly damage map."""

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker=dict(
                    size=8,
                    color=damage_values,
                    colorscale="Reds",
                    colorbar=dict(title="Damage Level"),
                    opacity=0.8,
                ),
                text=[f"Damage: {d:.3f}" for d in damage_values],
                hovertemplate="Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>%{text}<extra></extra>",
            )
        )

        fig.update_layout(
            **self._create_plotly_layout("Damage Distribution Map"),
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Z Position",
            ),
        )

        html_str = pyo.plot(fig, output_type="div", include_plotlyjs="cdn")
        return html_str

    def _create_matplotlib_damage_map(
        self, positions: np.ndarray, damage_values: np.ndarray
    ) -> Any:
        """Create matplotlib damage map."""

        self._setup_matplotlib_style()

        fig = plt.figure(figsize=self.config.figure_size)
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=damage_values,
            cmap="Reds",
            s=100,
            alpha=0.8,
        )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("Damage Distribution Map")

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label("Damage Level")

        return fig

    def _create_plotly_crack_plot(
        self, time_steps: np.ndarray, crack_lengths: np.ndarray
    ) -> str:
        """Create Plotly crack propagation plot."""

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=crack_lengths,
                mode="lines+markers",
                name="Crack Length",
                line=dict(color="red", width=3),
                marker=dict(size=6),
            )
        )

        # Add crack growth rate as secondary y-axis
        if len(crack_lengths) > 1:
            growth_rate = np.gradient(crack_lengths, time_steps)

            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=growth_rate,
                    mode="lines",
                    name="Growth Rate",
                    line=dict(color="blue", width=2, dash="dash"),
                    yaxis="y2",
                )
            )

        fig.update_layout(
            **self._create_plotly_layout("Crack Propagation"),
            xaxis_title="Time Step",
            yaxis_title="Crack Length",
            yaxis2=dict(title="Growth Rate", overlaying="y", side="right"),
        )

        html_str = pyo.plot(fig, output_type="div", include_plotlyjs="cdn")
        return html_str

    def _create_matplotlib_crack_plot(
        self, time_steps: np.ndarray, crack_lengths: np.ndarray
    ) -> Any:
        """Create matplotlib crack propagation plot."""

        self._setup_matplotlib_style()

        fig, ax1 = plt.subplots(figsize=self.config.figure_size)

        # Crack length
        line1 = ax1.plot(
            time_steps,
            crack_lengths,
            "r-o",
            linewidth=3,
            markersize=6,
            label="Crack Length",
        )
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Crack Length", color="red")
        ax1.tick_params(axis="y", labelcolor="red")
        ax1.grid(True)

        # Growth rate on secondary axis
        if len(crack_lengths) > 1:
            ax2 = ax1.twinx()
            growth_rate = np.gradient(crack_lengths, time_steps)
            line2 = ax2.plot(
                time_steps, growth_rate, "b--", linewidth=2, label="Growth Rate"
            )
            ax2.set_ylabel("Growth Rate", color="blue")
            ax2.tick_params(axis="y", labelcolor="blue")

            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="upper left")
        else:
            ax1.legend()

        plt.title("Crack Propagation Analysis")
        plt.tight_layout()

        return fig


class ExtremConditionsPlotter:
    """Main plotter class for extreme conditions visualizations."""

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.logger = get_logger(self.__class__.__name__)

        # Initialize visualizers
        self.dynamic_visualizer = DynamicResponseVisualizer(config)
        self.damage_visualizer = MaterialDamageVisualizer(config)

    def create_comprehensive_dashboard(
        self,
        simulation_results: SimulationResults,
        dynamic_properties: Dict[str, DynamicProperty] = None,
        failure_analysis: FailureAnalysis = None,
        experimental_data: ExperimentalDataset = None,
    ) -> str:
        """Create comprehensive interactive dashboard."""

        self.logger.info("Creating comprehensive extreme conditions dashboard")

        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly required for dashboard creation")

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Time Series Properties",
                "Stress-Strain Response",
                "Damage Evolution",
                "Failure Analysis",
                "3D Structure View",
                "Experimental Comparison",
            ],
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter3d"}, {"type": "scatter"}],
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )

        # Time series (row 1, col 1)
        if simulation_results.temperatures.size > 0:
            time_steps = np.arange(len(simulation_results.temperatures))
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=simulation_results.temperatures,
                    name="Temperature",
                    line=dict(color="red"),
                ),
                row=1,
                col=1,
            )

        if simulation_results.pressures.size > 0:
            time_steps = np.arange(len(simulation_results.pressures))
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=simulation_results.pressures,
                    name="Pressure",
                    line=dict(color="blue"),
                    yaxis="y2",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

        # Stress-strain (row 1, col 2)
        if (
            simulation_results.stress_tensor.size > 0
            and simulation_results.strain_tensor.size > 0
        ):
            stresses = [
                self.dynamic_visualizer._calculate_von_mises_stress(s)
                for s in simulation_results.stress_tensor
            ]
            strains = [
                self.dynamic_visualizer._calculate_equivalent_strain(s)
                for s in simulation_results.strain_tensor
            ]

            fig.add_trace(
                go.Scatter(
                    x=strains,
                    y=stresses,
                    mode="lines+markers",
                    name="Stress-Strain",
                    line=dict(color="green"),
                ),
                row=1,
                col=2,
            )

        # Damage evolution (row 2, col 1)
        if simulation_results.damage_fraction.size > 0:
            time_steps = np.arange(len(simulation_results.damage_fraction))
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=simulation_results.damage_fraction,
                    mode="lines+markers",
                    name="Damage",
                    line=dict(color="red"),
                ),
                row=2,
                col=1,
            )

        # Failure analysis (row 2, col 2)
        if failure_analysis:
            mechanisms = ["Primary", "Secondary", "Other"]
            probabilities = [
                failure_analysis.failure_probability,
                0.1 * len(failure_analysis.secondary_mechanisms),
                max(
                    0,
                    1
                    - failure_analysis.failure_probability
                    - 0.1 * len(failure_analysis.secondary_mechanisms),
                ),
            ]

            fig.add_trace(
                go.Bar(
                    x=mechanisms,
                    y=probabilities,
                    name="Failure Probability",
                    marker_color=["red", "orange", "gray"],
                ),
                row=2,
                col=2,
            )

        # 3D structure (row 3, col 1)
        if simulation_results.positions.size > 0:
            final_positions = (
                simulation_results.positions[-1]
                if len(simulation_results.positions) > 0
                else np.array([[0, 0, 0]])
            )

            fig.add_trace(
                go.Scatter3d(
                    x=final_positions[:, 0],
                    y=final_positions[:, 1],
                    z=final_positions[:, 2],
                    mode="markers",
                    marker=dict(size=4, color="blue"),
                    name="Final Structure",
                ),
                row=3,
                col=1,
            )

        # Experimental comparison (row 3, col 2)
        if experimental_data:
            exp_times = [dp.timestamp for dp in experimental_data.data_points]
            exp_values = [
                dp.measured_values.get("pressure", 0)
                for dp in experimental_data.data_points
            ]

            fig.add_trace(
                go.Scatter(
                    x=exp_times,
                    y=exp_values,
                    mode="markers",
                    name="Experimental",
                    marker=dict(color="red", symbol="diamond"),
                ),
                row=3,
                col=2,
            )

            # Add simulation data for comparison
            if simulation_results.pressures.size > 0:
                sim_times = np.arange(len(simulation_results.pressures))
                fig.add_trace(
                    go.Scatter(
                        x=sim_times,
                        y=simulation_results.pressures,
                        mode="lines",
                        name="Simulation",
                        line=dict(color="blue"),
                    ),
                    row=3,
                    col=2,
                )

        # Update layout
        fig.update_layout(
            title="Extreme Conditions Analysis Dashboard",
            height=1200,
            showlegend=True,
            hovermode="closest",
        )

        # Create HTML with controls
        html_str = pyo.plot(fig, output_type="div", include_plotlyjs="cdn")

        # Add custom controls if using Jupyter
        if IPYWIDGETS_AVAILABLE:
            controls_html = self._create_dashboard_controls()
            html_str = controls_html + html_str

        return html_str

    def _create_dashboard_controls(self) -> str:
        """Create dashboard control widgets."""

        if not IPYWIDGETS_AVAILABLE:
            return ""

        # Create control widgets
        time_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            step=1,
            description="Time Step:",
            continuous_update=False,
        )

        property_selector = widgets.SelectMultiple(
            options=["pressure", "temperature", "damage_fraction", "stress", "strain"],
            value=["pressure", "temperature"],
            description="Properties:",
            disabled=False,
        )

        view_selector = widgets.Dropdown(
            options=["2D", "3D", "Animation"],
            value="2D",
            description="View Mode:",
        )

        # Package into HTML
        controls_html = """
        <div style="border: 1px solid #ccc; padding: 10px; margin: 10px; background-color: #f9f9f9;">
            <h3>Dashboard Controls</h3>
            <p>Use the controls below to interact with the visualization:</p>
            <ul>
                <li>Time Slider: Navigate through simulation timesteps</li>
                <li>Property Selector: Choose which properties to display</li>
                <li>View Mode: Switch between 2D, 3D, and animated views</li>
            </ul>
        </div>
        """

        return controls_html

    def export_all_plots(
        self,
        simulation_results: SimulationResults,
        output_directory: str,
        dynamic_properties: Dict[str, DynamicProperty] = None,
        failure_analysis: FailureAnalysis = None,
    ) -> List[str]:
        """Export all available plots to files."""

        self.logger.info(f"Exporting all plots to {output_directory}")

        from pathlib import Path

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = []

        try:
            # Time series plot
            time_series = self.dynamic_visualizer.create_time_series_plot(
                simulation_results
            )
            if isinstance(time_series, str):
                file_path = output_path / "time_series.html"
                with open(file_path, "w") as f:
                    f.write(time_series)
                exported_files.append(str(file_path))

            # Stress-strain plot
            if simulation_results.stress_tensor.size > 0:
                stress_strain = (
                    self.dynamic_visualizer.create_stress_strain_visualization(
                        simulation_results
                    )
                )
                if isinstance(stress_strain, str):
                    file_path = output_path / "stress_strain.html"
                    with open(file_path, "w") as f:
                        f.write(stress_strain)
                    exported_files.append(str(file_path))

            # Damage evolution
            if simulation_results.damage_fraction.size > 0:
                damage_anim = self.dynamic_visualizer.create_damage_evolution_animation(
                    simulation_results, str(output_path / "damage_evolution.html")
                )
                if isinstance(damage_anim, str):
                    exported_files.append(str(output_path / "damage_evolution.html"))

            # Failure analysis
            if failure_analysis:
                failure_plot = self.damage_visualizer.create_failure_mechanism_plot(
                    failure_analysis, dynamic_properties
                )
                if isinstance(failure_plot, str):
                    file_path = output_path / "failure_analysis.html"
                    with open(file_path, "w") as f:
                        f.write(failure_plot)
                    exported_files.append(str(file_path))

            # Comprehensive dashboard
            dashboard = self.create_comprehensive_dashboard(
                simulation_results, dynamic_properties, failure_analysis
            )
            file_path = output_path / "comprehensive_dashboard.html"
            with open(file_path, "w") as f:
                f.write(dashboard)
            exported_files.append(str(file_path))

        except Exception as e:
            self.logger.error(f"Export failed: {e}")

        self.logger.info(f"Exported {len(exported_files)} files")
        return exported_files

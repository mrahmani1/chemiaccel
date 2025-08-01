"""
Interactive Crystallization Process Simulation

This module provides interactive simulations and visualizations for crystallization
processes, allowing users to visualize the evolution of supersaturation, crystal
size distribution, and polymorph purity throughout the designed crystallization process.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.animation as animation
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

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

from .core import (
    CrystallizationCore,
    CrystalProperties,
    PolymorphType,
    ProcessConditions,
    SimulationConfig,
)
from .kinetics_modeling import CrystallizationKineticsModel
from .process_optimization import CrystallizationOptimizer
from .structure_integration import StructurePropertyPredictor

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization settings.

    Attributes:
        plot_backend: Plotting backend ('matplotlib', 'plotly')
        animation_enabled: Whether to enable animations
        update_interval: Animation update interval (ms)
        figure_size: Figure size (width, height)
        color_scheme: Color scheme for plots
        interactive_widgets: Whether to include interactive widgets
    """

    plot_backend: str = "plotly"
    animation_enabled: bool = True
    update_interval: int = 100  # ms
    figure_size: Tuple[int, int] = (1200, 800)
    color_scheme: str = "viridis"
    interactive_widgets: bool = True


class CrystallizationVisualization(ABC):
    """Abstract base class for crystallization visualizations."""

    @abstractmethod
    def create_plot(self, data: Dict[str, Any]) -> Any:
        """Create visualization plot."""
        pass

    @abstractmethod
    def update_plot(self, data: Dict[str, Any]) -> None:
        """Update existing plot with new data."""
        pass


class SupersaturationTracker(CrystallizationVisualization):
    """
    Supersaturation evolution tracker and visualizer.
    """

    def __init__(self, config: VisualizationConfig):
        """
        Initialize supersaturation tracker.

        Args:
            config: Visualization configuration
        """
        self.config = config
        self.figure = None
        self.plot_data = {
            "time": [],
            "supersaturation": [],
            "temperature": [],
            "nucleation_rate": [],
            "solubility": [],
        }

    def create_plot(self, data: Dict[str, Any]) -> Any:
        """Create supersaturation tracking plot."""
        if self.config.plot_backend == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_plot(data)
        elif self.config.plot_backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_plot(data)
        else:
            raise ValueError(
                f"Plotting backend {self.config.plot_backend} not available"
            )

    def update_plot(self, data: Dict[str, Any]) -> None:
        """Update supersaturation plot with new data."""
        # Update internal data
        self._update_data(data)

        if self.config.plot_backend == "plotly" and self.figure:
            self._update_plotly_plot()
        elif self.config.plot_backend == "matplotlib" and self.figure:
            self._update_matplotlib_plot()

    def _create_plotly_plot(self, data: Dict[str, Any]) -> go.Figure:
        """Create Plotly supersaturation plot."""
        self._update_data(data)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Supersaturation Evolution",
                "Temperature Profile",
                "Nucleation Rate",
                "Solubility Curve",
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Supersaturation trace
        fig.add_trace(
            go.Scatter(
                x=self.plot_data["time"],
                y=self.plot_data["supersaturation"],
                mode="lines",
                name="Supersaturation",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Nucleation zone
        if self.plot_data["supersaturation"]:
            nucleation_threshold = 1.5
            fig.add_hline(
                y=nucleation_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Nucleation Threshold",
                row=1,
                col=1,
            )

        # Temperature profile
        fig.add_trace(
            go.Scatter(
                x=self.plot_data["time"],
                y=self.plot_data["temperature"],
                mode="lines",
                name="Temperature",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=2,
        )

        # Nucleation rate
        fig.add_trace(
            go.Scatter(
                x=self.plot_data["time"],
                y=self.plot_data["nucleation_rate"],
                mode="lines",
                name="Nucleation Rate",
                line=dict(color="green", width=2),
            ),
            row=2,
            col=1,
        )

        # Solubility curve
        if self.plot_data["temperature"] and self.plot_data["solubility"]:
            fig.add_trace(
                go.Scatter(
                    x=self.plot_data["temperature"],
                    y=self.plot_data["solubility"],
                    mode="lines+markers",
                    name="Solubility",
                    line=dict(color="purple", width=2),
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title="Supersaturation and Process Evolution", height=600, showlegend=True
        )

        # Update axes labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Supersaturation Ratio", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Temperature (K)", row=1, col=2)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Nucleation Rate (1/m³/s)", row=2, col=1)
        fig.update_xaxes(title_text="Temperature (K)", row=2, col=2)
        fig.update_yaxes(title_text="Solubility (g/L)", row=2, col=2)

        self.figure = fig
        return fig

    def _create_matplotlib_plot(self, data: Dict[str, Any]) -> plt.Figure:
        """Create Matplotlib supersaturation plot."""
        self._update_data(data)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Supersaturation evolution
        ax1.plot(
            self.plot_data["time"], self.plot_data["supersaturation"], "b-", linewidth=2
        )
        ax1.axhline(y=1.5, color="r", linestyle="--", label="Nucleation Threshold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Supersaturation Ratio")
        ax1.set_title("Supersaturation Evolution")
        ax1.legend()
        ax1.grid(True)

        # Temperature profile
        ax2.plot(
            self.plot_data["time"], self.plot_data["temperature"], "r-", linewidth=2
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Temperature (K)")
        ax2.set_title("Temperature Profile")
        ax2.grid(True)

        # Nucleation rate
        ax3.plot(
            self.plot_data["time"], self.plot_data["nucleation_rate"], "g-", linewidth=2
        )
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Nucleation Rate (1/m³/s)")
        ax3.set_title("Nucleation Rate")
        ax3.grid(True)

        # Solubility curve
        if self.plot_data["temperature"] and self.plot_data["solubility"]:
            ax4.plot(
                self.plot_data["temperature"],
                self.plot_data["solubility"],
                "purple",
                marker="o",
            )
            ax4.set_xlabel("Temperature (K)")
            ax4.set_ylabel("Solubility (g/L)")
            ax4.set_title("Solubility Curve")
            ax4.grid(True)

        plt.tight_layout()
        self.figure = fig
        return fig

    def _update_data(self, data: Dict[str, Any]) -> None:
        """Update internal plot data."""
        simulation_results = data.get("simulation_results", {})

        if "time_points" in simulation_results:
            self.plot_data["time"] = simulation_results["time_points"].tolist()

        if "results" in simulation_results:
            results = simulation_results["results"]
            if "supersaturation_evolution" in results:
                self.plot_data["supersaturation"] = results[
                    "supersaturation_evolution"
                ].tolist()

        # Extract process conditions
        conditions = data.get("conditions")
        if conditions and hasattr(conditions, "temperature_profile"):
            if conditions.temperature_profile:
                times = [t[1] for t in conditions.temperature_profile]
                temps = [t[0] for t in conditions.temperature_profile]
                self.plot_data["temperature"] = temps

        # Calculate nucleation rates (simplified)
        if self.plot_data["supersaturation"]:
            self.plot_data["nucleation_rate"] = [
                max(0, (s - 1.0) ** 3 * 1e6) for s in self.plot_data["supersaturation"]
            ]

        # Calculate solubility (temperature-dependent)
        if self.plot_data["temperature"]:
            self.plot_data["solubility"] = [
                0.1 * np.exp(-(2000.0 / 8.314) * (1 / T - 1 / 298.15))
                for T in self.plot_data["temperature"]
            ]

    def _update_plotly_plot(self) -> None:
        """Update Plotly plot with new data."""
        if not self.figure:
            return

        # Update traces with new data
        with self.figure.batch_update():
            # Update supersaturation
            self.figure.data[0].x = self.plot_data["time"]
            self.figure.data[0].y = self.plot_data["supersaturation"]

            # Update other traces if they exist
            if len(self.figure.data) > 1:
                self.figure.data[1].x = self.plot_data["time"]
                self.figure.data[1].y = self.plot_data["temperature"]

            if len(self.figure.data) > 2:
                self.figure.data[2].x = self.plot_data["time"]
                self.figure.data[2].y = self.plot_data["nucleation_rate"]

    def _update_matplotlib_plot(self) -> None:
        """Update Matplotlib plot with new data."""
        if not self.figure:
            return

        # Clear and redraw (simple approach)
        for ax in self.figure.axes:
            ax.clear()

        # Recreate plots with new data
        self._create_matplotlib_plot(
            {"simulation_results": {"time_points": np.array(self.plot_data["time"])}}
        )


class PolymorphPurityMonitor(CrystallizationVisualization):
    """
    Polymorph composition and purity monitor.
    """

    def __init__(self, config: VisualizationConfig):
        """
        Initialize polymorph purity monitor.

        Args:
            config: Visualization configuration
        """
        self.config = config
        self.figure = None
        self.polymorph_data = {}

    def create_plot(self, data: Dict[str, Any]) -> Any:
        """Create polymorph monitoring plot."""
        if self.config.plot_backend == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_plot(data)
        elif self.config.plot_backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_plot(data)
        else:
            raise ValueError(
                f"Plotting backend {self.config.plot_backend} not available"
            )

    def update_plot(self, data: Dict[str, Any]) -> None:
        """Update polymorph plot with new data."""
        self._update_polymorph_data(data)

        if self.config.plot_backend == "plotly" and self.figure:
            self._update_plotly_plot()

    def _create_plotly_plot(self, data: Dict[str, Any]) -> go.Figure:
        """Create Plotly polymorph monitoring plot."""
        self._update_polymorph_data(data)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Polymorph Distribution", "Purity Evolution"),
            specs=[[{"type": "pie"}, {"type": "scatter"}]],
        )

        # Pie chart for polymorph distribution
        if self.polymorph_data:
            polymorphs = list(self.polymorph_data.keys())
            compositions = list(self.polymorph_data.values())

            # Handle composition evolution
            if isinstance(compositions[0], list):
                # Use final composition
                final_compositions = [comp[-1] if comp else 0 for comp in compositions]
            else:
                final_compositions = compositions

            fig.add_trace(
                go.Pie(
                    labels=polymorphs,
                    values=final_compositions,
                    name="Polymorph Distribution",
                    hole=0.3,
                ),
                row=1,
                col=1,
            )

        # Purity evolution over time
        if self.polymorph_data:
            time_points = range(len(list(self.polymorph_data.values())[0]))

            for polymorph, evolution in self.polymorph_data.items():
                if isinstance(evolution, list):
                    fig.add_trace(
                        go.Scatter(
                            x=list(time_points),
                            y=evolution,
                            mode="lines",
                            name=f"{polymorph} Fraction",
                            line=dict(width=2),
                        ),
                        row=1,
                        col=2,
                    )

        fig.update_layout(
            title="Polymorph Composition and Purity Monitoring",
            height=500,
            showlegend=True,
        )

        fig.update_xaxes(title_text="Time Step", row=1, col=2)
        fig.update_yaxes(title_text="Fraction", row=1, col=2)

        self.figure = fig
        return fig

    def _create_matplotlib_plot(self, data: Dict[str, Any]) -> plt.Figure:
        """Create Matplotlib polymorph monitoring plot."""
        self._update_polymorph_data(data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart for polymorph distribution
        if self.polymorph_data:
            polymorphs = list(self.polymorph_data.keys())
            compositions = list(self.polymorph_data.values())

            # Use final composition for pie chart
            if isinstance(compositions[0], list):
                final_compositions = [comp[-1] if comp else 0 for comp in compositions]
            else:
                final_compositions = compositions

            ax1.pie(final_compositions, labels=polymorphs, autopct="%1.1f%%")
            ax1.set_title("Polymorph Distribution")

        # Evolution plot
        if self.polymorph_data:
            time_points = range(len(list(self.polymorph_data.values())[0]))

            for polymorph, evolution in self.polymorph_data.items():
                if isinstance(evolution, list):
                    ax2.plot(
                        time_points,
                        evolution,
                        marker="o",
                        label=f"{polymorph} Fraction",
                    )

        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Fraction")
        ax2.set_title("Purity Evolution")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        self.figure = fig
        return fig

    def _update_polymorph_data(self, data: Dict[str, Any]) -> None:
        """Update polymorph data from simulation results."""
        crystal_properties = data.get("crystal_properties")

        if crystal_properties and hasattr(crystal_properties, "polymorph_composition"):
            composition = crystal_properties.polymorph_composition

            # Convert PolymorphType keys to strings
            self.polymorph_data = {
                polymorph.value
                if hasattr(polymorph, "value")
                else str(polymorph): fraction
                for polymorph, fraction in composition.items()
            }

        # If no data available, create sample data
        if not self.polymorph_data:
            self.polymorph_data = {"Form_I": 0.7, "Form_II": 0.2, "Hydrate": 0.1}

    def _update_plotly_plot(self) -> None:
        """Update Plotly polymorph plot."""
        if not self.figure or not self.polymorph_data:
            return

        # Update pie chart
        polymorphs = list(self.polymorph_data.keys())
        if isinstance(list(self.polymorph_data.values())[0], list):
            compositions = [
                comp[-1] if comp else 0 for comp in self.polymorph_data.values()
            ]
        else:
            compositions = list(self.polymorph_data.values())

        with self.figure.batch_update():
            self.figure.data[0].labels = polymorphs
            self.figure.data[0].values = compositions


class ProcessVisualization:
    """
    Comprehensive process visualization combining multiple visualization components.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize process visualization.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.supersaturation_tracker = SupersaturationTracker(self.config)
        self.polymorph_monitor = PolymorphPurityMonitor(self.config)
        self.size_distribution_plot = None
        self.dashboard = None

    def create_dashboard(self, data: Dict[str, Any]) -> Any:
        """
        Create comprehensive visualization dashboard.

        Args:
            data: Complete simulation data

        Returns:
            Dashboard figure/widget
        """
        if self.config.plot_backend == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_dashboard(data)
        elif self.config.plot_backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_dashboard(data)
        else:
            raise ValueError(
                f"Dashboard creation not supported for {self.config.plot_backend}"
            )

    def update_dashboard(self, data: Dict[str, Any]) -> None:
        """Update dashboard with new data."""
        self.supersaturation_tracker.update_plot(data)
        self.polymorph_monitor.update_plot(data)

        if self.size_distribution_plot:
            self._update_size_distribution(data)

    def _create_plotly_dashboard(self, data: Dict[str, Any]) -> go.Figure:
        """Create Plotly dashboard."""
        # Create individual plots
        supersat_fig = self.supersaturation_tracker.create_plot(data)
        polymorph_fig = self.polymorph_monitor.create_plot(data)
        size_dist_fig = self._create_size_distribution_plot(data)

        # Combine into dashboard
        dashboard = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Supersaturation Evolution",
                "Temperature Profile",
                "Polymorph Distribution",
                "Purity Evolution",
                "Particle Size Distribution",
                "Process Summary",
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"type": "pie"}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "table"}],
            ],
        )

        # Add supersaturation data
        if supersat_fig and len(supersat_fig.data) > 0:
            dashboard.add_trace(supersat_fig.data[0], row=1, col=1)
            if len(supersat_fig.data) > 1:
                dashboard.add_trace(supersat_fig.data[1], row=1, col=2)

        # Add polymorph data
        if polymorph_fig and len(polymorph_fig.data) > 0:
            dashboard.add_trace(polymorph_fig.data[0], row=2, col=1)
            # Add evolution traces
            for i, trace in enumerate(polymorph_fig.data[1:], 1):
                dashboard.add_trace(trace, row=2, col=2)

        # Add size distribution
        if size_dist_fig and len(size_dist_fig.data) > 0:
            dashboard.add_trace(size_dist_fig.data[0], row=3, col=1)

        # Add process summary table
        summary_data = self._generate_process_summary(data)
        dashboard.add_trace(
            go.Table(
                header=dict(values=["Parameter", "Value"]),
                cells=dict(
                    values=[list(summary_data.keys()), list(summary_data.values())]
                ),
            ),
            row=3,
            col=2,
        )

        dashboard.update_layout(
            title="Crystallization Process Dashboard", height=1000, showlegend=True
        )

        self.dashboard = dashboard
        return dashboard

    def _create_matplotlib_dashboard(self, data: Dict[str, Any]) -> plt.Figure:
        """Create Matplotlib dashboard."""
        fig = plt.figure(figsize=(15, 12))

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Supersaturation evolution
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_supersaturation_matplotlib(ax1, data)

        # Polymorph distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_polymorph_pie_matplotlib(ax2, data)

        # Purity evolution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_purity_evolution_matplotlib(ax3, data)

        # Size distribution
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_size_distribution_matplotlib(ax4, data)

        # Process summary
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_process_summary_matplotlib(ax5, data)

        fig.suptitle("Crystallization Process Dashboard", fontsize=16)

        self.dashboard = fig
        return fig

    def _create_size_distribution_plot(self, data: Dict[str, Any]) -> go.Figure:
        """Create particle size distribution plot."""
        simulation_results = data.get("simulation_results", {})

        if "results" not in simulation_results:
            # Create empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No size distribution data available",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        results = simulation_results["results"]

        # Get final size distribution
        d10 = results.get("d10", [])
        d50 = results.get("d50", [])
        d90 = results.get("d90", [])

        if not d10 or not d50 or not d90:
            fig = go.Figure()
            fig.add_annotation(
                text="Incomplete size distribution data", x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Create histogram approximation
        final_d10, final_d50, final_d90 = d10[-1], d50[-1], d90[-1]

        # Generate size bins and distribution
        size_bins = np.logspace(
            np.log10(final_d10 * 0.5), np.log10(final_d90 * 1.5), 50
        )

        # Simple log-normal distribution approximation
        mu = np.log(final_d50)
        sigma = (np.log(final_d90) - np.log(final_d10)) / 4  # Rough estimate

        distribution = np.exp(-((np.log(size_bins) - mu) ** 2) / (2 * sigma**2)) / (
            size_bins * sigma * np.sqrt(2 * np.pi)
        )
        distribution = distribution / np.max(distribution)  # Normalize

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=size_bins * 1e6,  # Convert to micrometers
                y=distribution,
                mode="lines",
                fill="tozeroy",
                name="Size Distribution",
            )
        )

        # Add percentile markers
        fig.add_vline(x=final_d10 * 1e6, line_dash="dash", annotation_text="D10")
        fig.add_vline(x=final_d50 * 1e6, line_dash="dash", annotation_text="D50")
        fig.add_vline(x=final_d90 * 1e6, line_dash="dash", annotation_text="D90")

        fig.update_layout(
            title="Final Particle Size Distribution",
            xaxis_title="Particle Size (μm)",
            yaxis_title="Normalized Frequency",
            xaxis_type="log",
        )

        self.size_distribution_plot = fig
        return fig

    def _generate_process_summary(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate process summary data."""
        summary = {}

        # Extract key metrics
        crystal_properties = data.get("crystal_properties")
        if crystal_properties:
            summary["Final Yield (%)"] = f"{crystal_properties.yield_:.1f}"
            summary["Purity (%)"] = f"{crystal_properties.purity:.1f}"
            summary[
                "Mean Size (μm)"
            ] = f"{crystal_properties.particle_size_distribution.get('d50', 0) * 1e6:.1f}"
            summary["Surface Area (m²/g)"] = f"{crystal_properties.surface_area:.2f}"
            summary["Flowability Index"] = f"{crystal_properties.flowability_index:.2f}"

        # Process conditions
        conditions = data.get("conditions")
        if conditions:
            if (
                hasattr(conditions, "temperature_profile")
                and conditions.temperature_profile
            ):
                avg_temp = np.mean([t[0] for t in conditions.temperature_profile])
                summary["Avg Temperature (K)"] = f"{avg_temp:.1f}"

            summary[
                "Stirring Rate (rpm)"
            ] = f"{getattr(conditions, 'stirring_rate', 0):.0f}"

        # Simulation info
        simulation_results = data.get("simulation_results", {})
        if "time_points" in simulation_results:
            process_time = (
                simulation_results["time_points"][-1]
                - simulation_results["time_points"][0]
            )
            summary["Process Time (min)"] = f"{process_time / 60:.1f}"

        return summary

    def _update_size_distribution(self, data: Dict[str, Any]) -> None:
        """Update size distribution plot."""
        if self.size_distribution_plot:
            new_fig = self._create_size_distribution_plot(data)
            if new_fig and len(new_fig.data) > 0:
                self.size_distribution_plot.data = new_fig.data

    def _plot_supersaturation_matplotlib(self, ax, data: Dict[str, Any]) -> None:
        """Plot supersaturation evolution using matplotlib."""
        simulation_results = data.get("simulation_results", {})

        if "time_points" in simulation_results and "results" in simulation_results:
            time_points = simulation_results["time_points"]
            supersaturation = simulation_results["results"].get(
                "supersaturation_evolution", []
            )

            if supersaturation:
                ax.plot(
                    time_points,
                    supersaturation,
                    "b-",
                    linewidth=2,
                    label="Supersaturation",
                )
                ax.axhline(
                    y=1.5, color="r", linestyle="--", label="Nucleation Threshold"
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Supersaturation Ratio")
                ax.set_title("Supersaturation Evolution")
                ax.legend()
                ax.grid(True)

    def _plot_polymorph_pie_matplotlib(self, ax, data: Dict[str, Any]) -> None:
        """Plot polymorph distribution pie chart using matplotlib."""
        crystal_properties = data.get("crystal_properties")

        if crystal_properties and hasattr(crystal_properties, "polymorph_composition"):
            composition = crystal_properties.polymorph_composition

            if composition:
                polymorphs = [
                    p.value if hasattr(p, "value") else str(p)
                    for p in composition.keys()
                ]
                fractions = list(composition.values())

                ax.pie(fractions, labels=polymorphs, autopct="%1.1f%%")
                ax.set_title("Polymorph Distribution")

    def _plot_purity_evolution_matplotlib(self, ax, data: Dict[str, Any]) -> None:
        """Plot purity evolution using matplotlib."""
        # Placeholder for purity evolution
        ax.text(
            0.5,
            0.5,
            "Purity Evolution\n(Real-time data)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Purity Evolution")

    def _plot_size_distribution_matplotlib(self, ax, data: Dict[str, Any]) -> None:
        """Plot size distribution using matplotlib."""
        simulation_results = data.get("simulation_results", {})

        if "results" in simulation_results:
            results = simulation_results["results"]
            d10 = results.get("d10", [])
            d50 = results.get("d50", [])
            d90 = results.get("d90", [])

            if d10 and d50 and d90:
                final_d10, final_d50, final_d90 = d10[-1], d50[-1], d90[-1]

                # Simple bar plot of percentiles
                sizes = ["D10", "D50", "D90"]
                values = [final_d10 * 1e6, final_d50 * 1e6, final_d90 * 1e6]

                ax.bar(sizes, values)
                ax.set_ylabel("Particle Size (μm)")
                ax.set_title("Size Distribution Percentiles")

    def _plot_process_summary_matplotlib(self, ax, data: Dict[str, Any]) -> None:
        """Plot process summary using matplotlib."""
        summary = self._generate_process_summary(data)

        # Create table
        ax.axis("off")
        table_data = list(summary.items())

        table = ax.table(
            cellText=table_data,
            colLabels=["Parameter", "Value"],
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        ax.set_title("Process Summary", pad=20)


class CrystallizationSimulator(CrystallizationCore):
    """
    Main interactive crystallization simulator integrating all components.
    """

    def __init__(
        self,
        kinetics_model: CrystallizationKineticsModel,
        optimizer: Optional[CrystallizationOptimizer] = None,
        structure_predictor: Optional[StructurePropertyPredictor] = None,
        config: Optional[SimulationConfig] = None,
        viz_config: Optional[VisualizationConfig] = None,
    ):
        """
        Initialize crystallization simulator.

        Args:
            kinetics_model: Crystallization kinetics model
            optimizer: Process optimizer (optional)
            structure_predictor: Structure-property predictor (optional)
            config: Simulation configuration
            viz_config: Visualization configuration
        """
        super().__init__(config)
        self.kinetics_model = kinetics_model
        self.optimizer = optimizer
        self.structure_predictor = structure_predictor
        self.viz_config = viz_config or VisualizationConfig()

        self.process_visualization = ProcessVisualization(self.viz_config)
        self.simulation_results = {}

    def initialize(self) -> None:
        """Initialize the simulator."""
        if not self.kinetics_model.is_initialized():
            self.kinetics_model.initialize()

        if self.structure_predictor and not self.structure_predictor.is_initialized():
            self.structure_predictor.initialize()

        self._initialized = True
        logger.info("Crystallization simulator initialized")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        required_keys = ["conditions", "molecular_data"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required input: {key}")
        return True

    def run_interactive_simulation(
        self,
        conditions: ProcessConditions,
        molecular_data: Dict[str, Any],
        create_dashboard: bool = True,
    ) -> Dict[str, Any]:
        """
        Run interactive crystallization simulation.

        Args:
            conditions: Process conditions
            molecular_data: Molecular data
            create_dashboard: Whether to create visualization dashboard

        Returns:
            Complete simulation results with visualizations
        """
        if not self.is_initialized():
            self.initialize()

        self.validate_inputs(conditions=conditions, molecular_data=molecular_data)

        # Run kinetics simulation
        logger.info("Running kinetics simulation...")
        kinetics_results = self.kinetics_model.simulate_crystallization(conditions)

        # Predict crystal properties
        crystal_properties = None
        if self.structure_predictor:
            logger.info("Predicting crystal properties...")
            crystal_properties = (
                self.structure_predictor.predict_crystallization_properties(
                    molecular_data, conditions, kinetics_results
                )
            )

        # Compile results
        complete_results = {
            "simulation_results": kinetics_results,
            "crystal_properties": crystal_properties,
            "conditions": conditions,
            "molecular_data": molecular_data,
            "timestamp": np.datetime64("now").astype(str),
        }

        self.simulation_results = complete_results

        # Create visualizations
        dashboard = None
        if create_dashboard:
            logger.info("Creating visualization dashboard...")
            dashboard = self.process_visualization.create_dashboard(complete_results)

        # Store in simulation data
        self._simulation_data = {
            "complete_results": complete_results,
            "dashboard": dashboard,
        }

        return {
            "results": complete_results,
            "dashboard": dashboard,
            "visualization_config": self.viz_config,
        }

    def update_conditions(self, new_conditions: ProcessConditions) -> None:
        """Update process conditions and re-run simulation."""
        if not self.simulation_results:
            logger.warning("No previous simulation to update")
            return

        molecular_data = self.simulation_results.get("molecular_data", {})

        # Re-run simulation with new conditions
        updated_results = self.run_interactive_simulation(
            new_conditions, molecular_data, create_dashboard=True
        )

        logger.info("Simulation updated with new conditions")

    def optimize_process(
        self, objectives: List[str], n_iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Run process optimization with visualization.

        Args:
            objectives: Optimization objectives
            n_iterations: Number of optimization iterations

        Returns:
            Optimization results with visualizations
        """
        if not self.optimizer:
            raise ValueError("No optimizer configured")

        if not self.simulation_results:
            raise ValueError("Run initial simulation first")

        logger.info(f"Running process optimization with {n_iterations} iterations...")

        # Run optimization
        optimization_results = self.optimizer.optimize_process(n_iterations)

        # Visualize optimization progress
        optimization_viz = self._create_optimization_visualization(optimization_results)

        return {
            "optimization_results": optimization_results,
            "optimization_visualization": optimization_viz,
        }

    def export_results(self, filepath: str, format_type: str = "json") -> None:
        """
        Export simulation results to file.

        Args:
            filepath: Output file path
            format_type: Export format ('json', 'csv', 'html')
        """
        if not self.simulation_results:
            raise ValueError("No simulation results to export")

        if format_type == "json":
            import json

            # Convert numpy arrays to lists for JSON serialization
            exportable_results = self._prepare_for_json_export(self.simulation_results)

            with open(filepath, "w") as f:
                json.dump(exportable_results, f, indent=2)

        elif format_type == "html" and PLOTLY_AVAILABLE:
            dashboard = self._simulation_data.get("dashboard")
            if dashboard:
                dashboard.write_html(filepath)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        logger.info(f"Results exported to {filepath}")

    def _create_optimization_visualization(
        self, optimization_results: Dict[str, Any]
    ) -> Any:
        """Create optimization progress visualization."""
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        # Plot optimization history
        history = optimization_results.get("optimization_history", [])
        if history:
            iterations = list(range(1, len(history) + 1))
            objectives = [result.objectives for result in history]

            # Plot first objective
            if objectives and len(objectives[0]) > 0:
                first_obj_key = list(objectives[0].keys())[0]
                obj_values = [obj[first_obj_key] for obj in objectives]

                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=obj_values,
                        mode="lines+markers",
                        name=f"{first_obj_key} Evolution",
                    )
                )

        fig.update_layout(
            title="Optimization Progress",
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
        )

        return fig

    def _prepare_for_json_export(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for JSON export by converting numpy arrays."""
        import json

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        return convert_numpy(data)

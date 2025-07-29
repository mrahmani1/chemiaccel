"""
Real-time Interactive Visualization System

This module provides comprehensive visualization capabilities for real-time process simulation,
including property evolution, microstructure changes, and process parameter monitoring.
"""

import asyncio
import json
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from .realtime_process_simulation import (
    MaterialProperty,
    MicrostructureState,
    PropertyType,
    RealTimeSimulationCore,
    SimulationConfig,
    SimulationState,
)


class VisualizationMode(Enum):
    """Visualization rendering modes."""

    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    WEB_BASED = "web_based"
    HEADLESS = "headless"


class PlotType(Enum):
    """Types of visualization plots."""

    TIME_SERIES = "time_series"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    SURFACE = "surface"
    HISTOGRAM = "histogram"
    PHASE_DIAGRAM = "phase_diagram"
    MICROSTRUCTURE_MAP = "microstructure_map"


@dataclass
class VisualizationConfig:
    """Configuration for visualization system."""

    mode: VisualizationMode = VisualizationMode.PLOTLY
    update_interval: float = 0.1  # seconds
    max_history_points: int = 1000
    enable_real_time_updates: bool = True
    save_frames: bool = False
    output_directory: str = "visualization_output"
    enable_interaction: bool = True
    figure_size: Tuple[int, int] = (1200, 800)
    color_scheme: str = "viridis"
    enable_animations: bool = True
    export_format: str = "html"


@dataclass
class PlotConfiguration:
    """Configuration for individual plots."""

    title: str
    plot_type: PlotType
    x_label: str
    y_label: str
    properties: List[str]
    update_frequency: float = 1.0  # Hz
    auto_scale: bool = True
    color_map: Optional[str] = None
    show_uncertainty: bool = True
    enable_zoom: bool = True


class VisualizationEngine(ABC):
    """Abstract base class for visualization engines."""

    @abstractmethod
    def create_figure(self, config: PlotConfiguration) -> Any:
        """Create a new figure."""
        pass

    @abstractmethod
    def update_plot(self, figure: Any, data: Dict[str, Any]) -> None:
        """Update plot with new data."""
        pass

    @abstractmethod
    def save_figure(self, figure: Any, filepath: str) -> None:
        """Save figure to file."""
        pass


class PlotlyVisualizationEngine(VisualizationEngine):
    """Plotly-based visualization engine."""

    def __init__(self, config: VisualizationConfig):
        if not PLOTLY_AVAILABLE:
            raise ChemiAccelError("Plotly is required but not available")

        self.config = config
        self.figures: Dict[str, go.Figure] = {}

    def create_figure(self, config: PlotConfiguration) -> go.Figure:
        """Create a new Plotly figure."""

        if config.plot_type == PlotType.TIME_SERIES:
            fig = go.Figure()
            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_label,
                yaxis_title=config.y_label,
                width=self.config.figure_size[0],
                height=self.config.figure_size[1],
                template="plotly_white",
            )

            # Add traces for each property
            for prop in config.properties:
                fig.add_trace(
                    go.Scatter(
                        x=[], y=[], mode="lines+markers", name=prop, line=dict(width=2)
                    )
                )

        elif config.plot_type == PlotType.HEATMAP:
            fig = go.Figure(
                data=go.Heatmap(
                    z=np.random.rand(10, 10),  # Placeholder data
                    colorscale=config.color_map or self.config.color_scheme,
                )
            )
            fig.update_layout(
                title=config.title,
                width=self.config.figure_size[0],
                height=self.config.figure_size[1],
            )

        elif config.plot_type == PlotType.SURFACE:
            fig = go.Figure(
                data=[
                    go.Surface(
                        z=np.random.rand(20, 20),  # Placeholder data
                        colorscale=config.color_map or self.config.color_scheme,
                    )
                ]
            )
            fig.update_layout(
                title=config.title,
                scene=dict(
                    xaxis_title=config.x_label,
                    yaxis_title=config.y_label,
                    zaxis_title="Value",
                ),
                width=self.config.figure_size[0],
                height=self.config.figure_size[1],
            )

        else:
            # Default to scatter plot
            fig = go.Figure()
            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_label,
                yaxis_title=config.y_label,
                width=self.config.figure_size[0],
                height=self.config.figure_size[1],
            )

        return fig

    def update_plot(self, figure: go.Figure, data: Dict[str, Any]) -> None:
        """Update Plotly figure with new data."""

        if "time_series" in data:
            # Update time series data
            for i, (prop_name, values) in enumerate(data["time_series"].items()):
                if i < len(figure.data):
                    times = data.get("times", list(range(len(values))))
                    figure.data[i].x = times
                    figure.data[i].y = values

        elif "heatmap" in data:
            # Update heatmap data
            if len(figure.data) > 0 and hasattr(figure.data[0], "z"):
                figure.data[0].z = data["heatmap"]

        elif "surface" in data:
            # Update surface data
            if len(figure.data) > 0 and hasattr(figure.data[0], "z"):
                figure.data[0].z = data["surface"]

    def save_figure(self, figure: go.Figure, filepath: str) -> None:
        """Save Plotly figure to file."""
        if filepath.endswith(".html"):
            figure.write_html(filepath)
        elif filepath.endswith(".png"):
            figure.write_image(filepath)
        elif filepath.endswith(".pdf"):
            figure.write_image(filepath)
        else:
            figure.write_html(f"{filepath}.html")


class MatplotlibVisualizationEngine(VisualizationEngine):
    """Matplotlib-based visualization engine."""

    def __init__(self, config: VisualizationConfig):
        if not MATPLOTLIB_AVAILABLE:
            raise ChemiAccelError("Matplotlib is required but not available")

        self.config = config
        self.figures: Dict[str, plt.Figure] = {}

        # Setup matplotlib for interactive mode
        plt.ion()

    def create_figure(self, config: PlotConfiguration) -> plt.Figure:
        """Create a new matplotlib figure."""

        fig, ax = plt.subplots(
            figsize=(self.config.figure_size[0] / 100, self.config.figure_size[1] / 100)
        )

        ax.set_title(config.title)
        ax.set_xlabel(config.x_label)
        ax.set_ylabel(config.y_label)
        ax.grid(True, alpha=0.3)

        if config.plot_type == PlotType.TIME_SERIES:
            # Initialize lines for each property
            lines = []
            for prop in config.properties:
                (line,) = ax.plot([], [], label=prop, linewidth=2)
                lines.append(line)
            ax.legend()

            # Store lines in figure for later access
            fig.lines = lines

        elif config.plot_type == PlotType.HEATMAP:
            # Create initial heatmap
            im = ax.imshow(
                np.random.rand(10, 10),
                cmap=config.color_map or self.config.color_scheme,
            )
            plt.colorbar(im, ax=ax)
            fig.im = im

        plt.tight_layout()
        return fig

    def update_plot(self, figure: plt.Figure, data: Dict[str, Any]) -> None:
        """Update matplotlib figure with new data."""

        if "time_series" in data and hasattr(figure, "lines"):
            # Update time series data
            ax = figure.axes[0]
            times = data.get("times", [])

            for i, (prop_name, values) in enumerate(data["time_series"].items()):
                if i < len(figure.lines):
                    figure.lines[i].set_data(times, values)

            # Rescale axes
            ax.relim()
            ax.autoscale_view()

        elif "heatmap" in data and hasattr(figure, "im"):
            # Update heatmap data
            figure.im.set_array(data["heatmap"])
            figure.im.autoscale()

        # Redraw the figure
        figure.canvas.draw()
        figure.canvas.flush_events()

    def save_figure(self, figure: plt.Figure, filepath: str) -> None:
        """Save matplotlib figure to file."""
        figure.savefig(filepath, dpi=300, bbox_inches="tight")


class RealTimeVisualizer:
    """Main visualization controller for real-time simulation."""

    def __init__(
        self, simulation_core: RealTimeSimulationCore, config: VisualizationConfig
    ):
        self.simulation_core = simulation_core
        self.config = config

        # Create visualization engine
        if config.mode == VisualizationMode.PLOTLY:
            self.engine = PlotlyVisualizationEngine(config)
        elif config.mode == VisualizationMode.MATPLOTLIB:
            self.engine = MatplotlibVisualizationEngine(config)
        else:
            raise ChemiAccelError(f"Unsupported visualization mode: {config.mode}")

        # Data storage for visualization
        self.property_histories: Dict[str, deque] = {}
        self.time_history: deque = deque(maxlen=config.max_history_points)
        self.microstructure_history: deque = deque(maxlen=config.max_history_points)

        # Plot management
        self.active_plots: Dict[str, Any] = {}
        self.plot_configs: Dict[str, PlotConfiguration] = {}

        # Register with simulation core
        simulation_core.add_update_callback(self._on_simulation_update)

        # Animation control
        self.is_animating = False
        self.animation_thread: Optional[threading.Thread] = None

    def _on_simulation_update(self, state: SimulationState):
        """Handle simulation state updates."""

        # Update time history
        self.time_history.append(state.time)

        # Update property histories
        for prop_name, prop in state.material_properties.items():
            if prop_name not in self.property_histories:
                self.property_histories[prop_name] = deque(
                    maxlen=self.config.max_history_points
                )

            if isinstance(prop.value, (int, float)):
                self.property_histories[prop_name].append(prop.value)
            else:
                # For array values, store the mean
                self.property_histories[prop_name].append(np.mean(prop.value))

        # Update microstructure history
        self.microstructure_history.append(state.microstructure)

        # Update active plots if real-time updates are enabled
        if self.config.enable_real_time_updates:
            self._update_all_plots()

    def create_property_evolution_plot(
        self, properties: List[str], plot_name: str = "property_evolution"
    ) -> Any:
        """Create a plot for property evolution over time."""

        config = PlotConfiguration(
            title="Material Property Evolution",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time (s)",
            y_label="Property Value",
            properties=properties,
            show_uncertainty=True,
        )

        figure = self.engine.create_figure(config)
        self.active_plots[plot_name] = figure
        self.plot_configs[plot_name] = config

        return figure

    def create_microstructure_evolution_plot(
        self, parameters: List[str], plot_name: str = "microstructure_evolution"
    ) -> Any:
        """Create a plot for microstructure parameter evolution."""

        config = PlotConfiguration(
            title="Microstructure Evolution",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time (s)",
            y_label="Parameter Value",
            properties=parameters,
        )

        figure = self.engine.create_figure(config)
        self.active_plots[plot_name] = figure
        self.plot_configs[plot_name] = config

        return figure

    def create_process_conditions_plot(
        self, parameters: List[str], plot_name: str = "process_conditions"
    ) -> Any:
        """Create a plot for process condition evolution."""

        config = PlotConfiguration(
            title="Process Conditions",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time (s)",
            y_label="Condition Value",
            properties=parameters,
        )

        figure = self.engine.create_figure(config)
        self.active_plots[plot_name] = figure
        self.plot_configs[plot_name] = config

        return figure

    def create_property_correlation_plot(
        self, property_x: str, property_y: str, plot_name: str = "property_correlation"
    ) -> Any:
        """Create a scatter plot showing property correlation."""

        config = PlotConfiguration(
            title=f"{property_x} vs {property_y}",
            plot_type=PlotType.SCATTER,
            x_label=property_x,
            y_label=property_y,
            properties=[property_x, property_y],
        )

        figure = self.engine.create_figure(config)
        self.active_plots[plot_name] = figure
        self.plot_configs[plot_name] = config

        return figure

    def create_spatial_distribution_plot(
        self, property_name: str, plot_name: str = "spatial_distribution"
    ) -> Any:
        """Create a heatmap for spatial property distribution."""

        config = PlotConfiguration(
            title=f"Spatial Distribution: {property_name}",
            plot_type=PlotType.HEATMAP,
            x_label="X Position",
            y_label="Y Position",
            properties=[property_name],
        )

        figure = self.engine.create_figure(config)
        self.active_plots[plot_name] = figure
        self.plot_configs[plot_name] = config

        return figure

    def _update_all_plots(self):
        """Update all active plots with current data."""

        for plot_name, figure in self.active_plots.items():
            config = self.plot_configs[plot_name]

            if config.plot_type == PlotType.TIME_SERIES:
                # Prepare time series data
                data = {"time_series": {}, "times": list(self.time_history)}

                for prop in config.properties:
                    if prop in self.property_histories:
                        data["time_series"][prop] = list(self.property_histories[prop])
                    elif "microstructure" in plot_name:
                        # Handle microstructure parameters
                        values = self._extract_microstructure_parameter(prop)
                        data["time_series"][prop] = values
                    elif "process" in plot_name:
                        # Handle process conditions
                        values = self._extract_process_parameter(prop)
                        data["time_series"][prop] = values

                self.engine.update_plot(figure, data)

            elif config.plot_type == PlotType.SCATTER:
                # Handle scatter plots
                if len(config.properties) >= 2:
                    prop_x, prop_y = config.properties[0], config.properties[1]

                    x_data = list(self.property_histories.get(prop_x, []))
                    y_data = list(self.property_histories.get(prop_y, []))

                    min_len = min(len(x_data), len(y_data))
                    if min_len > 0:
                        data = {
                            "scatter": {"x": x_data[:min_len], "y": y_data[:min_len]}
                        }
                        self.engine.update_plot(figure, data)

            elif config.plot_type == PlotType.HEATMAP:
                # Handle spatial distribution (simplified)
                if len(self.property_histories) > 0:
                    # Create synthetic spatial data for demonstration
                    current_values = [
                        hist[-1] if hist else 0
                        for hist in self.property_histories.values()
                    ]
                    spatial_data = (
                        np.random.rand(10, 10) * np.mean(current_values)
                        if current_values
                        else np.zeros((10, 10))
                    )

                    data = {"heatmap": spatial_data}
                    self.engine.update_plot(figure, data)

    def _extract_microstructure_parameter(self, parameter: str) -> List[float]:
        """Extract microstructure parameter values from history."""
        values = []

        for microstructure in self.microstructure_history:
            if hasattr(microstructure, parameter):
                value = getattr(microstructure, parameter)
                if isinstance(value, (int, float)):
                    values.append(value)
                elif isinstance(value, dict):
                    values.append(sum(value.values()))
                elif hasattr(value, "__iter__"):
                    values.append(np.mean(value))
                else:
                    values.append(0.0)
            else:
                values.append(0.0)

        return values

    def _extract_process_parameter(self, parameter: str) -> List[float]:
        """Extract process condition parameter from simulation history."""
        values = []
        history = self.simulation_core.get_simulation_history()

        for state in history:
            if hasattr(state.process_conditions, parameter):
                value = getattr(state.process_conditions, parameter)
                values.append(float(value))
            else:
                values.append(0.0)

        return values

    def start_animation(self, interval: float = 0.1):
        """Start animated visualization updates."""
        if self.is_animating:
            return

        self.is_animating = True

        def animation_loop():
            while self.is_animating:
                self._update_all_plots()
                time.sleep(interval)

        self.animation_thread = threading.Thread(target=animation_loop)
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def stop_animation(self):
        """Stop animated visualization updates."""
        self.is_animating = False
        if self.animation_thread:
            self.animation_thread.join(timeout=1.0)

    def save_plot(self, plot_name: str, filepath: str):
        """Save a specific plot to file."""
        if plot_name in self.active_plots:
            self.engine.save_figure(self.active_plots[plot_name], filepath)
        else:
            raise ValueError(f"Plot '{plot_name}' not found")

    def save_all_plots(self, directory: str):
        """Save all active plots to directory."""
        import os

        os.makedirs(directory, exist_ok=True)

        for plot_name, figure in self.active_plots.items():
            filepath = os.path.join(
                directory, f"{plot_name}.{self.config.export_format}"
            )
            self.engine.save_figure(figure, filepath)

    def export_data(self, filepath: str):
        """Export visualization data to file."""
        data = {
            "config": self.config.__dict__,
            "time_history": list(self.time_history),
            "property_histories": {
                name: list(hist) for name, hist in self.property_histories.items()
            },
            "plot_configs": {
                name: {
                    "title": config.title,
                    "plot_type": config.plot_type.value,
                    "properties": config.properties,
                }
                for name, config in self.plot_configs.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def create_dashboard(self) -> Any:
        """Create a comprehensive dashboard with multiple plots."""

        if self.config.mode == VisualizationMode.PLOTLY:
            # Create subplot layout
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Property Evolution",
                    "Microstructure Evolution",
                    "Process Conditions",
                    "Property Correlation",
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
            )

            # Configure layout
            fig.update_layout(
                title="Real-time Process Simulation Dashboard",
                height=800,
                width=1200,
                showlegend=True,
            )

            return fig

        else:
            # For matplotlib, create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Real-time Process Simulation Dashboard")

            return fig

    def get_plot_data(self, plot_name: str) -> Dict[str, Any]:
        """Get current data for a specific plot."""
        if plot_name not in self.plot_configs:
            raise ValueError(f"Plot '{plot_name}' not found")

        config = self.plot_configs[plot_name]

        data = {
            "config": config.__dict__,
            "times": list(self.time_history),
            "properties": {},
        }

        for prop in config.properties:
            if prop in self.property_histories:
                data["properties"][prop] = list(self.property_histories[prop])

        return data


class VisualizationServer:
    """Web server for browser-based visualization."""

    def __init__(self, visualizer: RealTimeVisualizer, port: int = 8080):
        self.visualizer = visualizer
        self.port = port
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False

    def start_server(self):
        """Start the web server."""
        try:
            import flask
            from flask import Flask, jsonify, render_template_string

            app = Flask(__name__)

            @app.route("/")
            def dashboard():
                return render_template_string(
                    """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>ChemiAccel Real-time Simulation</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                </head>
                <body>
                    <h1>Real-time Process Simulation Dashboard</h1>
                    <div id="plots"></div>
                    <script>
                        // JavaScript for real-time plot updates
                        setInterval(function() {
                            fetch('/api/data')
                                .then(response => response.json())
                                .then(data => updatePlots(data));
                        }, 1000);

                        function updatePlots(data) {
                            // Update plots with new data
                            console.log('Updating plots with:', data);
                        }
                    </script>
                </body>
                </html>
                """
                )

            @app.route("/api/data")
            def get_data():
                return jsonify(
                    {
                        "property_histories": {
                            name: list(hist)
                            for name, hist in self.visualizer.property_histories.items()
                        },
                        "time_history": list(self.visualizer.time_history),
                    }
                )

            def run_server():
                app.run(host="0.0.0.0", port=self.port, debug=False)

            self.server_thread = threading.Thread(target=run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.is_running = True

        except ImportError:
            raise ChemiAccelError("Flask is required for web server functionality")

    def stop_server(self):
        """Stop the web server."""
        self.is_running = False
        # Note: Stopping Flask server programmatically is complex
        # In practice, would use more sophisticated server like gunicorn

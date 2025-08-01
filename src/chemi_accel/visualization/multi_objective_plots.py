"""
Multi-Objective Optimization Visualization Tools

Interactive plotting tools for visualizing Pareto fronts, trade-offs, and optimization landscapes.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    objectives: np.ndarray
    parameters: np.ndarray
    pareto_front: np.ndarray
    hypervolume: float
    convergence_history: List[float]
    generation_data: List[Dict[str, Any]]


class MultiObjectiveVisualizer:
    """
    Interactive visualization tools for multi-objective optimization results.

    Features:
    - 2D/3D Pareto front visualization
    - Parameter space exploration
    - Convergence tracking
    - Trade-off analysis
    - Interactive filtering and selection
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the visualizer.

        Args:
            theme: Plotly theme for styling
        """
        self.theme = theme
        self.colors = px.colors.qualitative.Set1

    def plot_pareto_front_2d(
        self,
        objectives: np.ndarray,
        pareto_mask: np.ndarray,
        objective_names: List[str],
        pareto_front: Optional[np.ndarray] = None,
        title: str = "Pareto Front Visualization",
        hover_data: Optional[Dict[str, List]] = None,
    ) -> go.Figure:
        """
        Create 2D Pareto front visualization.

        Args:
            objectives: Array of shape (n_samples, 2) with objective values
            pareto_mask: Boolean mask indicating Pareto optimal points
            objective_names: Names of the two objectives
            pareto_front: Optional theoretical Pareto front for comparison
            title: Plot title
            hover_data: Additional data to show on hover

        Returns:
            Plotly figure object
        """
        if objectives.shape[1] != 2:
            raise ValueError("This method requires exactly 2 objectives")

        fig = go.Figure()

        # Non-Pareto points
        non_pareto = objectives[~pareto_mask]
        if len(non_pareto) > 0:
            hover_template = (
                f"{objective_names[0]}: %{{x}}<br>{objective_names[1]}: %{{y}}"
            )
            if hover_data:
                for key, values in hover_data.items():
                    hover_template += f"<br>{key}: %{{customdata[{len(hover_data)}]}}"

            fig.add_trace(
                go.Scatter(
                    x=non_pareto[:, 0],
                    y=non_pareto[:, 1],
                    mode="markers",
                    name="Non-Pareto Solutions",
                    marker=dict(
                        size=8,
                        color="lightblue",
                        opacity=0.6,
                        line=dict(width=1, color="darkblue"),
                    ),
                    customdata=np.column_stack(
                        [hover_data[key] for key in hover_data.keys()]
                        if hover_data
                        else []
                    ),
                    hovertemplate=hover_template + "<extra></extra>",
                )
            )

        # Pareto points
        pareto_points = objectives[pareto_mask]
        if len(pareto_points) > 0:
            hover_template = (
                f"{objective_names[0]}: %{{x}}<br>{objective_names[1]}: %{{y}}"
            )
            if hover_data:
                pareto_hover_data = {
                    key: np.array(values)[pareto_mask]
                    for key, values in hover_data.items()
                }
                for key in hover_data.keys():
                    hover_template += f"<br>{key}: %{{customdata[{list(hover_data.keys()).index(key)}]}}"
                customdata = np.column_stack(
                    [pareto_hover_data[key] for key in hover_data.keys()]
                )
            else:
                customdata = None

            fig.add_trace(
                go.Scatter(
                    x=pareto_points[:, 0],
                    y=pareto_points[:, 1],
                    mode="markers",
                    name="Pareto Front",
                    marker=dict(
                        size=12,
                        color="red",
                        symbol="diamond",
                        line=dict(width=2, color="darkred"),
                    ),
                    customdata=customdata,
                    hovertemplate=hover_template + "<extra></extra>",
                )
            )

        # Theoretical Pareto front
        if pareto_front is not None:
            fig.add_trace(
                go.Scatter(
                    x=pareto_front[:, 0],
                    y=pareto_front[:, 1],
                    mode="lines",
                    name="Theoretical Pareto Front",
                    line=dict(color="green", width=3, dash="dash"),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=objective_names[0],
            yaxis_title=objective_names[1],
            template=self.theme,
            hovermode="closest",
            showlegend=True,
            width=800,
            height=600,
        )

        return fig

    def plot_pareto_front_3d(
        self,
        objectives: np.ndarray,
        pareto_mask: np.ndarray,
        objective_names: List[str],
        title: str = "3D Pareto Front",
        hover_data: Optional[Dict[str, List]] = None,
    ) -> go.Figure:
        """
        Create 3D Pareto front visualization.

        Args:
            objectives: Array of shape (n_samples, 3) with objective values
            pareto_mask: Boolean mask indicating Pareto optimal points
            objective_names: Names of the three objectives
            title: Plot title
            hover_data: Additional data to show on hover

        Returns:
            Plotly figure object
        """
        if objectives.shape[1] != 3:
            raise ValueError("This method requires exactly 3 objectives")

        fig = go.Figure()

        # Non-Pareto points
        non_pareto = objectives[~pareto_mask]
        if len(non_pareto) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=non_pareto[:, 0],
                    y=non_pareto[:, 1],
                    z=non_pareto[:, 2],
                    mode="markers",
                    name="Non-Pareto Solutions",
                    marker=dict(size=5, color="lightblue", opacity=0.6),
                )
            )

        # Pareto points
        pareto_points = objectives[pareto_mask]
        if len(pareto_points) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=pareto_points[:, 0],
                    y=pareto_points[:, 1],
                    z=pareto_points[:, 2],
                    mode="markers",
                    name="Pareto Front",
                    marker=dict(size=8, color="red", symbol="diamond"),
                )
            )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=objective_names[0],
                yaxis_title=objective_names[1],
                zaxis_title=objective_names[2],
            ),
            template=self.theme,
            width=900,
            height=700,
        )

        return fig

    def plot_parallel_coordinates(
        self,
        objectives: np.ndarray,
        parameters: np.ndarray,
        pareto_mask: np.ndarray,
        objective_names: List[str],
        parameter_names: List[str],
        title: str = "Multi-Objective Parameter Space",
    ) -> go.Figure:
        """
        Create parallel coordinates plot showing parameter-objective relationships.

        Args:
            objectives: Array of objective values
            parameters: Array of parameter values
            pareto_mask: Boolean mask for Pareto optimal points
            objective_names: Names of objectives
            parameter_names: Names of parameters
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Combine objectives and parameters
        all_data = np.column_stack([parameters, objectives])
        all_names = parameter_names + objective_names

        # Create color mapping based on Pareto optimality
        colors = ["red" if is_pareto else "blue" for is_pareto in pareto_mask]

        # Create dimensions for parallel coordinates
        dimensions = []
        for i, name in enumerate(all_names):
            dimensions.append(
                dict(
                    range=[all_data[:, i].min(), all_data[:, i].max()],
                    label=name,
                    values=all_data[:, i],
                )
            )

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=pareto_mask.astype(int),
                    colorscale=[[0, "lightblue"], [1, "red"]],
                    showscale=True,
                    cmin=0,
                    cmax=1,
                ),
                dimensions=dimensions,
            )
        )

        fig.update_layout(title=title, template=self.theme, width=1000, height=600)

        return fig

    def plot_convergence_history(
        self,
        convergence_data: Dict[str, List[float]],
        title: str = "Optimization Convergence",
    ) -> go.Figure:
        """
        Plot convergence history for multiple metrics.

        Args:
            convergence_data: Dictionary mapping metric names to value histories
            title: Plot title

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        for metric_name, values in convergence_data.items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode="lines+markers",
                    name=metric_name,
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Generation",
            yaxis_title="Metric Value",
            template=self.theme,
            hovermode="x unified",
            width=800,
            height=500,
        )

        return fig

    def plot_hypervolume_evolution(
        self,
        hypervolume_history: List[float],
        reference_point: Optional[np.ndarray] = None,
        title: str = "Hypervolume Evolution",
    ) -> go.Figure:
        """
        Plot hypervolume indicator evolution over generations.

        Args:
            hypervolume_history: List of hypervolume values per generation
            reference_point: Reference point used for hypervolume calculation
            title: Plot title

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        generations = list(range(len(hypervolume_history)))

        fig.add_trace(
            go.Scatter(
                x=generations,
                y=hypervolume_history,
                mode="lines+markers",
                name="Hypervolume",
                line=dict(color="blue", width=3),
                marker=dict(size=8, color="darkblue"),
                fill="tonexty" if len(hypervolume_history) > 1 else None,
            )
        )

        # Add reference point info if provided
        ref_text = ""
        if reference_point is not None:
            ref_text = f"<br>Reference Point: {reference_point}"

        fig.update_layout(
            title=title + ref_text,
            xaxis_title="Generation",
            yaxis_title="Hypervolume",
            template=self.theme,
            width=800,
            height=500,
            showlegend=False,
        )

        return fig

    def plot_objective_distribution(
        self,
        objectives: np.ndarray,
        pareto_mask: np.ndarray,
        objective_names: List[str],
        title: str = "Objective Value Distributions",
    ) -> go.Figure:
        """
        Plot distribution of objective values.

        Args:
            objectives: Array of objective values
            pareto_mask: Boolean mask for Pareto optimal points
            objective_names: Names of objectives
            title: Plot title

        Returns:
            Plotly figure object
        """
        n_objectives = objectives.shape[1]

        fig = make_subplots(
            rows=1, cols=n_objectives, subplot_titles=objective_names, shared_yaxis=True
        )

        for i, obj_name in enumerate(objective_names):
            # All points
            fig.add_trace(
                go.Histogram(
                    x=objectives[:, i],
                    name=f"All Solutions - {obj_name}",
                    opacity=0.6,
                    nbinsx=30,
                    marker_color="lightblue",
                ),
                row=1,
                col=i + 1,
            )

            # Pareto points
            pareto_objectives = objectives[pareto_mask]
            if len(pareto_objectives) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=pareto_objectives[:, i],
                        name=f"Pareto Solutions - {obj_name}",
                        opacity=0.8,
                        nbinsx=15,
                        marker_color="red",
                    ),
                    row=1,
                    col=i + 1,
                )

        fig.update_layout(
            title=title,
            template=self.theme,
            width=300 * n_objectives,
            height=500,
            barmode="overlay",
        )

        return fig

    def plot_trade_off_analysis(
        self,
        objectives: np.ndarray,
        pareto_mask: np.ndarray,
        objective_names: List[str],
        title: str = "Trade-off Analysis",
    ) -> go.Figure:
        """
        Create trade-off analysis visualization showing objective correlations.

        Args:
            objectives: Array of objective values
            pareto_mask: Boolean mask for Pareto optimal points
            objective_names: Names of objectives
            title: Plot title

        Returns:
            Plotly figure object
        """
        n_objectives = objectives.shape[1]
        pareto_objectives = objectives[pareto_mask]

        # Calculate correlation matrix for Pareto solutions
        if len(pareto_objectives) > 1:
            correlation_matrix = np.corrcoef(pareto_objectives.T)
        else:
            correlation_matrix = np.eye(n_objectives)

        # Create heatmap
        fig = ff.create_annotated_heatmap(
            z=correlation_matrix,
            x=objective_names,
            y=objective_names,
            annotation_text=np.around(correlation_matrix, decimals=2),
            colorscale="RdBu",
            showscale=True,
        )

        fig.update_layout(
            title=title + " (Pareto Solutions Correlation)",
            template=self.theme,
            width=600,
            height=600,
        )

        return fig

    def create_interactive_dashboard(
        self,
        result: OptimizationResult,
        objective_names: List[str],
        parameter_names: List[str],
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive interactive dashboard with multiple plots.

        Args:
            result: OptimizationResult containing all optimization data
            objective_names: Names of objectives
            parameter_names: Names of parameters

        Returns:
            Dictionary mapping plot names to figure objects
        """
        pareto_mask = np.zeros(len(result.objectives), dtype=bool)

        # Find Pareto optimal solutions
        for i, obj in enumerate(result.objectives):
            is_pareto = True
            for j, other_obj in enumerate(result.objectives):
                if i != j and np.all(other_obj <= obj) and np.any(other_obj < obj):
                    is_pareto = False
                    break
            pareto_mask[i] = is_pareto

        dashboard = {}

        # 2D Pareto front (if 2 objectives)
        if result.objectives.shape[1] == 2:
            dashboard["pareto_2d"] = self.plot_pareto_front_2d(
                result.objectives, pareto_mask, objective_names
            )

        # 3D Pareto front (if 3 objectives)
        elif result.objectives.shape[1] == 3:
            dashboard["pareto_3d"] = self.plot_pareto_front_3d(
                result.objectives, pareto_mask, objective_names
            )

        # Parallel coordinates
        dashboard["parallel_coords"] = self.plot_parallel_coordinates(
            result.objectives,
            result.parameters,
            pareto_mask,
            objective_names,
            parameter_names,
        )

        # Convergence history
        if result.convergence_history:
            convergence_data = {"Hypervolume": result.convergence_history}
            dashboard["convergence"] = self.plot_convergence_history(convergence_data)

        # Objective distributions
        dashboard["distributions"] = self.plot_objective_distribution(
            result.objectives, pareto_mask, objective_names
        )

        # Trade-off analysis
        dashboard["tradeoffs"] = self.plot_trade_off_analysis(
            result.objectives, pareto_mask, objective_names
        )

        return dashboard

    def save_dashboard_html(
        self,
        dashboard: Dict[str, go.Figure],
        filename: str = "optimization_dashboard.html",
    ):
        """
        Save interactive dashboard as HTML file.

        Args:
            dashboard: Dictionary of figure objects
            filename: Output HTML filename
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
        <title>Multi-Objective Optimization Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .plot-container { margin: 20px 0; }
        h1, h2 { color: #333; }
        </style>
        </head>
        <body>
        <h1>Multi-Objective Optimization Dashboard</h1>
        """

        for plot_name, fig in dashboard.items():
            html_content += f'<div class="plot-container"><h2>{plot_name.replace("_", " ").title()}</h2>'
            html_content += fig.to_html(include_plotlyjs=False, div_id=plot_name)
            html_content += "</div>"

        html_content += """
        </body>
        </html>
        """

        with open(filename, "w") as f:
            f.write(html_content)

        print(f"Dashboard saved to {filename}")

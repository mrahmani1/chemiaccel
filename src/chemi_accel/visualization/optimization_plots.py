"""
Optimization Visualization Tools

Additional plotting utilities for optimization analysis and general-purpose visualization.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class OptimizationPlotter:
    """
    General-purpose optimization visualization tools.

    Features:
    - Objective function landscapes
    - Parameter sensitivity analysis
    - Optimization history tracking
    - Constraint visualization
    - Uncertainty quantification plots
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the plotter.

        Args:
            theme: Plotly theme for styling
        """
        self.theme = theme
        self.colors = px.colors.qualitative.Set1

    def plot_objective_landscape_2d(
        self,
        objective_function: callable,
        param_ranges: Dict[str, Tuple[float, float]],
        param_names: List[str],
        resolution: int = 50,
        title: str = "Objective Function Landscape",
    ) -> go.Figure:
        """
        Plot 2D objective function landscape.

        Args:
            objective_function: Function to evaluate (takes dict of parameters)
            param_ranges: Parameter ranges {param_name: (min, max)}
            param_names: Names of parameters to plot (must be exactly 2)
            resolution: Grid resolution for landscape
            title: Plot title

        Returns:
            Plotly figure object
        """
        if len(param_names) != 2:
            raise ValueError("Exactly 2 parameters required for 2D landscape")

        param1, param2 = param_names

        # Create parameter grids
        x_range = param_ranges[param1]
        y_range = param_ranges[param2]

        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate objective function
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                params = {param1: X[i, j], param2: Y[i, j]}
                try:
                    Z[i, j] = objective_function(params)
                except:
                    Z[i, j] = np.nan

        fig = go.Figure()

        # Contour plot
        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=Z,
                colorscale="Viridis",
                contours=dict(showlabels=True, labelfont=dict(size=12)),
                hovertemplate=f"{param1}: %{{x:.3f}}<br>{param2}: %{{y:.3f}}<br>Objective: %{{z:.3f}}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=param1,
            yaxis_title=param2,
            template=self.theme,
            width=700,
            height=600,
        )

        return fig

    def plot_parameter_sensitivity(
        self,
        sensitivity_data: Dict[str, float],
        title: str = "Parameter Sensitivity Analysis",
    ) -> go.Figure:
        """
        Plot parameter sensitivity analysis.

        Args:
            sensitivity_data: Dict mapping parameter names to sensitivity values
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Sort by absolute sensitivity
        sorted_params = sorted(
            sensitivity_data.items(), key=lambda x: abs(x[1]), reverse=True
        )

        params, sensitivities = zip(*sorted_params)

        # Color by positive/negative
        colors = ["red" if s < 0 else "blue" for s in sensitivities]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=list(params),
                y=list(sensitivities),
                marker_color=colors,
                hovertemplate="Parameter: %{x}<br>Sensitivity: %{y:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Parameters",
            yaxis_title="Sensitivity",
            template=self.theme,
            width=800,
            height=500,
        )

        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        return fig

    def plot_optimization_history(
        self, history_data: Dict[str, List[float]], title: str = "Optimization History"
    ) -> go.Figure:
        """
        Plot optimization history with multiple metrics.

        Args:
            history_data: Dict mapping metric names to value histories
            title: Plot title

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        for i, (metric_name, values) in enumerate(history_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode="lines+markers",
                    name=metric_name,
                    line=dict(color=self.colors[i % len(self.colors)], width=2),
                    marker=dict(size=6),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Iteration",
            yaxis_title="Metric Value",
            template=self.theme,
            width=800,
            height=500,
            hovermode="x unified",
        )

        return fig

    def plot_constraint_visualization(
        self,
        feasible_region: np.ndarray,
        infeasible_region: np.ndarray,
        param_names: List[str],
        title: str = "Constraint Visualization",
    ) -> go.Figure:
        """
        Visualize feasible and infeasible regions.

        Args:
            feasible_region: Array of feasible parameter combinations
            infeasible_region: Array of infeasible parameter combinations
            param_names: Parameter names
            title: Plot title

        Returns:
            Plotly figure object
        """
        if len(param_names) != 2:
            raise ValueError(
                "Exactly 2 parameters required for constraint visualization"
            )

        fig = go.Figure()

        # Feasible region
        if len(feasible_region) > 0:
            fig.add_trace(
                go.Scatter(
                    x=feasible_region[:, 0],
                    y=feasible_region[:, 1],
                    mode="markers",
                    name="Feasible",
                    marker=dict(color="green", size=6, opacity=0.6),
                    hovertemplate=f"{param_names[0]}: %{{x:.3f}}<br>{param_names[1]}: %{{y:.3f}}<br>Status: Feasible<extra></extra>",
                )
            )

        # Infeasible region
        if len(infeasible_region) > 0:
            fig.add_trace(
                go.Scatter(
                    x=infeasible_region[:, 0],
                    y=infeasible_region[:, 1],
                    mode="markers",
                    name="Infeasible",
                    marker=dict(color="red", size=6, opacity=0.6, symbol="x"),
                    hovertemplate=f"{param_names[0]}: %{{x:.3f}}<br>{param_names[1]}: %{{y:.3f}}<br>Status: Infeasible<extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=param_names[0],
            yaxis_title=param_names[1],
            template=self.theme,
            width=700,
            height=600,
            hovermode="closest",
        )

        return fig

    def plot_uncertainty_quantification(
        self,
        mean_predictions: np.ndarray,
        std_predictions: np.ndarray,
        x_values: np.ndarray,
        true_values: Optional[np.ndarray] = None,
        confidence_level: float = 0.95,
        title: str = "Uncertainty Quantification",
    ) -> go.Figure:
        """
        Plot predictions with uncertainty bounds.

        Args:
            mean_predictions: Mean predicted values
            std_predictions: Standard deviation of predictions
            x_values: X-axis values (e.g., parameter values or sample indices)
            true_values: Optional true values for comparison
            confidence_level: Confidence level for uncertainty bounds
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Calculate confidence intervals
        from scipy import stats

        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)

        upper_bound = mean_predictions + z_score * std_predictions
        lower_bound = mean_predictions - z_score * std_predictions

        fig = go.Figure()

        # Uncertainty band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_values, x_values[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{confidence_level:.0%} Confidence",
                hoverinfo="skip",
            )
        )

        # Mean prediction
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=mean_predictions,
                mode="lines",
                name="Mean Prediction",
                line=dict(color="blue", width=2),
                hovertemplate="X: %{x:.3f}<br>Mean: %{y:.3f}<extra></extra>",
            )
        )

        # True values (if provided)
        if true_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=true_values,
                    mode="markers",
                    name="True Values",
                    marker=dict(color="red", size=6, symbol="circle"),
                    hovertemplate="X: %{x:.3f}<br>True: %{y:.3f}<extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Input",
            yaxis_title="Prediction",
            template=self.theme,
            width=800,
            height=500,
            hovermode="x unified",
        )

        return fig

    def plot_pareto_set_evolution(
        self,
        pareto_sets: List[np.ndarray],
        generation_labels: List[str],
        objective_names: List[str],
        title: str = "Pareto Set Evolution",
    ) -> go.Figure:
        """
        Plot evolution of Pareto sets over generations.

        Args:
            pareto_sets: List of Pareto set arrays for each generation
            generation_labels: Labels for each generation
            objective_names: Names of objectives
            title: Plot title

        Returns:
            Plotly figure object
        """
        if len(objective_names) != 2:
            raise ValueError("Currently supports only 2 objectives")

        fig = go.Figure()

        # Color mapping for generations
        colors = px.colors.qualitative.Set1

        for i, (pareto_set, gen_label) in enumerate(
            zip(pareto_sets, generation_labels)
        ):
            if len(pareto_set) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=pareto_set[:, 0],
                        y=pareto_set[:, 1],
                        mode="markers+lines",
                        name=gen_label,
                        marker=dict(color=colors[i % len(colors)], size=8, opacity=0.8),
                        line=dict(color=colors[i % len(colors)], width=2, dash="dash"),
                        hovertemplate=f"{objective_names[0]}: %{{x:.3f}}<br>{objective_names[1]}: %{{y:.3f}}<br>Generation: {gen_label}<extra></extra>",
                    )
                )

        fig.update_layout(
            title=title,
            xaxis_title=objective_names[0],
            yaxis_title=objective_names[1],
            template=self.theme,
            width=800,
            height=600,
            hovermode="closest",
        )

        return fig

    def plot_algorithm_comparison(
        self,
        algorithm_results: Dict[str, Dict[str, List[float]]],
        title: str = "Algorithm Performance Comparison",
    ) -> go.Figure:
        """
        Compare performance of different optimization algorithms.

        Args:
            algorithm_results: Dict mapping algorithm names to their metric histories
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Get all metrics
        all_metrics = set()
        for results in algorithm_results.values():
            all_metrics.update(results.keys())

        all_metrics = list(all_metrics)
        n_metrics = len(all_metrics)

        if n_metrics == 0:
            return go.Figure().add_annotation(
                text="No metrics available for comparison",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Create subplots
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols, subplot_titles=all_metrics, vertical_spacing=0.1
        )

        colors = px.colors.qualitative.Set1

        for i, metric in enumerate(all_metrics):
            row = i // n_cols + 1
            col = i % n_cols + 1

            for j, (alg_name, results) in enumerate(algorithm_results.items()):
                if metric in results:
                    values = results[metric]
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(values))),
                            y=values,
                            mode="lines+markers",
                            name=alg_name,
                            line=dict(color=colors[j % len(colors)], width=2),
                            marker=dict(size=4),
                            legendgroup=alg_name,
                            showlegend=(i == 0),  # Only show legend for first metric
                        ),
                        row=row,
                        col=col,
                    )

            fig.update_xaxes(title_text="Generation", row=row, col=col)
            fig.update_yaxes(title_text=metric, row=row, col=col)

        fig.update_layout(
            title=title,
            template=self.theme,
            width=400 * n_cols,
            height=350 * n_rows,
            hovermode="x unified",
        )

        return fig

    def plot_design_space_coverage(
        self,
        sampled_points: np.ndarray,
        param_names: List[str],
        param_ranges: Dict[str, Tuple[float, float]],
        title: str = "Design Space Coverage",
    ) -> go.Figure:
        """
        Visualize coverage of design space by sampled points.

        Args:
            sampled_points: Array of sampled parameter combinations
            param_names: Parameter names
            param_ranges: Parameter ranges
            title: Plot title

        Returns:
            Plotly figure object
        """
        if len(param_names) != 2:
            raise ValueError("Currently supports only 2 parameters")

        param1, param2 = param_names

        fig = go.Figure()

        # Sampled points
        fig.add_trace(
            go.Scatter(
                x=sampled_points[:, 0],
                y=sampled_points[:, 1],
                mode="markers",
                name="Sampled Points",
                marker=dict(color="blue", size=6, opacity=0.7),
                hovertemplate=f"{param1}: %{{x:.3f}}<br>{param2}: %{{y:.3f}}<extra></extra>",
            )
        )

        # Design space boundaries
        x_range = param_ranges[param1]
        y_range = param_ranges[param2]

        fig.add_shape(
            type="rect",
            x0=x_range[0],
            y0=y_range[0],
            x1=x_range[1],
            y1=y_range[1],
            line=dict(color="red", width=2, dash="dash"),
            fillcolor="rgba(255,0,0,0.1)",
        )

        fig.update_layout(
            title=title,
            xaxis_title=param1,
            yaxis_title=param2,
            template=self.theme,
            width=700,
            height=600,
            hovermode="closest",
        )

        return fig

    def create_optimization_summary_dashboard(
        self, optimization_data: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive optimization summary dashboard.

        Args:
            optimization_data: Dictionary containing optimization results and metadata

        Returns:
            Dictionary of figure objects
        """
        dashboard = {}

        # Extract data
        history = optimization_data.get("history", {})
        sensitivity = optimization_data.get("sensitivity", {})
        final_population = optimization_data.get("final_population", np.array([]))
        param_names = optimization_data.get("parameter_names", [])

        # Optimization history
        if history:
            dashboard["history"] = self.plot_optimization_history(history)

        # Parameter sensitivity
        if sensitivity:
            dashboard["sensitivity"] = self.plot_parameter_sensitivity(sensitivity)

        # Design space coverage
        if len(final_population) > 0 and len(param_names) >= 2:
            param_ranges = optimization_data.get("parameter_ranges", {})
            if param_ranges:
                dashboard["coverage"] = self.plot_design_space_coverage(
                    final_population[:, :2],  # First 2 parameters
                    param_names[:2],
                    param_ranges,
                )

        return dashboard

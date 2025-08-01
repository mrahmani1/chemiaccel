"""
Polymer Property Prediction Dashboards

Interactive dashboards for polymer property visualization, composition analysis,
and structure-property relationship exploration.
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
class PolymerData:
    """Container for polymer data."""

    name: str
    monomers: List[str]
    composition: Dict[str, float]
    properties: Dict[str, float]
    molecular_weight: float
    structure_representation: str


@dataclass
class CopolymerData:
    """Container for copolymer data."""

    monomer_a: str
    monomer_b: str
    ratio_a: float
    ratio_b: float
    properties: Dict[str, float]
    sequence_info: Optional[Dict[str, Any]] = None


class PolymerPropertyDashboard:
    """
    Interactive dashboards for polymer property analysis and visualization.

    Features:
    - Property prediction visualization
    - Composition-property relationships
    - Structure-property correlations
    - Copolymer analysis
    - Thermal property analysis
    - Mechanical property visualization
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the dashboard.

        Args:
            theme: Plotly theme for styling
        """
        self.theme = theme
        self.colors = px.colors.qualitative.Set1

    def create_property_overview_dashboard(
        self, polymers: List[PolymerData], title: str = "Polymer Property Overview"
    ) -> go.Figure:
        """
        Create overview dashboard of polymer properties.

        Args:
            polymers: List of PolymerData objects
            title: Dashboard title

        Returns:
            Plotly figure object
        """
        if not polymers:
            return go.Figure().add_annotation(
                text="No polymer data provided",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Extract common properties
        all_properties = set()
        for polymer in polymers:
            all_properties.update(polymer.properties.keys())

        common_properties = list(all_properties)[:6]  # Limit to 6 for visualization

        if not common_properties:
            return go.Figure().add_annotation(
                text="No common properties found",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Create subplots
        n_props = len(common_properties)
        n_cols = min(3, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=common_properties,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        polymer_names = [p.name for p in polymers]

        for i, prop in enumerate(common_properties):
            row = i // n_cols + 1
            col = i % n_cols + 1

            # Extract property values
            values = []
            labels = []
            for polymer in polymers:
                if prop in polymer.properties:
                    values.append(polymer.properties[prop])
                    labels.append(polymer.name)

            if values:
                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=values,
                        name=prop,
                        marker_color=self.colors[i % len(self.colors)],
                        showlegend=False,
                        hovertemplate=f"{prop}: %{{y:.3f}}<br>Polymer: %{{x}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

                fig.update_xaxes(tickangle=45, row=row, col=col)
                fig.update_yaxes(title_text=prop, row=row, col=col)

        fig.update_layout(
            title=title,
            template=self.theme,
            width=400 * n_cols,
            height=350 * n_rows,
            showlegend=False,
        )

        return fig

    def create_composition_property_plot(
        self,
        copolymers: List[CopolymerData],
        property_name: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create composition vs property visualization for copolymers.

        Args:
            copolymers: List of CopolymerData objects
            property_name: Property to visualize
            title: Plot title

        Returns:
            Plotly figure object
        """
        if not copolymers:
            return go.Figure().add_annotation(
                text="No copolymer data provided",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        if title is None:
            title = f"Composition vs {property_name}"

        # Extract data
        ratios_a = []
        property_values = []
        hover_text = []

        for copolymer in copolymers:
            if property_name in copolymer.properties:
                ratios_a.append(copolymer.ratio_a)
                property_values.append(copolymer.properties[property_name])
                hover_text.append(
                    f"Monomer A: {copolymer.monomer_a}<br>"
                    f"Monomer B: {copolymer.monomer_b}<br>"
                    f"Ratio A: {copolymer.ratio_a:.3f}<br>"
                    f"Ratio B: {copolymer.ratio_b:.3f}<br>"
                    f"{property_name}: {copolymer.properties[property_name]:.3f}"
                )

        if not ratios_a:
            return go.Figure().add_annotation(
                text=f"No data available for property: {property_name}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        fig = go.Figure()

        # Scatter plot of data points
        fig.add_trace(
            go.Scatter(
                x=ratios_a,
                y=property_values,
                mode="markers",
                name="Copolymers",
                marker=dict(
                    size=10,
                    color=property_values,
                    colorscale="Viridis",
                    colorbar=dict(title=property_name),
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_text,
            )
        )

        # Add trend line if enough data points
        if len(ratios_a) >= 5:
            try:
                z = np.polyfit(ratios_a, property_values, 2)  # Quadratic fit
                x_trend = np.linspace(min(ratios_a), max(ratios_a), 100)
                y_trend = np.polyval(z, x_trend)

                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode="lines",
                        name="Trend",
                        line=dict(color="red", width=2, dash="dash"),
                        hoverinfo="skip",
                    )
                )
            except:
                pass  # Skip trend line if fitting fails

        fig.update_layout(
            title=title,
            xaxis_title="Fraction of Monomer A",
            yaxis_title=property_name,
            template=self.theme,
            width=800,
            height=600,
            hovermode="closest",
        )

        return fig

    def create_thermal_properties_dashboard(
        self, polymers: List[PolymerData], title: str = "Thermal Properties Analysis"
    ) -> go.Figure:
        """
        Create thermal properties analysis dashboard.

        Args:
            polymers: List of PolymerData objects
            title: Dashboard title

        Returns:
            Plotly figure object
        """
        thermal_props = [
            "glass_transition_temp",
            "melting_temp",
            "decomposition_temp",
            "thermal_conductivity",
            "heat_capacity",
        ]

        # Filter available thermal properties
        available_props = []
        for prop in thermal_props:
            if any(prop in polymer.properties for polymer in polymers):
                available_props.append(prop)

        if not available_props:
            return go.Figure().add_annotation(
                text="No thermal properties available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        n_props = len(available_props)
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=available_props[:4],  # Limit to 4 subplots
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for i, prop in enumerate(available_props[:4]):
            row, col = positions[i]

            # Extract data
            values = []
            names = []
            for polymer in polymers:
                if prop in polymer.properties:
                    values.append(polymer.properties[prop])
                    names.append(polymer.name)

            if values:
                # Box plot for distribution
                fig.add_trace(
                    go.Box(
                        y=values,
                        name=prop,
                        marker_color=self.colors[i % len(self.colors)],
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

                # Add individual points
                fig.add_trace(
                    go.Scatter(
                        x=[prop] * len(values),
                        y=values,
                        mode="markers",
                        name=f"{prop}_points",
                        marker=dict(
                            size=8, color=self.colors[i % len(self.colors)], opacity=0.6
                        ),
                        text=names,
                        hovertemplate="%{text}: %{y:.3f}<extra></extra>",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            title=title, template=self.theme, width=800, height=700, showlegend=False
        )

        return fig

    def create_mechanical_properties_dashboard(
        self, polymers: List[PolymerData], title: str = "Mechanical Properties Analysis"
    ) -> go.Figure:
        """
        Create mechanical properties analysis dashboard.

        Args:
            polymers: List of PolymerData objects
            title: Dashboard title

        Returns:
            Plotly figure object
        """
        mechanical_props = [
            "youngs_modulus",
            "tensile_strength",
            "elongation_at_break",
            "toughness",
            "hardness",
            "impact_strength",
        ]

        # Filter available properties
        available_props = []
        for prop in mechanical_props:
            if any(prop in polymer.properties for polymer in polymers):
                available_props.append(prop)

        if len(available_props) < 2:
            return go.Figure().add_annotation(
                text="Insufficient mechanical properties for analysis",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Create correlation matrix
        prop_data = {}
        polymer_names = []

        for polymer in polymers:
            valid_props = {
                prop: polymer.properties[prop]
                for prop in available_props
                if prop in polymer.properties
            }

            if len(valid_props) >= 2:  # Need at least 2 properties
                for prop, value in valid_props.items():
                    if prop not in prop_data:
                        prop_data[prop] = []
                    prop_data[prop].append(value)
                polymer_names.append(polymer.name)

        if not prop_data:
            return go.Figure().add_annotation(
                text="No valid mechanical property data",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Create DataFrame
        df = pd.DataFrame(prop_data)

        # Calculate correlation matrix
        if len(df.columns) >= 2:
            corr_matrix = df.corr()

            # Create subplots
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Property Correlations", "Property Distributions"],
                specs=[[{"type": "heatmap"}, {"type": "violin"}]],
                column_widths=[0.5, 0.5],
            )

            # Correlation heatmap
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                    hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Violin plots for distributions
            for i, prop in enumerate(df.columns):
                fig.add_trace(
                    go.Violin(
                        y=df[prop],
                        name=prop,
                        box_visible=True,
                        meanline_visible=True,
                        marker_color=self.colors[i % len(self.colors)],
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
        else:
            # Single property case
            prop = list(prop_data.keys())[0]
            values = prop_data[prop]

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=values, nbinsx=20, marker_color=self.colors[0], opacity=0.7
                )
            )

            fig.update_layout(
                title=f"{title} - {prop}", xaxis_title=prop, yaxis_title="Count"
            )

        fig.update_layout(title=title, template=self.theme, width=1000, height=500)

        return fig

    def create_structure_property_correlation(
        self,
        polymers: List[PolymerData],
        structural_feature: str,
        property_name: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create structure-property correlation plot.

        Args:
            polymers: List of PolymerData objects
            structural_feature: Name of structural feature
            property_name: Name of property to correlate
            title: Plot title

        Returns:
            Plotly figure object
        """
        if title is None:
            title = f"{structural_feature} vs {property_name}"

        # Extract data
        x_values = []
        y_values = []
        hover_text = []

        for polymer in polymers:
            if (
                structural_feature in polymer.properties
                and property_name in polymer.properties
            ):
                x_values.append(polymer.properties[structural_feature])
                y_values.append(polymer.properties[property_name])
                hover_text.append(
                    f"Polymer: {polymer.name}<br>"
                    f"{structural_feature}: {polymer.properties[structural_feature]:.3f}<br>"
                    f"{property_name}: {polymer.properties[property_name]:.3f}<br>"
                    f"MW: {polymer.molecular_weight:.0f} g/mol"
                )

        if len(x_values) < 2:
            return go.Figure().add_annotation(
                text="Insufficient data for correlation analysis",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        fig = go.Figure()

        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name="Polymers",
                marker=dict(
                    size=10,
                    color="blue",
                    opacity=0.7,
                    line=dict(width=1, color="darkblue"),
                ),
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_text,
            )
        )

        # Add trend line
        if len(x_values) >= 3:
            try:
                z = np.polyfit(x_values, y_values, 1)  # Linear fit
                x_trend = np.linspace(min(x_values), max(x_values), 100)
                y_trend = np.polyval(z, x_trend)

                # Calculate R²
                y_pred = np.polyval(z, x_values)
                ss_res = np.sum((y_values - y_pred) ** 2)
                ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode="lines",
                        name=f"Trend (R² = {r_squared:.3f})",
                        line=dict(color="red", width=2, dash="dash"),
                    )
                )
            except:
                pass

        fig.update_layout(
            title=title,
            xaxis_title=structural_feature,
            yaxis_title=property_name,
            template=self.theme,
            width=800,
            height=600,
            hovermode="closest",
        )

        return fig

    def create_molecular_weight_analysis(
        self, polymers: List[PolymerData], title: str = "Molecular Weight vs Properties"
    ) -> go.Figure:
        """
        Create molecular weight analysis dashboard.

        Args:
            polymers: List of PolymerData objects
            title: Dashboard title

        Returns:
            Plotly figure object
        """
        # Get properties that correlate with molecular weight
        weight_dependent_props = [
            "viscosity",
            "tensile_strength",
            "toughness",
            "glass_transition_temp",
        ]

        available_props = []
        for prop in weight_dependent_props:
            if any(prop in polymer.properties for polymer in polymers):
                available_props.append(prop)

        if not available_props:
            available_props = list(
                set().union(*[p.properties.keys() for p in polymers])
            )[:4]

        if not available_props:
            return go.Figure().add_annotation(
                text="No properties available for molecular weight analysis",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Create subplots
        n_props = min(4, len(available_props))
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[f"MW vs {prop}" for prop in available_props[:4]],
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for i, prop in enumerate(available_props[:4]):
            row, col = positions[i]

            # Extract data
            mw_values = []
            prop_values = []
            names = []

            for polymer in polymers:
                if prop in polymer.properties:
                    mw_values.append(polymer.molecular_weight)
                    prop_values.append(polymer.properties[prop])
                    names.append(polymer.name)

            if len(mw_values) >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=mw_values,
                        y=prop_values,
                        mode="markers",
                        name=prop,
                        marker=dict(
                            size=8, color=self.colors[i % len(self.colors)], opacity=0.7
                        ),
                        text=names,
                        hovertemplate="%{text}<br>MW: %{x:.0f}<br>"
                        + f"{prop}: %{{y:.3f}}<extra></extra>",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

                # Add trend line
                if len(mw_values) >= 3:
                    try:
                        z = np.polyfit(mw_values, prop_values, 1)
                        x_trend = np.linspace(min(mw_values), max(mw_values), 50)
                        y_trend = np.polyval(z, x_trend)

                        fig.add_trace(
                            go.Scatter(
                                x=x_trend,
                                y=y_trend,
                                mode="lines",
                                name=f"{prop}_trend",
                                line=dict(color="red", width=1, dash="dash"),
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=row,
                            col=col,
                        )
                    except:
                        pass

                fig.update_xaxes(
                    title_text="Molecular Weight (g/mol)", row=row, col=col
                )
                fig.update_yaxes(title_text=prop, row=row, col=col)

        fig.update_layout(
            title=title, template=self.theme, width=900, height=700, showlegend=False
        )

        return fig

    def create_comprehensive_polymer_dashboard(
        self,
        polymers: List[PolymerData],
        copolymers: Optional[List[CopolymerData]] = None,
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive polymer analysis dashboard.

        Args:
            polymers: List of PolymerData objects
            copolymers: Optional list of CopolymerData objects

        Returns:
            Dictionary of figure objects
        """
        dashboard = {}

        # Property overview
        dashboard["overview"] = self.create_property_overview_dashboard(polymers)

        # Thermal properties
        dashboard["thermal"] = self.create_thermal_properties_dashboard(polymers)

        # Mechanical properties
        dashboard["mechanical"] = self.create_mechanical_properties_dashboard(polymers)

        # Molecular weight analysis
        dashboard["molecular_weight"] = self.create_molecular_weight_analysis(polymers)

        # Copolymer analysis if data available
        if copolymers:
            # Find common properties in copolymers
            common_props = set()
            for copolymer in copolymers:
                common_props.update(copolymer.properties.keys())

            for prop in list(common_props)[:3]:  # Limit to 3 properties
                dashboard[f"copolymer_{prop}"] = self.create_composition_property_plot(
                    copolymers, prop
                )

        return dashboard

    def export_dashboard_html(
        self, dashboard: Dict[str, go.Figure], filename: str = "polymer_dashboard.html"
    ):
        """
        Export dashboard as HTML file.

        Args:
            dashboard: Dictionary of figure objects
            filename: Output filename
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
        <title>Polymer Property Analysis Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f6fa;
        }
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 30px;
        }
        .plot-container {
            background: white;
            margin: 30px 0;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #3498db;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }
        h2 {
            color: #34495e;
            font-size: 1.5em;
            margin-bottom: 20px;
        }
        .description {
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 15px;
            line-height: 1.6;
        }
        </style>
        </head>
        <body>
        <div class="dashboard-container">
        <h1>🧬 Polymer Property Analysis Dashboard</h1>
        <div class="description">
        Interactive visualization of polymer properties, structure-property relationships, and copolymer analysis.
        </div>
        """

        plot_descriptions = {
            "overview": "Comprehensive overview of key polymer properties across all samples.",
            "thermal": "Analysis of thermal properties including glass transition and melting temperatures.",
            "mechanical": "Mechanical property analysis including correlations and distributions.",
            "molecular_weight": "Relationship between molecular weight and various polymer properties.",
            "copolymer": "Copolymer composition-property relationships.",
        }

        for plot_name, fig in dashboard.items():
            plot_title = plot_name.replace("_", " ").title()
            description = plot_descriptions.get(
                plot_name.split("_")[0], "Detailed analysis of polymer characteristics."
            )

            html_content += '<div class="plot-container">'
            html_content += f"<h2>{plot_title}</h2>"
            html_content += f'<div class="description">{description}</div>'
            html_content += fig.to_html(include_plotlyjs=False, div_id=plot_name)
            html_content += "</div>"

        html_content += """
        </div>
        </body>
        </html>
        """

        with open(filename, "w") as f:
            f.write(html_content)

        print(f"Polymer dashboard exported to {filename}")

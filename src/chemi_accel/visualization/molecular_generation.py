"""
Molecular Generation Result Visualization

Interactive tools for visualizing generated molecules, property distributions,
and chemical space exploration.
"""

import base64
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


@dataclass
class GeneratedMolecule:
    """Container for generated molecule data."""

    smiles: str
    mol: Optional[Chem.Mol]
    properties: Dict[str, float]
    generation_score: float
    validity_score: float
    novelty_score: float
    diversity_score: float


class MolecularGenerationVisualizer:
    """
    Interactive visualization tools for molecular generation results.

    Features:
    - Generated molecule galleries
    - Property distribution analysis
    - Chemical space visualization
    - Generation quality metrics
    - Scaffold and fragment analysis
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the visualizer.

        Args:
            theme: Plotly theme for styling
        """
        self.theme = theme
        self.colors = px.colors.qualitative.Set1

    def mol_to_svg(self, mol: Chem.Mol, size: Tuple[int, int] = (300, 300)) -> str:
        """
        Convert RDKit molecule to SVG string.

        Args:
            mol: RDKit molecule object
            size: Image size (width, height)

        Returns:
            SVG string representation
        """
        if mol is None:
            return ""

        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def create_molecule_gallery(
        self,
        molecules: List[GeneratedMolecule],
        sort_by: str = "generation_score",
        n_display: int = 20,
        title: str = "Generated Molecules Gallery",
    ) -> go.Figure:
        """
        Create interactive gallery of generated molecules.

        Args:
            molecules: List of GeneratedMolecule objects
            sort_by: Property to sort molecules by
            n_display: Number of molecules to display
            title: Gallery title

        Returns:
            Plotly figure object
        """
        # Sort molecules
        sorted_molecules = sorted(
            molecules, key=lambda x: getattr(x, sort_by), reverse=True
        )[:n_display]

        # Calculate grid dimensions
        n_cols = min(5, len(sorted_molecules))
        n_rows = (len(sorted_molecules) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[
                f"Score: {getattr(mol, sort_by):.3f}" for mol in sorted_molecules
            ],
            specs=[[{"type": "xy"}] * n_cols for _ in range(n_rows)],
        )

        for i, mol_data in enumerate(sorted_molecules):
            row = i // n_cols + 1
            col = i % n_cols + 1

            # Generate molecule image
            if mol_data.mol is not None:
                svg_text = self.mol_to_svg(mol_data.mol, (200, 200))

                # Create hover text with properties
                hover_text = f"SMILES: {mol_data.smiles}<br>"
                hover_text += f"Generation Score: {mol_data.generation_score:.3f}<br>"
                hover_text += f"Validity: {mol_data.validity_score:.3f}<br>"
                hover_text += f"Novelty: {mol_data.novelty_score:.3f}<br>"
                for prop, value in mol_data.properties.items():
                    hover_text += f"{prop}: {value:.3f}<br>"

                # Add invisible scatter point for hover functionality
                fig.add_trace(
                    go.Scatter(
                        x=[0.5],
                        y=[0.5],
                        mode="markers",
                        marker=dict(size=1, opacity=0),
                        hovertemplate=hover_text + "<extra></extra>",
                        showlegend=False,
                        name=f"Molecule {i+1}",
                    ),
                    row=row,
                    col=col,
                )

                # Add molecule structure as annotation
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    text=mol_data.smiles[:30]
                    + ("..." if len(mol_data.smiles) > 30 else ""),
                    showarrow=False,
                    row=row,
                    col=col,
                    font=dict(size=10),
                )

            # Remove axes for cleaner look
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)

        fig.update_layout(
            title=title,
            template=self.theme,
            width=300 * n_cols,
            height=300 * n_rows,
            showlegend=False,
        )

        return fig

    def plot_property_distributions(
        self,
        molecules: List[GeneratedMolecule],
        reference_molecules: Optional[List[GeneratedMolecule]] = None,
        properties: Optional[List[str]] = None,
        title: str = "Property Distributions",
    ) -> go.Figure:
        """
        Plot property distributions of generated molecules.

        Args:
            molecules: List of generated molecules
            reference_molecules: Optional reference molecules for comparison
            properties: List of properties to plot
            title: Plot title

        Returns:
            Plotly figure object
        """
        if not molecules:
            raise ValueError("No molecules provided")

        # Extract properties
        if properties is None:
            properties = list(molecules[0].properties.keys())

        n_props = len(properties)
        n_cols = min(3, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=properties,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )

        for i, prop in enumerate(properties):
            row = i // n_cols + 1
            col = i % n_cols + 1

            # Generated molecules
            values = [mol.properties.get(prop, np.nan) for mol in molecules]
            values = [v for v in values if not np.isnan(v)]

            if values:
                fig.add_trace(
                    go.Histogram(
                        x=values,
                        name="Generated",
                        opacity=0.7,
                        nbinsx=30,
                        marker_color=self.colors[0],
                        legendgroup="generated",
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=col,
                )

            # Reference molecules
            if reference_molecules:
                ref_values = [
                    mol.properties.get(prop, np.nan) for mol in reference_molecules
                ]
                ref_values = [v for v in ref_values if not np.isnan(v)]

                if ref_values:
                    fig.add_trace(
                        go.Histogram(
                            x=ref_values,
                            name="Reference",
                            opacity=0.7,
                            nbinsx=30,
                            marker_color=self.colors[1],
                            legendgroup="reference",
                            showlegend=(i == 0),
                        ),
                        row=row,
                        col=col,
                    )

            fig.update_xaxes(title_text=prop, row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)

        fig.update_layout(
            title=title,
            template=self.theme,
            width=400 * n_cols,
            height=300 * n_rows,
            barmode="overlay",
        )

        return fig

    def plot_chemical_space_2d(
        self,
        molecules: List[GeneratedMolecule],
        reference_molecules: Optional[List[GeneratedMolecule]] = None,
        x_property: str = "molecular_weight",
        y_property: str = "logp",
        color_by: str = "generation_score",
        title: str = "Chemical Space Visualization",
    ) -> go.Figure:
        """
        Create 2D chemical space visualization.

        Args:
            molecules: List of generated molecules
            reference_molecules: Optional reference molecules
            x_property: Property for x-axis
            y_property: Property for y-axis
            color_by: Property to color points by
            title: Plot title

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Generated molecules
        x_vals = [mol.properties.get(x_property, np.nan) for mol in molecules]
        y_vals = [mol.properties.get(y_property, np.nan) for mol in molecules]
        colors = [getattr(mol, color_by, np.nan) for mol in molecules]

        # Filter out NaN values
        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isnan(colors))
        x_vals = np.array(x_vals)[valid_mask]
        y_vals = np.array(y_vals)[valid_mask]
        colors = np.array(colors)[valid_mask]
        valid_molecules = [mol for i, mol in enumerate(molecules) if valid_mask[i]]

        if len(x_vals) > 0:
            hover_text = []
            for mol in valid_molecules:
                text = f"SMILES: {mol.smiles}<br>"
                text += f"{x_property}: {mol.properties.get(x_property, 'N/A')}<br>"
                text += f"{y_property}: {mol.properties.get(y_property, 'N/A')}<br>"
                text += f"{color_by}: {getattr(mol, color_by, 'N/A')}"
                hover_text.append(text)

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name="Generated Molecules",
                    marker=dict(
                        size=8,
                        color=colors,
                        colorscale="Viridis",
                        colorbar=dict(title=color_by),
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=hover_text,
                )
            )

        # Reference molecules
        if reference_molecules:
            ref_x = [
                mol.properties.get(x_property, np.nan) for mol in reference_molecules
            ]
            ref_y = [
                mol.properties.get(y_property, np.nan) for mol in reference_molecules
            ]

            ref_mask = ~(np.isnan(ref_x) | np.isnan(ref_y))
            ref_x = np.array(ref_x)[ref_mask]
            ref_y = np.array(ref_y)[ref_mask]

            if len(ref_x) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=ref_x,
                        y=ref_y,
                        mode="markers",
                        name="Reference Molecules",
                        marker=dict(
                            size=6,
                            color="lightgray",
                            opacity=0.6,
                            line=dict(width=1, color="gray"),
                        ),
                    )
                )

        fig.update_layout(
            title=title,
            xaxis_title=x_property,
            yaxis_title=y_property,
            template=self.theme,
            width=800,
            height=600,
            hovermode="closest",
        )

        return fig

    def plot_generation_metrics(
        self,
        molecules: List[GeneratedMolecule],
        title: str = "Generation Quality Metrics",
    ) -> go.Figure:
        """
        Plot generation quality metrics.

        Args:
            molecules: List of generated molecules
            title: Plot title

        Returns:
            Plotly figure object
        """
        metrics = [
            "validity_score",
            "novelty_score",
            "diversity_score",
            "generation_score",
        ]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[metric.replace("_", " ").title() for metric in metrics],
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for i, metric in enumerate(metrics):
            row, col = positions[i]
            values = [getattr(mol, metric, np.nan) for mol in molecules]
            values = [v for v in values if not np.isnan(v)]

            if values:
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=values,
                        nbinsx=30,
                        marker_color=self.colors[i % len(self.colors)],
                        opacity=0.7,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

                # Add mean line
                mean_val = np.mean(values)
                fig.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {mean_val:.3f}",
                    row=row,
                    col=col,
                )

        fig.update_layout(title=title, template=self.theme, width=800, height=600)

        return fig

    def plot_scaffold_analysis(
        self,
        molecules: List[GeneratedMolecule],
        top_n: int = 10,
        title: str = "Scaffold Analysis",
    ) -> go.Figure:
        """
        Analyze and visualize molecular scaffolds.

        Args:
            molecules: List of generated molecules
            top_n: Number of top scaffolds to show
            title: Plot title

        Returns:
            Plotly figure object
        """
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError:
            warnings.warn("RDKit scaffolds not available")
            return go.Figure()

        # Extract scaffolds
        scaffold_counts = {}
        for mol_data in molecules:
            if mol_data.mol is not None:
                try:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol_data.mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                    scaffold_counts[scaffold_smiles] = (
                        scaffold_counts.get(scaffold_smiles, 0) + 1
                    )
                except:
                    continue

        if not scaffold_counts:
            return go.Figure().add_annotation(
                text="No valid scaffolds found",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Get top scaffolds
        top_scaffolds = sorted(
            scaffold_counts.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        scaffolds, counts = zip(*top_scaffolds)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(range(len(scaffolds))),
                    y=counts,
                    text=[f"{s[:20]}..." if len(s) > 20 else s for s in scaffolds],
                    textposition="outside",
                    marker_color=px.colors.qualitative.Set1[0],
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Scaffold Rank",
            yaxis_title="Count",
            template=self.theme,
            width=800,
            height=500,
        )

        return fig

    def create_tanimoto_similarity_heatmap(
        self,
        molecules: List[GeneratedMolecule],
        sample_size: int = 50,
        title: str = "Molecular Similarity Heatmap",
    ) -> go.Figure:
        """
        Create Tanimoto similarity heatmap for molecular comparison.

        Args:
            molecules: List of generated molecules
            sample_size: Number of molecules to sample for comparison
            title: Plot title

        Returns:
            Plotly figure object
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import DataStructs
        except ImportError:
            warnings.warn("RDKit not available for similarity calculation")
            return go.Figure()

        # Sample molecules
        if len(molecules) > sample_size:
            sampled_molecules = np.random.choice(molecules, sample_size, replace=False)
        else:
            sampled_molecules = molecules

        # Calculate fingerprints
        fps = []
        valid_mols = []
        for mol_data in sampled_molecules:
            if mol_data.mol is not None:
                try:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_data.mol, 2)
                    fps.append(fp)
                    valid_mols.append(mol_data)
                except:
                    continue

        if len(fps) < 2:
            return go.Figure().add_annotation(
                text="Insufficient valid molecules for similarity analysis",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Calculate similarity matrix
        n_mols = len(fps)
        similarity_matrix = np.zeros((n_mols, n_mols))

        for i in range(n_mols):
            for j in range(n_mols):
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarity_matrix[i][j] = similarity

        # Create labels
        labels = [f"Mol_{i+1}" for i in range(n_mols)]

        fig = go.Figure(
            data=go.Heatmap(
                z=similarity_matrix,
                x=labels,
                y=labels,
                colorscale="RdYlBu_r",
                zmin=0,
                zmax=1,
                hovertemplate="Similarity: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(title=title, template=self.theme, width=600, height=600)

        return fig

    def create_comprehensive_dashboard(
        self,
        molecules: List[GeneratedMolecule],
        reference_molecules: Optional[List[GeneratedMolecule]] = None,
        properties: Optional[List[str]] = None,
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive dashboard for molecular generation analysis.

        Args:
            molecules: List of generated molecules
            reference_molecules: Optional reference molecules
            properties: List of properties to analyze

        Returns:
            Dictionary of figure objects
        """
        dashboard = {}

        # Molecule gallery
        dashboard["gallery"] = self.create_molecule_gallery(molecules)

        # Property distributions
        dashboard["properties"] = self.plot_property_distributions(
            molecules, reference_molecules, properties
        )

        # Chemical space
        if properties and len(properties) >= 2:
            dashboard["chemical_space"] = self.plot_chemical_space_2d(
                molecules, reference_molecules, properties[0], properties[1]
            )

        # Generation metrics
        dashboard["metrics"] = self.plot_generation_metrics(molecules)

        # Scaffold analysis
        dashboard["scaffolds"] = self.plot_scaffold_analysis(molecules)

        # Similarity heatmap
        dashboard["similarity"] = self.create_tanimoto_similarity_heatmap(molecules)

        return dashboard

    def export_dashboard_html(
        self,
        dashboard: Dict[str, go.Figure],
        filename: str = "molecular_generation_dashboard.html",
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
        <title>Molecular Generation Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }
        .dashboard-container { max-width: 1200px; margin: 0 auto; }
        .plot-container {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; text-align: center; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #3498db;
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        </style>
        </head>
        <body>
        <div class="dashboard-container">
        <h1>Molecular Generation Analysis Dashboard</h1>
        """

        for plot_name, fig in dashboard.items():
            plot_title = plot_name.replace("_", " ").title()
            html_content += f'<div class="plot-container"><h2>{plot_title}</h2>'
            html_content += fig.to_html(include_plotlyjs=False, div_id=plot_name)
            html_content += "</div>"

        html_content += """
        </div>
        </body>
        </html>
        """

        with open(filename, "w") as f:
            f.write(html_content)

        print(f"Dashboard exported to {filename}")

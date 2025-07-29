"""
Interactive Route Visualization and Analysis.

This module develops interactive tools to visualize proposed synthesis routes,
displaying predicted properties for each step and allowing users to compare
trade-offs between different routes.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import FancyBboxPatch

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    plt = None
    patches = None
    sns = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None
    pyo = None

try:
    import networkx as nx

    _NETWORKX_AVAILABLE = True
except ImportError:
    _NETWORKX_AVAILABLE = False
    nx = None

try:
    import streamlit as st

    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False
    st = None

from .forward_prediction import ForwardPredictionResult, RouteEvaluationResult
from .multi_objective_optimization import (
    OptimizationResult,
    OptimizationSolution,
    ParetoFront,
)
from .route_enumeration import SynthesisNode, SynthesisRoute, SynthesisTree

# Set up logging
logger = logging.getLogger(__name__)


# Enums
class VisualizationType(Enum):
    """Types of visualization."""

    ROUTE_TREE = "route_tree"
    STEP_BY_STEP = "step_by_step"
    PARETO_FRONT = "pareto_front"
    OBJECTIVE_COMPARISON = "objective_comparison"
    REACTION_CONDITIONS = "reaction_conditions"
    YIELD_ANALYSIS = "yield_analysis"
    COST_BREAKDOWN = "cost_breakdown"
    TIMELINE_VIEW = "timeline_view"


class ColorScheme(Enum):
    """Color schemes for visualization."""

    DEFAULT = "default"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    COOLWARM = "coolwarm"
    SPECTRAL = "spectral"
    CUSTOM = "custom"


class LayoutAlgorithm(Enum):
    """Layout algorithms for graph visualization."""

    HIERARCHICAL = "hierarchical"
    FORCE_DIRECTED = "force_directed"
    CIRCULAR = "circular"
    TREE = "tree"
    GRID = "grid"


class InteractionMode(Enum):
    """Interaction modes for visualization."""

    VIEW_ONLY = "view_only"
    HOVER_INFO = "hover_info"
    CLICK_DETAILS = "click_details"
    DRAG_AND_DROP = "drag_and_drop"
    EDIT_MODE = "edit_mode"


# Data Structures
@dataclass
class VisualizationConfig:
    """Configuration for route visualization."""

    visualization_type: VisualizationType = VisualizationType.ROUTE_TREE
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    layout_algorithm: LayoutAlgorithm = LayoutAlgorithm.HIERARCHICAL
    interaction_mode: InteractionMode = InteractionMode.HOVER_INFO
    show_molecular_structures: bool = True
    show_reaction_conditions: bool = True
    show_predicted_yields: bool = True
    show_confidence_scores: bool = True
    show_cost_estimates: bool = True
    figure_width: int = 1200
    figure_height: int = 800
    node_size_factor: float = 1.0
    edge_width_factor: float = 1.0
    font_size: int = 10
    export_format: str = "html"  # html, png, svg, pdf
    theme: str = "plotly_white"


@dataclass
class VisualizationElement:
    """Base class for visualization elements."""

    element_id: str
    element_type: str
    position: Tuple[float, float] = (0.0, 0.0)
    properties: Dict[str, Any] = field(default_factory=dict)
    styling: Dict[str, Any] = field(default_factory=dict)
    interactivity: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MoleculeNode(VisualizationElement):
    """Visualization element for molecules."""

    smiles: str = ""
    molecule_image: Optional[str] = None
    molecular_properties: Dict[str, float] = field(default_factory=dict)
    is_purchasable: bool = False
    cost_estimate: float = 0.0
    complexity_score: float = 0.0

    def __post_init__(self):
        if not self.element_id:
            self.element_id = f"mol_{abs(hash(self.smiles))}"
        self.element_type = "molecule"


@dataclass
class ReactionEdge(VisualizationElement):
    """Visualization element for reactions."""

    source_node: str = ""
    target_node: str = ""
    reaction_type: str = ""
    predicted_yield: float = 0.0
    confidence: float = 0.0
    reaction_conditions: Dict[str, Any] = field(default_factory=dict)
    byproducts: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.element_id:
            self.element_id = f"rxn_{abs(hash(self.source_node + self.target_node))}"
        self.element_type = "reaction"


@dataclass
class VisualizationData:
    """Complete visualization data structure."""

    nodes: List[MoleculeNode] = field(default_factory=list)
    edges: List[ReactionEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    layout_data: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: MoleculeNode):
        """Add molecule node."""
        self.nodes.append(node)

    def add_edge(self, edge: ReactionEdge):
        """Add reaction edge."""
        self.edges.append(edge)

    def get_node_by_id(self, node_id: str) -> Optional[MoleculeNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.element_id == node_id:
                return node
        return None


# Exceptions
class VisualizationError(Exception):
    """Base exception for visualization errors."""

    pass


class RenderingError(VisualizationError):
    """Error in rendering visualization."""

    pass


class LayoutError(VisualizationError):
    """Error in layout calculation."""

    pass


class InteractionError(VisualizationError):
    """Error in user interaction handling."""

    pass


# Main Classes
class MoleculeRenderer:
    """Renders molecular structures for visualization."""

    def __init__(self):
        self.image_cache: Dict[str, str] = {}

    def render_molecule(
        self,
        smiles: str,
        size: Tuple[int, int] = (200, 200),
        highlight_atoms: Optional[List[int]] = None,
    ) -> Optional[str]:
        """Render molecule as image."""
        try:
            # Check cache
            cache_key = f"{smiles}_{size[0]}x{size[1]}"
            if cache_key in self.image_cache:
                return self.image_cache[cache_key]

            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None

            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)

            # Create image
            img = Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms or [])

            # Convert to base64 string for web display
            import base64
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            img_data = f"data:image/png;base64,{img_str}"

            # Cache result
            self.image_cache[cache_key] = img_data

            return img_data

        except Exception as e:
            logger.error(f"Error rendering molecule {smiles}: {e}")
            return None

    def render_reaction(
        self,
        reactants: List[str],
        products: List[str],
        size: Tuple[int, int] = (400, 200),
    ) -> Optional[str]:
        """Render reaction scheme."""
        try:
            # Create reaction string
            reactant_str = ".".join(reactants)
            product_str = ".".join(products)
            reaction_smiles = f"{reactant_str}>>{product_str}"

            # Parse reaction
            rxn = AllChem.ReactionFromSmarts(reaction_smiles)
            if not rxn:
                return None

            # Render reaction
            img = Draw.ReactionToImage(rxn, subImgSize=size)

            # Convert to base64
            import base64
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            img_data = f"data:image/png;base64,{img_str}"

            return img_data

        except Exception as e:
            logger.error(f"Error rendering reaction: {e}")
            return None


class LayoutCalculator:
    """Calculates layouts for route visualization."""

    def __init__(self):
        pass

    def calculate_layout(
        self,
        vis_data: VisualizationData,
        algorithm: LayoutAlgorithm = LayoutAlgorithm.HIERARCHICAL,
        **kwargs,
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using specified algorithm."""
        try:
            if algorithm == LayoutAlgorithm.HIERARCHICAL:
                return self._calculate_hierarchical_layout(vis_data, **kwargs)
            elif algorithm == LayoutAlgorithm.FORCE_DIRECTED:
                return self._calculate_force_directed_layout(vis_data, **kwargs)
            elif algorithm == LayoutAlgorithm.CIRCULAR:
                return self._calculate_circular_layout(vis_data, **kwargs)
            elif algorithm == LayoutAlgorithm.TREE:
                return self._calculate_tree_layout(vis_data, **kwargs)
            elif algorithm == LayoutAlgorithm.GRID:
                return self._calculate_grid_layout(vis_data, **kwargs)
            else:
                return self._calculate_hierarchical_layout(vis_data, **kwargs)

        except Exception as e:
            logger.error(f"Error calculating layout: {e}")
            raise LayoutError(f"Layout calculation failed: {e}")

    def _calculate_hierarchical_layout(
        self, vis_data: VisualizationData, **kwargs
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate hierarchical layout based on synthesis depth."""
        positions = {}

        # Group nodes by synthesis depth (if available)
        depth_groups = {}
        for node in vis_data.nodes:
            depth = node.properties.get("depth", 0)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node)

        # Calculate positions
        max_depth = max(depth_groups.keys()) if depth_groups else 0
        y_spacing = 100
        x_spacing = 150

        for depth, nodes in depth_groups.items():
            y = (max_depth - depth) * y_spacing

            # Center nodes horizontally
            total_width = (len(nodes) - 1) * x_spacing
            start_x = -total_width / 2

            for i, node in enumerate(nodes):
                x = start_x + i * x_spacing
                positions[node.element_id] = (x, y)

        return positions

    def _calculate_force_directed_layout(
        self, vis_data: VisualizationData, **kwargs
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate force-directed layout using NetworkX."""
        if not _NETWORKX_AVAILABLE:
            return self._calculate_hierarchical_layout(vis_data, **kwargs)

        try:
            # Create NetworkX graph
            G = nx.DiGraph()

            for node in vis_data.nodes:
                G.add_node(node.element_id)

            for edge in vis_data.edges:
                G.add_edge(edge.source_node, edge.target_node)

            # Calculate spring layout
            pos = nx.spring_layout(
                G,
                k=kwargs.get("k", 1),
                iterations=kwargs.get("iterations", 50),
                scale=kwargs.get("scale", 1000),
            )

            # Convert to required format
            positions = {}
            for node_id, (x, y) in pos.items():
                positions[node_id] = (float(x), float(y))

            return positions

        except Exception as e:
            logger.error(f"Error in force-directed layout: {e}")
            return self._calculate_hierarchical_layout(vis_data, **kwargs)

    def _calculate_circular_layout(
        self, vis_data: VisualizationData, **kwargs
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate circular layout."""
        positions = {}
        n_nodes = len(vis_data.nodes)

        if n_nodes == 0:
            return positions

        radius = kwargs.get("radius", 200)
        center = kwargs.get("center", (0, 0))

        for i, node in enumerate(vis_data.nodes):
            angle = 2 * np.pi * i / n_nodes
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            positions[node.element_id] = (x, y)

        return positions

    def _calculate_tree_layout(
        self, vis_data: VisualizationData, **kwargs
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate tree layout."""
        if not _NETWORKX_AVAILABLE:
            return self._calculate_hierarchical_layout(vis_data, **kwargs)

        try:
            # Create NetworkX graph
            G = nx.DiGraph()

            for node in vis_data.nodes:
                G.add_node(node.element_id)

            for edge in vis_data.edges:
                G.add_edge(edge.source_node, edge.target_node)

            # Find root node (node with no incoming edges)
            root = None
            for node_id in G.nodes():
                if G.in_degree(node_id) == 0:
                    root = node_id
                    break

            if root:
                # Calculate hierarchical layout
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot", root=root)
            else:
                pos = nx.spring_layout(G)

            positions = {}
            for node_id, (x, y) in pos.items():
                positions[node_id] = (float(x), float(y))

            return positions

        except Exception as e:
            logger.error(f"Error in tree layout: {e}")
            return self._calculate_hierarchical_layout(vis_data, **kwargs)

    def _calculate_grid_layout(
        self, vis_data: VisualizationData, **kwargs
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate grid layout."""
        positions = {}
        n_nodes = len(vis_data.nodes)

        if n_nodes == 0:
            return positions

        # Calculate grid dimensions
        cols = kwargs.get("cols", int(np.ceil(np.sqrt(n_nodes))))
        rows = int(np.ceil(n_nodes / cols))

        spacing_x = kwargs.get("spacing_x", 150)
        spacing_y = kwargs.get("spacing_y", 100)

        for i, node in enumerate(vis_data.nodes):
            row = i // cols
            col = i % cols

            x = col * spacing_x
            y = row * spacing_y

            positions[node.element_id] = (x, y)

        return positions


class InteractiveVisualizer:
    """Creates interactive visualizations for synthesis routes."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.molecule_renderer = MoleculeRenderer()
        self.layout_calculator = LayoutCalculator()

    def visualize_synthesis_route(
        self,
        route: SynthesisRoute,
        evaluation_result: Optional[RouteEvaluationResult] = None,
    ) -> str:
        """Create interactive visualization of synthesis route."""
        try:
            # Convert route to visualization data
            vis_data = self._route_to_visualization_data(route, evaluation_result)

            # Calculate layout
            positions = self.layout_calculator.calculate_layout(
                vis_data, self.config.layout_algorithm
            )

            # Create visualization
            if _PLOTLY_AVAILABLE:
                return self._create_plotly_route_visualization(vis_data, positions)
            elif _MATPLOTLIB_AVAILABLE:
                return self._create_matplotlib_route_visualization(vis_data, positions)
            else:
                return self._create_text_visualization(vis_data)

        except Exception as e:
            logger.error(f"Error visualizing synthesis route: {e}")
            raise RenderingError(f"Route visualization failed: {e}")

    def visualize_pareto_front(
        self,
        optimization_result: OptimizationResult,
        x_objective: str = "total_cost",
        y_objective: str = "overall_yield",
    ) -> str:
        """Create interactive Pareto front visualization."""
        try:
            if not _PLOTLY_AVAILABLE:
                return self._create_text_pareto_visualization(optimization_result)

            pareto_solutions = optimization_result.get_pareto_optimal_solutions()

            if not pareto_solutions:
                return "<div>No Pareto optimal solutions found.</div>"

            # Extract objective values
            x_values = []
            y_values = []
            hover_texts = []

            for solution in pareto_solutions:
                x_val = solution.objective_values.get(x_objective, 0.0)
                y_val = solution.objective_values.get(y_objective, 0.0)

                x_values.append(x_val)
                y_values.append(y_val)

                # Create hover text
                hover_text = f"Route: {solution.route.route_id}<br>"
                hover_text += f"Steps: {solution.route.total_steps}<br>"
                hover_text += f"Cost: ${x_val:.2f}<br>"
                hover_text += f"Yield: {y_val:.1%}"
                hover_texts.append(hover_text)

            # Create scatter plot
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="circle"),
                    text=hover_texts,
                    hovertemplate="%{text}<extra></extra>",
                    name="Pareto Optimal Routes",
                )
            )

            fig.update_layout(
                title="Pareto Front: Route Optimization",
                xaxis_title=x_objective.replace("_", " ").title(),
                yaxis_title=y_objective.replace("_", " ").title(),
                width=self.config.figure_width,
                height=self.config.figure_height,
                template=self.config.theme,
            )

            return fig.to_html(include_plotlyjs="cdn")

        except Exception as e:
            logger.error(f"Error visualizing Pareto front: {e}")
            raise RenderingError(f"Pareto front visualization failed: {e}")

    def compare_routes(
        self,
        routes: List[SynthesisRoute],
        evaluation_results: Optional[List[RouteEvaluationResult]] = None,
    ) -> str:
        """Create comparative visualization of multiple routes."""
        try:
            if not _PLOTLY_AVAILABLE:
                return self._create_text_route_comparison(routes)

            # Create subplot layout
            n_routes = len(routes)
            cols = min(3, n_routes)
            rows = int(np.ceil(n_routes / cols))

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f"Route {i+1}" for i in range(n_routes)],
                specs=[[{"type": "xy"} for _ in range(cols)] for _ in range(rows)],
            )

            for i, route in enumerate(routes):
                row = i // cols + 1
                col = i % cols + 1

                # Create route metrics
                metrics = self._extract_route_metrics(
                    route, evaluation_results[i] if evaluation_results else None
                )

                # Add bar chart for this route
                fig.add_trace(
                    go.Bar(
                        x=list(metrics.keys()),
                        y=list(metrics.values()),
                        name=f"Route {i+1}",
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=col,
                )

            fig.update_layout(
                title="Route Comparison",
                width=self.config.figure_width,
                height=self.config.figure_height,
                template=self.config.theme,
            )

            return fig.to_html(include_plotlyjs="cdn")

        except Exception as e:
            logger.error(f"Error comparing routes: {e}")
            raise RenderingError(f"Route comparison failed: {e}")

    def _route_to_visualization_data(
        self, route: SynthesisRoute, evaluation_result: Optional[RouteEvaluationResult]
    ) -> VisualizationData:
        """Convert synthesis route to visualization data."""
        vis_data = VisualizationData()

        # Add molecule nodes
        for i, node in enumerate(route.route_nodes):
            mol_node = MoleculeNode(
                smiles=node.smiles,
                is_purchasable=node.is_purchasable,
                cost_estimate=node.cost_estimate,
                complexity_score=node.complexity_score,
            )

            # Add molecular properties
            mol_node.molecular_properties = {
                "depth": node.depth,
                "confidence": node.confidence,
            }

            # Render molecule if requested
            if self.config.show_molecular_structures:
                mol_node.molecule_image = self.molecule_renderer.render_molecule(
                    node.smiles
                )

            vis_data.add_node(mol_node)

        # Add reaction edges
        for i in range(len(route.route_nodes) - 1):
            source_node = route.route_nodes[i]
            target_node = route.route_nodes[i + 1]

            edge = ReactionEdge(
                source_node=f"mol_{abs(hash(source_node.smiles))}",
                target_node=f"mol_{abs(hash(target_node.smiles))}",
            )

            # Add prediction data if available
            if evaluation_result and i < len(evaluation_result.step_predictions):
                step_pred = evaluation_result.step_predictions[i]
                edge.predicted_yield = step_pred.main_product_yield
                edge.confidence = step_pred.confidence
                edge.reaction_conditions = {
                    "temperature": step_pred.optimal_conditions.temperature
                    if step_pred.optimal_conditions
                    else 298.15,
                    "catalyst": step_pred.optimal_conditions.catalyst
                    if step_pred.optimal_conditions
                    else None,
                }

            vis_data.add_edge(edge)

        # Add metadata
        vis_data.metadata = {
            "route_id": route.route_id,
            "target_smiles": route.target_smiles,
            "total_steps": route.total_steps,
            "overall_confidence": route.overall_confidence,
            "total_cost": route.total_cost,
            "complexity": route.complexity.value
            if hasattr(route.complexity, "value")
            else str(route.complexity),
        }

        return vis_data

    def _create_plotly_route_visualization(
        self, vis_data: VisualizationData, positions: Dict[str, Tuple[float, float]]
    ) -> str:
        """Create Plotly-based route visualization."""
        fig = go.Figure()

        # Add edges first (so they appear behind nodes)
        edge_x = []
        edge_y = []
        edge_info = []

        for edge in vis_data.edges:
            source_pos = positions.get(edge.source_node, (0, 0))
            target_pos = positions.get(edge.target_node, (0, 0))

            edge_x.extend([source_pos[0], target_pos[0], None])
            edge_y.extend([source_pos[1], target_pos[1], None])

            edge_info.append(
                {"yield": edge.predicted_yield, "confidence": edge.confidence}
            )

        # Add edge trace
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=2, color="gray"),
                hoverinfo="none",
                mode="lines",
                name="Reactions",
            )
        )

        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        hover_texts = []

        for node in vis_data.nodes:
            pos = positions.get(node.element_id, (0, 0))
            node_x.append(pos[0])
            node_y.append(pos[1])

            # Node text
            if self.config.show_molecular_structures and node.molecule_image:
                node_text.append("")  # Will use images
            else:
                node_text.append(node.smiles)

            # Node color based on properties
            if node.is_purchasable:
                node_colors.append("green")
            else:
                node_colors.append("lightblue")

            # Node size based on complexity
            base_size = 20 * self.config.node_size_factor
            size_factor = 1 + node.complexity_score * 0.1
            node_sizes.append(base_size * size_factor)

            # Hover text
            hover_text = f"SMILES: {node.smiles}<br>"
            hover_text += f"Purchasable: {'Yes' if node.is_purchasable else 'No'}<br>"
            hover_text += f"Cost: ${node.cost_estimate:.2f}<br>"
            hover_text += f"Complexity: {node.complexity_score:.2f}"
            hover_texts.append(hover_text)

        # Add node trace
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color="black"),
                ),
                text=node_text,
                textposition="middle center",
                hovertext=hover_texts,
                hovertemplate="%{hovertext}<extra></extra>",
                name="Molecules",
            )
        )

        # Update layout
        fig.update_layout(
            title=f"Synthesis Route: {vis_data.metadata.get('route_id', 'Unknown')}",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text=f"Target: {vis_data.metadata.get('target_smiles', '')}<br>"
                    + f"Steps: {vis_data.metadata.get('total_steps', 0)}<br>"
                    + f"Cost: ${vis_data.metadata.get('total_cost', 0):.2f}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=0.995,
                    xanchor="left",
                    yanchor="top",
                    bordercolor="black",
                    borderwidth=1,
                    bgcolor="white",
                    font=dict(size=10),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=self.config.figure_width,
            height=self.config.figure_height,
            template=self.config.theme,
        )

        return fig.to_html(include_plotlyjs="cdn")

    def _create_matplotlib_route_visualization(
        self, vis_data: VisualizationData, positions: Dict[str, Tuple[float, float]]
    ) -> str:
        """Create Matplotlib-based route visualization."""
        if not _MATPLOTLIB_AVAILABLE:
            return self._create_text_visualization(vis_data)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw edges
        for edge in vis_data.edges:
            source_pos = positions.get(edge.source_node, (0, 0))
            target_pos = positions.get(edge.target_node, (0, 0))

            ax.arrow(
                source_pos[0],
                source_pos[1],
                target_pos[0] - source_pos[0],
                target_pos[1] - source_pos[1],
                head_width=10,
                head_length=15,
                fc="gray",
                ec="gray",
                alpha=0.7,
            )

        # Draw nodes
        for node in vis_data.nodes:
            pos = positions.get(node.element_id, (0, 0))

            color = "green" if node.is_purchasable else "lightblue"
            size = 300 + node.complexity_score * 50

            ax.scatter(pos[0], pos[1], s=size, c=color, alpha=0.8, edgecolors="black")
            ax.text(pos[0], pos[1], node.smiles, ha="center", va="center", fontsize=8)

        ax.set_title(f"Synthesis Route: {vis_data.metadata.get('route_id', 'Unknown')}")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

        # Save to string
        import base64
        import io

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        img_str = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        html = f'<img src="data:image/png;base64,{img_str}" alt="Route Visualization"/>'
        return html

    def _create_text_visualization(self, vis_data: VisualizationData) -> str:
        """Create text-based visualization as fallback."""
        html = "<div style='font-family: monospace; padding: 20px;'>"
        html += (
            f"<h3>Synthesis Route: {vis_data.metadata.get('route_id', 'Unknown')}</h3>"
        )
        html += f"<p><strong>Target:</strong> {vis_data.metadata.get('target_smiles', '')}</p>"
        html += (
            f"<p><strong>Steps:</strong> {vis_data.metadata.get('total_steps', 0)}</p>"
        )
        html += f"<p><strong>Total Cost:</strong> ${vis_data.metadata.get('total_cost', 0):.2f}</p>"
        html += "<h4>Molecules:</h4>"
        html += "<ul>"

        for node in vis_data.nodes:
            html += f"<li>{node.smiles} "
            if node.is_purchasable:
                html += "(Purchasable)"
            html += f" - Cost: ${node.cost_estimate:.2f}</li>"

        html += "</ul></div>"
        return html

    def _create_text_pareto_visualization(
        self, optimization_result: OptimizationResult
    ) -> str:
        """Create text-based Pareto front visualization."""
        html = "<div style='font-family: monospace; padding: 20px;'>"
        html += "<h3>Pareto Front Analysis</h3>"

        pareto_solutions = optimization_result.get_pareto_optimal_solutions()

        if not pareto_solutions:
            html += "<p>No Pareto optimal solutions found.</p>"
        else:
            html += f"<p>Found {len(pareto_solutions)} Pareto optimal solutions:</p>"
            html += "<table border='1' style='border-collapse: collapse;'>"
            html += "<tr><th>Route</th><th>Steps</th><th>Cost</th><th>Yield</th></tr>"

            for i, solution in enumerate(pareto_solutions):
                html += "<tr>"
                html += f"<td>Route {i+1}</td>"
                html += f"<td>{solution.route.total_steps}</td>"
                html += f"<td>${solution.route.total_cost:.2f}</td>"
                html += f"<td>{solution.route.overall_confidence:.1%}</td>"
                html += "</tr>"

            html += "</table>"

        html += "</div>"
        return html

    def _create_text_route_comparison(self, routes: List[SynthesisRoute]) -> str:
        """Create text-based route comparison."""
        html = "<div style='font-family: monospace; padding: 20px;'>"
        html += "<h3>Route Comparison</h3>"
        html += "<table border='1' style='border-collapse: collapse;'>"
        html += "<tr><th>Route</th><th>Target</th><th>Steps</th><th>Cost</th><th>Confidence</th></tr>"

        for i, route in enumerate(routes):
            html += "<tr>"
            html += f"<td>Route {i+1}</td>"
            html += f"<td>{route.target_smiles}</td>"
            html += f"<td>{route.total_steps}</td>"
            html += f"<td>${route.total_cost:.2f}</td>"
            html += f"<td>{route.overall_confidence:.1%}</td>"
            html += "</tr>"

        html += "</table></div>"
        return html

    def _extract_route_metrics(
        self, route: SynthesisRoute, evaluation_result: Optional[RouteEvaluationResult]
    ) -> Dict[str, float]:
        """Extract metrics for comparison."""
        metrics = {
            "Steps": float(route.total_steps),
            "Cost": route.total_cost,
            "Confidence": route.overall_confidence,
        }

        if evaluation_result:
            metrics["Overall Yield"] = evaluation_result.overall_yield
            metrics["Selectivity"] = evaluation_result.overall_selectivity
            metrics["Feasibility"] = evaluation_result.feasibility_score

        return metrics


# Streamlit Dashboard (if available)
if _STREAMLIT_AVAILABLE:

    class StreamlitDashboard:
        """Streamlit-based interactive dashboard."""

        def __init__(self):
            self.visualizer = InteractiveVisualizer()

        def create_dashboard(self):
            """Create Streamlit dashboard."""
            st.set_page_config(
                page_title="ChemiAccel Route Visualizer", page_icon="🧪", layout="wide"
            )

            st.title("🧪 ChemiAccel Synthesis Route Visualizer")
            st.sidebar.title("Navigation")

            # Sidebar options
            page = st.sidebar.selectbox(
                "Choose a page",
                ["Route Visualization", "Pareto Front Analysis", "Route Comparison"],
            )

            if page == "Route Visualization":
                self._route_visualization_page()
            elif page == "Pareto Front Analysis":
                self._pareto_analysis_page()
            elif page == "Route Comparison":
                self._route_comparison_page()

        def _route_visualization_page(self):
            """Route visualization page."""
            st.header("Synthesis Route Visualization")

            # Input area
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Route Input")
                target_smiles = st.text_input("Target Molecule SMILES", "CCO")

                # Visualization options
                st.subheader("Visualization Options")
                show_structures = st.checkbox("Show Molecular Structures", True)
                show_yields = st.checkbox("Show Predicted Yields", True)
                show_costs = st.checkbox("Show Cost Estimates", True)

                layout_algorithm = st.selectbox(
                    "Layout Algorithm",
                    ["Hierarchical", "Force Directed", "Circular", "Tree"],
                )

            with col2:
                st.subheader("Route Visualization")
                # This would integrate with actual route data
                st.info("Connect your synthesis route data here")

        def _pareto_analysis_page(self):
            """Pareto front analysis page."""
            st.header("Pareto Front Analysis")
            st.info("Upload optimization results to visualize Pareto fronts")

        def _route_comparison_page(self):
            """Route comparison page."""
            st.header("Route Comparison")
            st.info("Compare multiple synthesis routes side by side")


# Utility Functions
def create_visualization_config(
    visualization_type: VisualizationType = VisualizationType.ROUTE_TREE, **kwargs
) -> VisualizationConfig:
    """Create visualization configuration."""
    return VisualizationConfig(visualization_type=visualization_type, **kwargs)


def visualize_synthesis_route_simple(
    route: SynthesisRoute, output_file: Optional[str] = None
) -> str:
    """Simple route visualization function."""
    visualizer = InteractiveVisualizer()
    html_content = visualizer.visualize_synthesis_route(route)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content


def create_pareto_front_plot(
    optimization_result: OptimizationResult, output_file: Optional[str] = None
) -> str:
    """Create Pareto front plot."""
    visualizer = InteractiveVisualizer()
    html_content = visualizer.visualize_pareto_front(optimization_result)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content


def launch_streamlit_dashboard():
    """Launch Streamlit dashboard."""
    if _STREAMLIT_AVAILABLE:
        dashboard = StreamlitDashboard()
        dashboard.create_dashboard()
    else:
        print("Streamlit not available. Install with: pip install streamlit")

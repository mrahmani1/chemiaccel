"""
Interactive visualization of mixture properties.

Creates interactive dashboards and 3D visualizations for:
- Phase diagrams
- Property-composition landscapes
- Molecular-level insights into component interactions
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.exceptions import ChemiAccelError
from .core import ComponentData, MixtureData
from .dynamic_processes import FormulationProcess
from .interactions import MolecularInteraction
from .phase_behavior import PhaseEquilibrium


@dataclass
class PropertyLandscape:
    """Data structure for property-composition landscapes."""

    composition_grid: np.ndarray
    property_grid: np.ndarray
    property_name: str
    components: List[str]
    metadata: Dict[str, Any]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the property grid."""
        return self.property_grid.shape

    @property
    def n_components(self) -> int:
        """Number of components."""
        return len(self.components)


class MixtureVisualizer:
    """
    Interactive visualization system for mixture properties.

    Provides tools for creating interactive dashboards, 3D visualizations,
    and molecular-level insight displays.
    """

    def __init__(self):
        self.plot_style = "seaborn"
        self.color_schemes = {
            "phase_diagram": "viridis",
            "interactions": "plasma",
            "properties": "coolwarm",
            "composition": "Set3",
        }

    def create_phase_diagram(
        self,
        phase_equilibrium: PhaseEquilibrium,
        diagram_type: str = "2d",
        interactive: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create phase diagram visualization.

        Args:
            phase_equilibrium: Phase equilibrium data
            diagram_type: Type of diagram ('2d', '3d', 'ternary')
            interactive: Whether to create interactive plot

        Returns:
            Visualization data and configuration
        """

        if not phase_equilibrium.phase_diagram_data:
            raise ChemiAccelError("No phase diagram data available")

        diagram_data = phase_equilibrium.phase_diagram_data

        if diagram_type == "2d":
            return self._create_2d_phase_diagram(diagram_data, interactive, **kwargs)
        elif diagram_type == "3d":
            return self._create_3d_phase_diagram(diagram_data, interactive, **kwargs)
        elif diagram_type == "ternary":
            return self._create_ternary_phase_diagram(
                diagram_data, interactive, **kwargs
            )
        else:
            raise ValueError(f"Unsupported diagram type: {diagram_type}")

    def _create_2d_phase_diagram(
        self, diagram_data: Dict[str, Any], interactive: bool, **kwargs
    ) -> Dict[str, Any]:
        """Create 2D phase diagram."""

        data_points = diagram_data["data_points"]

        # Extract temperature and pressure data
        temperatures = [point["temperature"] for point in data_points]
        pressures = [point["pressure"] for point in data_points]
        n_phases = [point["n_phases"] for point in data_points]

        # Create meshgrid for contour plot
        T_range = diagram_data.get(
            "temperature_range", (min(temperatures), max(temperatures))
        )
        P_range = diagram_data.get("pressure_range", (min(pressures), max(pressures)))

        T_grid = np.linspace(T_range[0], T_range[1], 50)
        P_grid = np.linspace(P_range[0], P_range[1], 50)
        T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)

        # Interpolate number of phases onto grid
        from scipy.interpolate import griddata

        phase_mesh = griddata(
            (temperatures, pressures),
            n_phases,
            (T_mesh, P_mesh),
            method="cubic",
            fill_value=1,
        )

        plot_config = {
            "type": "2d_phase_diagram",
            "data": {
                "T_mesh": T_mesh,
                "P_mesh": P_mesh,
                "phase_mesh": phase_mesh,
                "scatter_data": {
                    "temperatures": temperatures,
                    "pressures": pressures,
                    "n_phases": n_phases,
                },
            },
            "layout": {
                "title": "2D Phase Diagram",
                "xlabel": "Temperature (K)",
                "ylabel": "Pressure (Pa)",
                "colorbar_label": "Number of Phases",
            },
            "interactive": interactive,
            "color_scheme": self.color_schemes["phase_diagram"],
        }

        if interactive:
            plot_config["widgets"] = self._create_phase_diagram_widgets()

        return plot_config

    def _create_3d_phase_diagram(
        self, diagram_data: Dict[str, Any], interactive: bool, **kwargs
    ) -> Dict[str, Any]:
        """Create 3D phase diagram."""

        data_points = diagram_data["data_points"]

        # Extract data for 3D plot
        temperatures = [point["temperature"] for point in data_points]
        pressures = [point["pressure"] for point in data_points]
        n_phases = [point["n_phases"] for point in data_points]

        # Create 3D surface data (simplified)
        plot_config = {
            "type": "3d_phase_diagram",
            "data": {
                "temperatures": temperatures,
                "pressures": pressures,
                "n_phases": n_phases,
            },
            "layout": {
                "title": "3D Phase Diagram",
                "xlabel": "Temperature (K)",
                "ylabel": "Pressure (Pa)",
                "zlabel": "Number of Phases",
            },
            "interactive": interactive,
            "color_scheme": self.color_schemes["phase_diagram"],
        }

        return plot_config

    def _create_ternary_phase_diagram(
        self, diagram_data: Dict[str, Any], interactive: bool, **kwargs
    ) -> Dict[str, Any]:
        """Create ternary phase diagram for 3-component systems."""

        # This would be implemented for 3-component systems
        plot_config = {
            "type": "ternary_phase_diagram",
            "data": {},
            "layout": {
                "title": "Ternary Phase Diagram",
            },
            "interactive": interactive,
            "color_scheme": self.color_schemes["phase_diagram"],
        }

        return plot_config

    def create_property_landscape(
        self,
        mixture_data: List[MixtureData],
        property_name: str,
        resolution: int = 50,
        interactive: bool = True,
        **kwargs,
    ) -> PropertyLandscape:
        """
        Create property-composition landscape.

        Args:
            mixture_data: List of mixture compositions and properties
            property_name: Property to visualize
            resolution: Grid resolution
            interactive: Whether to create interactive plot

        Returns:
            PropertyLandscape object
        """

        if not mixture_data:
            raise ValueError("No mixture data provided")

        # Extract composition and property data
        compositions = []
        properties = []
        components = mixture_data[0].smiles_list

        for mixture in mixture_data:
            compositions.append(mixture.composition_vector)
            prop_value = mixture.mixture_properties.get(property_name, 0.0)
            properties.append(prop_value)

        compositions = np.array(compositions)
        properties = np.array(properties)

        # Create composition grid
        if len(components) == 2:
            # Binary system
            x1_grid = np.linspace(0, 1, resolution)
            x2_grid = 1 - x1_grid
            composition_grid = np.column_stack([x1_grid, x2_grid])

        elif len(components) == 3:
            # Ternary system
            composition_grid = self._create_ternary_grid(resolution)

        else:
            # Higher dimensional - use first 2 components
            x1_grid, x2_grid = np.meshgrid(
                np.linspace(0, 1, resolution), np.linspace(0, 1, resolution)
            )
            # Normalize to ensure sum <= 1
            mask = (x1_grid + x2_grid) <= 1
            composition_grid = np.column_stack([x1_grid[mask], x2_grid[mask]])

        # Interpolate properties onto grid
        from scipy.interpolate import griddata

        if compositions.shape[1] >= 2:
            property_grid = griddata(
                compositions[:, :2],
                properties,
                composition_grid[:, :2],
                method="cubic",
                fill_value=np.nan,
            )
        else:
            property_grid = np.full(len(composition_grid), np.nan)

        return PropertyLandscape(
            composition_grid=composition_grid,
            property_grid=property_grid,
            property_name=property_name,
            components=components,
            metadata={
                "resolution": resolution,
                "n_data_points": len(mixture_data),
                "property_range": (np.min(properties), np.max(properties)),
            },
        )

    def _create_ternary_grid(self, resolution: int) -> np.ndarray:
        """Create triangular grid for ternary diagrams."""

        compositions = []

        for i in range(resolution + 1):
            for j in range(resolution + 1 - i):
                x1 = i / resolution
                x2 = j / resolution
                x3 = 1 - x1 - x2

                if x3 >= 0:  # Valid composition
                    compositions.append([x1, x2, x3])

        return np.array(compositions)

    def visualize_property_landscape(
        self,
        landscape: PropertyLandscape,
        plot_type: str = "contour",
        interactive: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Visualize property landscape.

        Args:
            landscape: PropertyLandscape data
            plot_type: Type of plot ('contour', 'surface', 'heatmap')
            interactive: Whether to create interactive plot

        Returns:
            Visualization configuration
        """

        if plot_type == "contour":
            return self._create_contour_plot(landscape, interactive, **kwargs)
        elif plot_type == "surface":
            return self._create_surface_plot(landscape, interactive, **kwargs)
        elif plot_type == "heatmap":
            return self._create_heatmap_plot(landscape, interactive, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

    def _create_contour_plot(
        self, landscape: PropertyLandscape, interactive: bool, **kwargs
    ) -> Dict[str, Any]:
        """Create contour plot of property landscape."""

        if landscape.n_components == 2:
            # Binary system contour plot
            x1 = landscape.composition_grid[:, 0]
            x2 = landscape.composition_grid[:, 1]

            plot_config = {
                "type": "contour_plot",
                "data": {"x": x1, "y": x2, "z": landscape.property_grid},
                "layout": {
                    "title": f"{landscape.property_name} vs Composition",
                    "xlabel": f"{landscape.components[0]} Fraction",
                    "ylabel": f"{landscape.components[1]} Fraction",
                    "colorbar_label": landscape.property_name,
                },
                "interactive": interactive,
                "color_scheme": self.color_schemes["properties"],
            }

        elif landscape.n_components == 3:
            # Ternary system contour plot
            plot_config = {
                "type": "ternary_contour",
                "data": {
                    "compositions": landscape.composition_grid,
                    "property_values": landscape.property_grid,
                },
                "layout": {
                    "title": f"{landscape.property_name} Ternary Plot",
                    "vertex_labels": landscape.components,
                },
                "interactive": interactive,
                "color_scheme": self.color_schemes["properties"],
            }

        else:
            raise ValueError("Contour plots only supported for 2-3 components")

        return plot_config

    def _create_surface_plot(
        self, landscape: PropertyLandscape, interactive: bool, **kwargs
    ) -> Dict[str, Any]:
        """Create 3D surface plot of property landscape."""

        plot_config = {
            "type": "surface_plot",
            "data": {
                "composition_grid": landscape.composition_grid,
                "property_grid": landscape.property_grid,
            },
            "layout": {
                "title": f"{landscape.property_name} 3D Surface",
                "xlabel": "Composition 1",
                "ylabel": "Composition 2",
                "zlabel": landscape.property_name,
            },
            "interactive": interactive,
            "color_scheme": self.color_schemes["properties"],
        }

        return plot_config

    def _create_heatmap_plot(
        self, landscape: PropertyLandscape, interactive: bool, **kwargs
    ) -> Dict[str, Any]:
        """Create heatmap of property landscape."""

        plot_config = {
            "type": "heatmap",
            "data": {
                "composition_grid": landscape.composition_grid,
                "property_grid": landscape.property_grid,
            },
            "layout": {
                "title": f"{landscape.property_name} Heatmap",
                "xlabel": "Composition Axis 1",
                "ylabel": "Composition Axis 2",
                "colorbar_label": landscape.property_name,
            },
            "interactive": interactive,
            "color_scheme": self.color_schemes["properties"],
        }

        return plot_config

    def visualize_interactions(
        self,
        interactions: List[MolecularInteraction],
        mixture: MixtureData,
        visualization_type: str = "network",
        interactive: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Visualize molecular interactions.

        Args:
            interactions: List of molecular interactions
            mixture: Mixture data
            visualization_type: Type of visualization
            interactive: Whether to create interactive plot

        Returns:
            Visualization configuration
        """

        if visualization_type == "network":
            return self._create_interaction_network(
                interactions, mixture, interactive, **kwargs
            )
        elif visualization_type == "matrix":
            return self._create_interaction_matrix(
                interactions, mixture, interactive, **kwargs
            )
        elif visualization_type == "energy_plot":
            return self._create_energy_plot(interactions, interactive, **kwargs)
        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")

    def _create_interaction_network(
        self,
        interactions: List[MolecularInteraction],
        mixture: MixtureData,
        interactive: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create network visualization of interactions."""

        # Build network data
        nodes = []
        for i, comp in enumerate(mixture.components):
            nodes.append(
                {
                    "id": i,
                    "label": comp.smiles,
                    "size": comp.mol_fraction * 100,
                    "color": f"component_{i}",
                }
            )

        edges = []
        for interaction in interactions:
            edges.append(
                {
                    "source": interaction.component_i,
                    "target": interaction.component_j,
                    "weight": interaction.strength,
                    "type": interaction.interaction_type,
                    "energy": interaction.energy,
                    "color": f"interaction_{interaction.interaction_type}",
                }
            )

        plot_config = {
            "type": "network_plot",
            "data": {"nodes": nodes, "edges": edges},
            "layout": {
                "title": "Molecular Interaction Network",
                "layout_algorithm": "force_directed",
            },
            "interactive": interactive,
            "color_scheme": self.color_schemes["interactions"],
        }

        if interactive:
            plot_config["widgets"] = {
                "interaction_filter": {
                    "type": "dropdown",
                    "options": list(set(e["type"] for e in edges)),
                    "label": "Filter by interaction type",
                },
                "energy_threshold": {
                    "type": "slider",
                    "min": min(e["energy"] for e in edges),
                    "max": max(e["energy"] for e in edges),
                    "label": "Energy threshold",
                },
            }

        return plot_config

    def _create_interaction_matrix(
        self,
        interactions: List[MolecularInteraction],
        mixture: MixtureData,
        interactive: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create interaction matrix heatmap."""

        n_comp = mixture.n_components

        # Build interaction matrices for each type
        interaction_types = list(set(i.interaction_type for i in interactions))
        matrices = {}

        for itype in interaction_types:
            matrix = np.zeros((n_comp, n_comp))
            for interaction in interactions:
                if interaction.interaction_type == itype:
                    i, j = interaction.component_i, interaction.component_j
                    matrix[i, j] = interaction.energy
                    if i != j:
                        matrix[j, i] = interaction.energy
            matrices[itype] = matrix

        plot_config = {
            "type": "interaction_matrix",
            "data": {"matrices": matrices, "component_labels": mixture.smiles_list},
            "layout": {
                "title": "Interaction Energy Matrix",
                "xlabel": "Component",
                "ylabel": "Component",
            },
            "interactive": interactive,
            "color_scheme": self.color_schemes["interactions"],
        }

        if interactive:
            plot_config["widgets"] = {
                "interaction_type": {
                    "type": "tabs",
                    "options": interaction_types,
                    "label": "Interaction Type",
                }
            }

        return plot_config

    def _create_energy_plot(
        self, interactions: List[MolecularInteraction], interactive: bool, **kwargs
    ) -> Dict[str, Any]:
        """Create energy distribution plot."""

        # Group interactions by type
        by_type = {}
        for interaction in interactions:
            itype = interaction.interaction_type
            if itype not in by_type:
                by_type[itype] = []
            by_type[itype].append(interaction.energy)

        plot_config = {
            "type": "energy_distribution",
            "data": {"interaction_energies": by_type},
            "layout": {
                "title": "Interaction Energy Distribution",
                "xlabel": "Energy (kJ/mol)",
                "ylabel": "Density",
            },
            "interactive": interactive,
            "color_scheme": self.color_schemes["interactions"],
        }

        return plot_config

    def create_process_animation(
        self,
        process_results: FormulationProcess,
        property_names: List[str],
        animation_type: str = "time_series",
        interactive: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create animation of dynamic processes.

        Args:
            process_results: Results from process simulation
            property_names: Properties to animate
            animation_type: Type of animation
            interactive: Whether to create interactive animation

        Returns:
            Animation configuration
        """

        if animation_type == "time_series":
            return self._create_time_series_animation(
                process_results, property_names, interactive, **kwargs
            )
        elif animation_type == "composition_evolution":
            return self._create_composition_animation(
                process_results, interactive, **kwargs
            )
        elif animation_type == "microstructure_evolution":
            return self._create_microstructure_animation(
                process_results, interactive, **kwargs
            )
        else:
            raise ValueError(f"Unsupported animation type: {animation_type}")

    def _create_time_series_animation(
        self,
        process_results: FormulationProcess,
        property_names: List[str],
        interactive: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create time series animation."""

        time_data = []
        for i, (t, props) in enumerate(
            zip(process_results.time_points, process_results.property_evolution)
        ):
            frame_data = {"time": t}
            for prop in property_names:
                frame_data[prop] = props.get(prop, 0.0)
            time_data.append(frame_data)

        plot_config = {
            "type": "time_series_animation",
            "data": {"time_series": time_data, "properties": property_names},
            "layout": {
                "title": f"{process_results.process_type} Evolution",
                "xlabel": "Time (s)",
                "ylabel": "Property Value",
            },
            "interactive": interactive,
            "animation": {"frame_duration": 100, "loop": True},  # ms
        }

        if interactive:
            plot_config["widgets"] = {
                "play_pause": {"type": "button", "label": "Play/Pause"},
                "speed_control": {
                    "type": "slider",
                    "min": 0.1,
                    "max": 5.0,
                    "value": 1.0,
                    "label": "Animation Speed",
                },
            }

        return plot_config

    def _create_composition_animation(
        self, process_results: FormulationProcess, interactive: bool, **kwargs
    ) -> Dict[str, Any]:
        """Create composition evolution animation."""

        composition_data = []
        for i, mixture in enumerate(process_results.mixture_evolution):
            frame_data = {
                "time": process_results.time_points[i],
                "composition": mixture.composition_vector,
                "components": mixture.smiles_list,
            }
            composition_data.append(frame_data)

        plot_config = {
            "type": "composition_animation",
            "data": {"composition_evolution": composition_data},
            "layout": {
                "title": "Composition Evolution",
                "chart_type": "pie",  # or 'bar', 'ternary'
            },
            "interactive": interactive,
            "animation": {"frame_duration": 200, "loop": True},
        }

        return plot_config

    def _create_microstructure_animation(
        self, process_results: FormulationProcess, interactive: bool, **kwargs
    ) -> Dict[str, Any]:
        """Create microstructure evolution animation."""

        microstructure_data = []
        for i, microstructure in enumerate(process_results.microstructure_evolution):
            frame_data = {
                "time": process_results.time_points[i],
                "microstructure": microstructure,
            }
            microstructure_data.append(frame_data)

        plot_config = {
            "type": "microstructure_animation",
            "data": {"microstructure_evolution": microstructure_data},
            "layout": {"title": "Microstructure Evolution"},
            "interactive": interactive,
            "animation": {"frame_duration": 300, "loop": True},
        }

        return plot_config

    def create_dashboard(
        self,
        mixture: MixtureData,
        phase_equilibrium: Optional[PhaseEquilibrium] = None,
        interactions: Optional[List[MolecularInteraction]] = None,
        process_results: Optional[FormulationProcess] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create comprehensive interactive dashboard.

        Args:
            mixture: Mixture data
            phase_equilibrium: Phase behavior data
            interactions: Molecular interactions
            process_results: Process simulation results

        Returns:
            Dashboard configuration
        """

        dashboard_panels = []

        # Mixture composition panel
        composition_panel = {
            "type": "composition_panel",
            "title": "Mixture Composition",
            "data": {
                "components": [comp.smiles for comp in mixture.components],
                "fractions": [comp.mol_fraction for comp in mixture.components],
            },
            "visualization": "pie_chart",
        }
        dashboard_panels.append(composition_panel)

        # Phase diagram panel
        if phase_equilibrium:
            phase_panel = {
                "type": "phase_panel",
                "title": "Phase Behavior",
                "data": self.create_phase_diagram(phase_equilibrium, interactive=True),
                "visualization": "phase_diagram",
            }
            dashboard_panels.append(phase_panel)

        # Interactions panel
        if interactions:
            interactions_panel = {
                "type": "interactions_panel",
                "title": "Molecular Interactions",
                "data": self.visualize_interactions(
                    interactions, mixture, interactive=True
                ),
                "visualization": "network",
            }
            dashboard_panels.append(interactions_panel)

        # Process evolution panel
        if process_results:
            process_panel = {
                "type": "process_panel",
                "title": "Process Evolution",
                "data": self.create_process_animation(
                    process_results, ["temperature", "pressure"], interactive=True
                ),
                "visualization": "animation",
            }
            dashboard_panels.append(process_panel)

        # Properties summary panel
        properties_panel = {
            "type": "properties_panel",
            "title": "Properties Summary",
            "data": {"mixture_properties": mixture.mixture_properties},
            "visualization": "table",
        }
        dashboard_panels.append(properties_panel)

        dashboard_config = {
            "type": "interactive_dashboard",
            "title": "Mixture Analysis Dashboard",
            "panels": dashboard_panels,
            "layout": "grid",  # or 'tabs', 'accordion'
            "theme": "modern",
            "responsive": True,
        }

        return dashboard_config

    def _create_phase_diagram_widgets(self) -> Dict[str, Any]:
        """Create interactive widgets for phase diagrams."""

        widgets = {
            "temperature_range": {
                "type": "range_slider",
                "min": 200,
                "max": 500,
                "value": [250, 400],
                "label": "Temperature Range (K)",
            },
            "pressure_range": {
                "type": "range_slider",
                "min": 1e4,
                "max": 1e7,
                "value": [1e5, 1e6],
                "label": "Pressure Range (Pa)",
            },
            "resolution": {
                "type": "slider",
                "min": 10,
                "max": 100,
                "value": 50,
                "label": "Grid Resolution",
            },
        }

        return widgets

    def export_visualization(
        self,
        plot_config: Dict[str, Any],
        export_format: str = "html",
        filename: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Export visualization to file.

        Args:
            plot_config: Plot configuration
            export_format: Export format ('html', 'png', 'svg', 'pdf')
            filename: Output filename

        Returns:
            Path to exported file
        """

        if filename is None:
            filename = f"mixture_visualization.{export_format}"

        # This would integrate with actual plotting libraries
        # to generate and export the visualizations

        return filename

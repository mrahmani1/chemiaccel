"""
Interactive Lifecycle Simulation and Visualization.

This module provides comprehensive interactive tools to simulate and visualize
the evolution of material properties, degradation products, and environmental
impacts over the entire lifecycle, with dashboards to compare sustainability
profiles of different material candidates.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Check for optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, State, callback, dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Import from other sustainability modules
from .degradation_modeling import (
    DegradationPrediction,
    DegradationState,
    EnvironmentalConditions,
    EnvironmentalStressor,
    MultiScaleDegradationPredictor,
    create_accelerated_aging_conditions,
    create_outdoor_weathering_conditions,
    create_standard_material,
)
from .lca_integration import (
    ImpactCategory,
    LCAResults,
    LifecycleStage,
    MatrixLCACalculator,
    create_polymer_manufacturing_process,
    create_recycling_process,
)
from .recyclability_assessment import (
    ComprehensiveRecyclabilityAssessor,
    MaterialComposition,
    RecyclabilityAssessment,
    create_standard_plastic_composition,
)

logger = logging.getLogger(__name__)


class LifecyclePhase(Enum):
    """Phases in material lifecycle."""

    RAW_MATERIAL_EXTRACTION = "raw_material_extraction"
    MANUFACTURING = "manufacturing"
    TRANSPORTATION = "transportation"
    USE_PHASE = "use_phase"
    END_OF_LIFE = "end_of_life"
    RECYCLING = "recycling"


class VisualizationType(Enum):
    """Types of visualizations available."""

    TIMELINE = "timeline"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    BAR_CHART = "bar_chart"
    SANKEY = "sankey"
    RADAR = "radar"
    TREEMAP = "treemap"
    ANIMATION = "animation"


class ComparisonMetric(Enum):
    """Metrics for material comparison."""

    DEGRADATION_RATE = "degradation_rate"
    RECYCLABILITY_SCORE = "recyclability_score"
    CARBON_FOOTPRINT = "carbon_footprint"
    DURABILITY = "durability"
    COST = "cost"
    OVERALL_SUSTAINABILITY = "overall_sustainability"


@dataclass
class LifecycleEvent:
    """Single event in material lifecycle."""

    time: float
    phase: LifecyclePhase
    event_type: str
    properties: Dict[str, Any]
    environmental_impact: Dict[ImpactCategory, float]
    degradation_products: List[str]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecycleTrajectory:
    """Complete lifecycle trajectory for a material."""

    material_id: str
    events: List[LifecycleEvent]
    total_duration: float
    lifecycle_phases: Dict[LifecyclePhase, float]
    cumulative_impacts: Dict[ImpactCategory, float]
    final_fate: str
    sustainability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Configuration for lifecycle simulation."""

    total_duration: float = 8760.0  # 1 year in hours
    time_step: float = 24.0  # 1 day in hours
    environmental_scenarios: List[str] = field(default_factory=lambda: ["standard"])
    include_degradation: bool = True
    include_recyclability: bool = True
    include_lca: bool = True
    use_gpu: bool = True
    parallel_simulation: bool = True
    output_frequency: float = 168.0  # 1 week in hours


@dataclass
class ComparisonResult:
    """Result of material comparison analysis."""

    materials: List[str]
    comparison_metrics: Dict[ComparisonMetric, Dict[str, float]]
    rankings: Dict[ComparisonMetric, List[str]]
    pareto_optimal: List[str]
    trade_offs: Dict[str, float]
    recommendations: List[str]
    visualization_data: Dict[str, Any]


class LifecycleSimulator:
    """Core simulator for material lifecycle analysis."""

    def __init__(self, config: SimulationConfig):
        self.config = config

        # Initialize predictors
        self.degradation_predictor = MultiScaleDegradationPredictor(
            use_gpu=config.use_gpu
        )
        self.recyclability_assessor = ComprehensiveRecyclabilityAssessor(
            use_gpu=config.use_gpu
        )
        self.lca_calculator = MatrixLCACalculator(use_gpu=config.use_gpu)

        # Simulation state
        self.current_simulations = {}
        self.simulation_history = []

    async def simulate_lifecycle(
        self,
        material_id: str,
        initial_properties: Dict[str, Any],
        usage_conditions: EnvironmentalConditions,
    ) -> LifecycleTrajectory:
        """Simulate complete material lifecycle."""

        events = []
        current_time = 0.0
        current_properties = initial_properties.copy()
        cumulative_impacts = dict.fromkeys(ImpactCategory, 0.0)

        # Phase 1: Raw material extraction and manufacturing
        manufacturing_event = await self._simulate_manufacturing_phase(
            material_id, current_properties, current_time
        )
        events.append(manufacturing_event)
        current_time += 24.0  # 1 day for manufacturing

        # Update cumulative impacts
        for cat, impact in manufacturing_event.environmental_impact.items():
            cumulative_impacts[cat] += impact

        # Phase 2: Transportation
        transport_event = await self._simulate_transportation_phase(
            material_id, current_properties, current_time
        )
        events.append(transport_event)
        current_time += 24.0  # 1 day for transportation

        for cat, impact in transport_event.environmental_impact.items():
            cumulative_impacts[cat] += impact

        # Phase 3: Use phase with degradation simulation
        use_events = await self._simulate_use_phase(
            material_id, current_properties, usage_conditions, current_time
        )
        events.extend(use_events)

        if use_events:
            current_time = use_events[-1].time
            current_properties = use_events[-1].properties.copy()

            for event in use_events:
                for cat, impact in event.environmental_impact.items():
                    cumulative_impacts[cat] += impact

        # Phase 4: End-of-life assessment
        eol_event = await self._simulate_end_of_life_phase(
            material_id, current_properties, current_time
        )
        events.append(eol_event)
        current_time += 24.0  # 1 day for end-of-life processing

        for cat, impact in eol_event.environmental_impact.items():
            cumulative_impacts[cat] += impact

        # Calculate lifecycle phases duration
        lifecycle_phases = self._calculate_phase_durations(events)

        # Calculate overall sustainability score
        sustainability_score = self._calculate_sustainability_score(
            events, cumulative_impacts
        )

        # Determine final fate
        final_fate = eol_event.metadata.get("fate", "landfill")

        return LifecycleTrajectory(
            material_id=material_id,
            events=events,
            total_duration=current_time,
            lifecycle_phases=lifecycle_phases,
            cumulative_impacts=cumulative_impacts,
            final_fate=final_fate,
            sustainability_score=sustainability_score,
        )

    async def _simulate_manufacturing_phase(
        self, material_id: str, properties: Dict[str, Any], start_time: float
    ) -> LifecycleEvent:
        """Simulate manufacturing phase."""

        # Create manufacturing process for LCA
        try:
            process = create_polymer_manufacturing_process("PET")  # Default to PET
            characterization_factors = (
                self.lca_calculator._create_default_characterization_factors()
            )
            lca_results = self.lca_calculator.calculate_impacts(
                [process], characterization_factors
            )
            environmental_impact = lca_results.impact_scores
        except Exception as e:
            logger.warning(f"LCA calculation failed in manufacturing: {e}")
            environmental_impact = {
                ImpactCategory.CLIMATE_CHANGE: 2.1,
                ImpactCategory.ACIDIFICATION: 0.01,
                ImpactCategory.EUTROPHICATION: 0.005,
            }

        return LifecycleEvent(
            time=start_time,
            phase=LifecyclePhase.MANUFACTURING,
            event_type="production",
            properties=properties.copy(),
            environmental_impact=environmental_impact,
            degradation_products=[],
            description="Material manufacturing and production",
            metadata={"process_type": "polymer_production", "energy_intensity": 2.5},
        )

    async def _simulate_transportation_phase(
        self, material_id: str, properties: Dict[str, Any], start_time: float
    ) -> LifecycleEvent:
        """Simulate transportation phase."""

        # Simple transportation impact model
        transport_distance = 500.0  # km
        fuel_consumption = 0.3  # L/km
        co2_factor = 2.6  # kg CO2/L fuel

        co2_emissions = (
            transport_distance * fuel_consumption * co2_factor / 1000
        )  # per kg material

        environmental_impact = {
            ImpactCategory.CLIMATE_CHANGE: co2_emissions,
            ImpactCategory.ACIDIFICATION: co2_emissions * 0.01,
            ImpactCategory.EUTROPHICATION: co2_emissions * 0.005,
        }

        return LifecycleEvent(
            time=start_time,
            phase=LifecyclePhase.TRANSPORTATION,
            event_type="transport",
            properties=properties.copy(),
            environmental_impact=environmental_impact,
            degradation_products=[],
            description=f"Transportation over {transport_distance} km",
            metadata={"distance": transport_distance, "transport_mode": "truck"},
        )

    async def _simulate_use_phase(
        self,
        material_id: str,
        properties: Dict[str, Any],
        conditions: EnvironmentalConditions,
        start_time: float,
    ) -> List[LifecycleEvent]:
        """Simulate use phase with degradation."""

        events = []

        if not self.config.include_degradation:
            # Simple use phase without degradation
            use_event = LifecycleEvent(
                time=start_time + self.config.total_duration,
                phase=LifecyclePhase.USE_PHASE,
                event_type="usage",
                properties=properties.copy(),
                environmental_impact=dict.fromkeys(ImpactCategory, 0.0),
                degradation_products=[],
                description="Material use phase",
                metadata={"degradation_included": False},
            )
            return [use_event]

        # Run degradation simulation
        try:
            material_props = create_standard_material("polymer")
            material_props.molecular_weight = properties.get("molecular_weight", 50000)

            degradation_prediction = (
                await self.degradation_predictor.predict_degradation(
                    material_props, conditions
                )
            )

            # Convert degradation states to lifecycle events
            for i, state in enumerate(degradation_prediction.degradation_states):
                if state.time >= self.config.output_frequency * i:
                    # Calculate environmental impact from degradation
                    degradation_impact = self._calculate_degradation_impact(state)

                    # Extract degradation products
                    degradation_products = self._extract_degradation_products(
                        state, degradation_prediction.dominant_mechanisms
                    )

                    event = LifecycleEvent(
                        time=start_time + state.time,
                        phase=LifecyclePhase.USE_PHASE,
                        event_type="degradation",
                        properties={
                            "molecular_weight_retention": state.molecular_weight_retention,
                            "mechanical_retention": state.mechanical_retention,
                            "chemical_changes": state.chemical_changes,
                        },
                        environmental_impact=degradation_impact,
                        degradation_products=degradation_products,
                        description=f"Degradation state at {state.time:.1f} hours",
                        metadata={
                            "degradation_mechanisms": [
                                m.value
                                for m in degradation_prediction.dominant_mechanisms
                            ]
                        },
                    )
                    events.append(event)

        except Exception as e:
            logger.warning(f"Degradation simulation failed: {e}")
            # Fallback to simple degradation model
            events = self._simple_degradation_simulation(
                material_id, properties, start_time, conditions
            )

        return events

    async def _simulate_end_of_life_phase(
        self, material_id: str, properties: Dict[str, Any], start_time: float
    ) -> LifecycleEvent:
        """Simulate end-of-life phase."""

        # Assess recyclability
        fate = "landfill"
        environmental_impact = dict.fromkeys(ImpactCategory, 0.0)

        if self.config.include_recyclability:
            try:
                composition = MaterialComposition(
                    polymer_content={
                        "polymer": properties.get("molecular_weight_retention", 0.8)
                    },
                    additive_content={"additives": 0.05},
                    filler_content={},
                )

                assessment = await self.recyclability_assessor.assess_comprehensive_recyclability(
                    composition
                )

                if assessment.circularity_metrics.recyclability_score > 0.6:
                    fate = "recycling"
                    # Calculate recycling impact
                    try:
                        recycling_process = create_recycling_process(
                            "mechanical", "PET"
                        )
                        characterization_factors = (
                            self.lca_calculator._create_default_characterization_factors()
                        )
                        lca_results = self.lca_calculator.calculate_impacts(
                            [recycling_process], characterization_factors
                        )
                        environmental_impact = lca_results.impact_scores
                    except:
                        environmental_impact = {
                            ImpactCategory.CLIMATE_CHANGE: 0.5,
                            ImpactCategory.ACIDIFICATION: 0.002,
                            ImpactCategory.EUTROPHICATION: 0.001,
                        }
                elif assessment.circularity_metrics.biodegradability_score > 0.5:
                    fate = "biodegradation"
                    environmental_impact = dict.fromkeys(ImpactCategory, 0.1)
                else:
                    fate = "landfill"
                    environmental_impact = dict.fromkeys(ImpactCategory, 0.05)

            except Exception as e:
                logger.warning(f"Recyclability assessment failed: {e}")
                fate = "landfill"
                environmental_impact = dict.fromkeys(ImpactCategory, 0.05)

        return LifecycleEvent(
            time=start_time,
            phase=LifecyclePhase.END_OF_LIFE,
            event_type="disposal",
            properties=properties.copy(),
            environmental_impact=environmental_impact,
            degradation_products=[],
            description=f"End-of-life: {fate}",
            metadata={
                "fate": fate,
                "recyclability_assessed": self.config.include_recyclability,
            },
        )

    def _calculate_degradation_impact(
        self, state: DegradationState
    ) -> Dict[ImpactCategory, float]:
        """Calculate environmental impact from degradation."""

        # Simple model: impact proportional to degradation
        degradation_factor = 1.0 - state.molecular_weight_retention

        base_impacts = {
            ImpactCategory.CLIMATE_CHANGE: 0.01,
            ImpactCategory.ACIDIFICATION: 0.0001,
            ImpactCategory.EUTROPHICATION: 0.00005,
        }

        return {
            cat: impact * degradation_factor for cat, impact in base_impacts.items()
        }

    def _extract_degradation_products(
        self, state: DegradationState, mechanisms: List[Any]
    ) -> List[str]:
        """Extract degradation products based on mechanisms."""

        products = []

        for mechanism in mechanisms:
            mechanism_str = (
                mechanism.value if hasattr(mechanism, "value") else str(mechanism)
            )

            if "uv" in mechanism_str.lower():
                products.extend(["free_radicals", "carbonyl_compounds"])
            elif "thermal" in mechanism_str.lower():
                products.extend(["volatile_organics", "char"])
            elif "oxidative" in mechanism_str.lower():
                products.extend(["peroxides", "aldehydes", "acids"])
            elif "hydrolytic" in mechanism_str.lower():
                products.extend(["oligomers", "monomers"])

        return list(set(products))  # Remove duplicates

    def _simple_degradation_simulation(
        self,
        material_id: str,
        properties: Dict[str, Any],
        start_time: float,
        conditions: EnvironmentalConditions,
    ) -> List[LifecycleEvent]:
        """Simple degradation simulation fallback."""

        events = []
        time_points = np.arange(
            0, self.config.total_duration, self.config.output_frequency
        )

        for t in time_points:
            # Simple exponential decay
            degradation_rate = 0.0001  # per hour
            retention = np.exp(-degradation_rate * t)

            event = LifecycleEvent(
                time=start_time + t,
                phase=LifecyclePhase.USE_PHASE,
                event_type="simple_degradation",
                properties={
                    "molecular_weight_retention": retention,
                    "mechanical_retention": {"tensile_strength": retention},
                    "chemical_changes": {},
                },
                environmental_impact=dict.fromkeys(
                    ImpactCategory, 0.001 * (1 - retention)
                ),
                degradation_products=["unknown_products"],
                description=f"Simple degradation at {t:.1f} hours",
                metadata={"model": "exponential_decay"},
            )
            events.append(event)

        return events

    def _calculate_phase_durations(
        self, events: List[LifecycleEvent]
    ) -> Dict[LifecyclePhase, float]:
        """Calculate duration of each lifecycle phase."""

        phase_durations = dict.fromkeys(LifecyclePhase, 0.0)

        # Group events by phase
        phase_events = {}
        for event in events:
            if event.phase not in phase_events:
                phase_events[event.phase] = []
            phase_events[event.phase].append(event)

        # Calculate durations
        for phase, phase_event_list in phase_events.items():
            if len(phase_event_list) > 1:
                times = [e.time for e in phase_event_list]
                phase_durations[phase] = max(times) - min(times)
            else:
                # Single event phases get default duration
                phase_durations[phase] = 24.0  # 1 day

        return phase_durations

    def _calculate_sustainability_score(
        self,
        events: List[LifecycleEvent],
        cumulative_impacts: Dict[ImpactCategory, float],
    ) -> float:
        """Calculate overall sustainability score."""

        scores = []

        # Environmental impact score (lower is better)
        total_climate_impact = cumulative_impacts.get(ImpactCategory.CLIMATE_CHANGE, 0)
        if total_climate_impact > 0:
            scores.append(1.0 / (1.0 + total_climate_impact))
        else:
            scores.append(1.0)

        # Durability score
        use_events = [e for e in events if e.phase == LifecyclePhase.USE_PHASE]
        if use_events:
            final_retention = use_events[-1].properties.get(
                "molecular_weight_retention", 0.5
            )
            scores.append(final_retention)

        # End-of-life score
        eol_events = [e for e in events if e.phase == LifecyclePhase.END_OF_LIFE]
        if eol_events:
            fate = eol_events[0].metadata.get("fate", "landfill")
            if fate == "recycling":
                scores.append(1.0)
            elif fate == "biodegradation":
                scores.append(0.8)
            else:
                scores.append(0.3)

        return np.mean(scores) if scores else 0.5


class LifecycleVisualizer:
    """Advanced visualization engine for lifecycle data."""

    def __init__(self, use_plotly: bool = True):
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE
        self.color_schemes = {
            "phases": {
                LifecyclePhase.MANUFACTURING: "#1f77b4",
                LifecyclePhase.TRANSPORTATION: "#ff7f0e",
                LifecyclePhase.USE_PHASE: "#2ca02c",
                LifecyclePhase.END_OF_LIFE: "#d62728",
            },
            "impacts": {
                ImpactCategory.CLIMATE_CHANGE: "#FF6B6B",
                ImpactCategory.ACIDIFICATION: "#4ECDC4",
                ImpactCategory.EUTROPHICATION: "#45B7D1",
            },
        }

    def create_lifecycle_timeline(
        self,
        trajectory: LifecycleTrajectory,
        properties_to_show: Optional[List[str]] = None,
    ) -> Union[go.Figure, str]:
        """Create interactive timeline visualization."""

        if not self.use_plotly:
            return "Plotly not available for timeline visualization"

        properties_to_show = properties_to_show or ["molecular_weight_retention"]

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Material Properties Over Time", "Environmental Impact"),
            vertical_spacing=0.1,
        )

        # Plot material properties
        for prop in properties_to_show:
            times = []
            values = []

            for event in trajectory.events:
                if prop in event.properties:
                    times.append(event.time)
                    if isinstance(event.properties[prop], dict):
                        values.append(list(event.properties[prop].values())[0])
                    else:
                        values.append(event.properties[prop])

            if times and values:
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=values,
                        mode="lines+markers",
                        name=prop.replace("_", " ").title(),
                        line=dict(width=2),
                        marker=dict(size=6),
                    ),
                    row=1,
                    col=1,
                )

        # Plot environmental impacts
        impact_data = {cat.value: [] for cat in ImpactCategory}
        times = []

        cumulative_impacts = dict.fromkeys(ImpactCategory, 0.0)
        for event in trajectory.events:
            times.append(event.time)
            for cat, impact in event.environmental_impact.items():
                cumulative_impacts[cat] += impact
                impact_data[cat.value].append(cumulative_impacts[cat])

        for cat_name, values in impact_data.items():
            if values and any(v > 0 for v in values):
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=values,
                        mode="lines",
                        name=f"{cat_name.replace('_', ' ').title()} (cumulative)",
                        line=dict(width=2),
                    ),
                    row=2,
                    col=1,
                )

        # Add phase annotations
        phase_colors = self.color_schemes["phases"]
        for event in trajectory.events:
            if event.event_type in ["production", "transport", "disposal"]:
                fig.add_vline(
                    x=event.time,
                    line_dash="dash",
                    line_color=phase_colors.get(event.phase, "gray"),
                    annotation_text=event.phase.value.replace("_", " ").title(),
                    annotation_position="top",
                )

        fig.update_layout(
            title=f"Lifecycle Analysis: {trajectory.material_id}",
            height=800,
            showlegend=True,
            hovermode="x unified",
        )

        fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
        fig.update_yaxes(title_text="Property Value", row=1, col=1)
        fig.update_yaxes(title_text="Impact (kg CO2-eq)", row=2, col=1)

        return fig

    def create_phase_breakdown(
        self, trajectory: LifecycleTrajectory
    ) -> Union[go.Figure, str]:
        """Create phase breakdown visualization."""

        if not self.use_plotly:
            return "Plotly not available for phase breakdown"

        # Calculate impact by phase
        phase_impacts = {
            phase: dict.fromkeys(ImpactCategory, 0.0) for phase in LifecyclePhase
        }

        for event in trajectory.events:
            for cat, impact in event.environmental_impact.items():
                phase_impacts[event.phase][cat] += impact

        # Create stacked bar chart
        fig = go.Figure()

        phases = list(phase_impacts.keys())
        impact_categories = list(ImpactCategory)

        for cat in impact_categories:
            values = [phase_impacts[phase][cat] for phase in phases]
            fig.add_trace(
                go.Bar(
                    name=cat.value.replace("_", " ").title(),
                    x=[phase.value.replace("_", " ").title() for phase in phases],
                    y=values,
                    marker_color=self.color_schemes["impacts"].get(cat, "gray"),
                )
            )

        fig.update_layout(
            title="Environmental Impact by Lifecycle Phase",
            barmode="stack",
            xaxis_title="Lifecycle Phase",
            yaxis_title="Environmental Impact",
            height=500,
        )

        return fig

    def create_degradation_heatmap(
        self,
        trajectories: List[LifecycleTrajectory],
        property_name: str = "molecular_weight_retention",
    ) -> Union[go.Figure, str]:
        """Create heatmap of degradation across materials and time."""

        if not self.use_plotly:
            return "Plotly not available for heatmap"

        # Extract data for heatmap
        material_names = [t.material_id for t in trajectories]
        time_points = set()

        for traj in trajectories:
            for event in traj.events:
                if event.phase == LifecyclePhase.USE_PHASE:
                    time_points.add(event.time)

        time_points = sorted(list(time_points))

        # Build data matrix
        data_matrix = []
        for traj in trajectories:
            row = []
            for time_point in time_points:
                # Find closest event
                use_events = [
                    e for e in traj.events if e.phase == LifecyclePhase.USE_PHASE
                ]
                if use_events:
                    closest_event = min(
                        use_events, key=lambda e: abs(e.time - time_point)
                    )
                    value = closest_event.properties.get(property_name, 0.5)
                    if isinstance(value, dict):
                        value = list(value.values())[0] if value.values() else 0.5
                    row.append(value)
                else:
                    row.append(0.5)
            data_matrix.append(row)

        fig = go.Figure(
            data=go.Heatmap(
                z=data_matrix,
                x=[f"{t:.0f}h" for t in time_points],
                y=material_names,
                colorscale="RdYlGn",
                colorbar=dict(title=property_name.replace("_", " ").title()),
            )
        )

        fig.update_layout(
            title=f"{property_name.replace('_', ' ').title()} Across Materials and Time",
            xaxis_title="Time",
            yaxis_title="Material",
            height=600,
        )

        return fig

    def create_sustainability_radar(
        self, trajectories: List[LifecycleTrajectory]
    ) -> Union[go.Figure, str]:
        """Create radar chart for sustainability comparison."""

        if not self.use_plotly:
            return "Plotly not available for radar chart"

        fig = go.Figure()

        metrics = ["Durability", "Recyclability", "Low Impact", "Overall Score"]

        for traj in trajectories:
            # Calculate metrics
            eol_events = [
                e for e in traj.events if e.phase == LifecyclePhase.END_OF_LIFE
            ]
            use_events = [e for e in traj.events if e.phase == LifecyclePhase.USE_PHASE]

            durability = (
                use_events[-1].properties.get("molecular_weight_retention", 0.5)
                if use_events
                else 0.5
            )
            recyclability = (
                1.0
                if eol_events and eol_events[0].metadata.get("fate") == "recycling"
                else 0.3
            )
            low_impact = 1.0 / (
                1.0 + traj.cumulative_impacts.get(ImpactCategory.CLIMATE_CHANGE, 1.0)
            )
            overall_score = traj.sustainability_score

            values = [durability, recyclability, low_impact, overall_score]

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill="toself",
                    name=traj.material_id,
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Sustainability Radar Chart",
            height=600,
        )

        return fig

    def create_sankey_diagram(
        self, trajectory: LifecycleTrajectory
    ) -> Union[go.Figure, str]:
        """Create Sankey diagram for material flow."""

        if not self.use_plotly:
            return "Plotly not available for Sankey diagram"

        # Define nodes and links for material flow
        phases = [phase.value.replace("_", " ").title() for phase in LifecyclePhase]
        eol_fate = trajectory.final_fate.title()

        node_labels = phases + [eol_fate]

        # Create links between consecutive phases
        source_indices = list(range(len(phases) - 1))  # All phases except last
        target_indices = list(range(1, len(phases)))  # All phases except first
        values = [100] * (len(phases) - 1)  # Assume 100% flow between phases

        # Add end-of-life flow
        source_indices.append(len(phases) - 1)  # From END_OF_LIFE
        target_indices.append(len(phases))  # To fate
        values.append(100)

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=node_labels,
                        color="blue",
                    ),
                    link=dict(
                        source=source_indices, target=target_indices, value=values
                    ),
                )
            ]
        )

        fig.update_layout(title=f"Material Flow: {trajectory.material_id}", height=400)

        return fig


class MaterialComparator:
    """Advanced material comparison and benchmarking."""

    def __init__(self):
        self.comparison_metrics = {
            ComparisonMetric.DEGRADATION_RATE: lambda t: self._extract_degradation_rate(
                t
            ),
            ComparisonMetric.RECYCLABILITY_SCORE: lambda t: self._extract_recyclability_score(
                t
            ),
            ComparisonMetric.CARBON_FOOTPRINT: lambda t: t.cumulative_impacts.get(
                ImpactCategory.CLIMATE_CHANGE, 0
            ),
            ComparisonMetric.DURABILITY: lambda t: self._extract_durability(t),
            ComparisonMetric.OVERALL_SUSTAINABILITY: lambda t: t.sustainability_score,
        }

    def compare_materials(
        self,
        trajectories: List[LifecycleTrajectory],
        metrics: Optional[List[ComparisonMetric]] = None,
    ) -> ComparisonResult:
        """Compare multiple materials across sustainability metrics."""

        metrics = metrics or list(ComparisonMetric)
        materials = [t.material_id for t in trajectories]

        # Calculate metric values
        comparison_metrics = {}
        for metric in metrics:
            if metric in self.comparison_metrics:
                values = {}
                for traj in trajectories:
                    try:
                        values[traj.material_id] = self.comparison_metrics[metric](traj)
                    except Exception as e:
                        logger.warning(
                            f"Failed to calculate {metric} for {traj.material_id}: {e}"
                        )
                        values[traj.material_id] = 0.0
                comparison_metrics[metric] = values

        # Calculate rankings
        rankings = {}
        for metric, values in comparison_metrics.items():
            # Sort materials by metric value (higher is better for most metrics)
            if metric in [
                ComparisonMetric.DEGRADATION_RATE,
                ComparisonMetric.CARBON_FOOTPRINT,
            ]:
                # Lower is better for these metrics
                sorted_materials = sorted(values.keys(), key=lambda m: values[m])
            else:
                # Higher is better
                sorted_materials = sorted(
                    values.keys(), key=lambda m: values[m], reverse=True
                )
            rankings[metric] = sorted_materials

        # Identify Pareto optimal solutions
        pareto_optimal = self._identify_pareto_optimal(trajectories, comparison_metrics)

        # Calculate trade-offs
        trade_offs = self._calculate_trade_offs(comparison_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            comparison_metrics, rankings, pareto_optimal
        )

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(
            comparison_metrics, rankings
        )

        return ComparisonResult(
            materials=materials,
            comparison_metrics=comparison_metrics,
            rankings=rankings,
            pareto_optimal=pareto_optimal,
            trade_offs=trade_offs,
            recommendations=recommendations,
            visualization_data=visualization_data,
        )

    def _extract_degradation_rate(self, trajectory: LifecycleTrajectory) -> float:
        """Extract degradation rate from trajectory."""
        use_events = [
            e for e in trajectory.events if e.phase == LifecyclePhase.USE_PHASE
        ]
        if not use_events:
            return 0.001

        initial_retention = 1.0
        final_retention = use_events[-1].properties.get(
            "molecular_weight_retention", 0.5
        )
        time_span = use_events[-1].time - use_events[0].time

        if time_span > 0:
            return (initial_retention - final_retention) / time_span
        else:
            return 0.001

    def _extract_recyclability_score(self, trajectory: LifecycleTrajectory) -> float:
        """Extract recyclability score from trajectory."""
        eol_events = [
            e for e in trajectory.events if e.phase == LifecyclePhase.END_OF_LIFE
        ]
        if eol_events and eol_events[0].metadata.get("fate") == "recycling":
            return 1.0
        elif eol_events and eol_events[0].metadata.get("fate") == "biodegradation":
            return 0.8
        else:
            return 0.3

    def _extract_durability(self, trajectory: LifecycleTrajectory) -> float:
        """Extract durability score from trajectory."""
        use_events = [
            e for e in trajectory.events if e.phase == LifecyclePhase.USE_PHASE
        ]
        if use_events:
            return use_events[-1].properties.get("molecular_weight_retention", 0.5)
        else:
            return 0.5

    def _identify_pareto_optimal(
        self,
        trajectories: List[LifecycleTrajectory],
        comparison_metrics: Dict[ComparisonMetric, Dict[str, float]],
    ) -> List[str]:
        """Identify Pareto optimal materials."""

        pareto_optimal = []

        for traj in trajectories:
            is_dominated = False

            for other_traj in trajectories:
                if other_traj.material_id == traj.material_id:
                    continue

                # Check if other material dominates this one
                dominates = True
                better_in_any = False

                for metric in comparison_metrics:
                    this_value = comparison_metrics[metric][traj.material_id]
                    other_value = comparison_metrics[metric][other_traj.material_id]

                    # Adjust for minimization metrics
                    if metric in [
                        ComparisonMetric.DEGRADATION_RATE,
                        ComparisonMetric.CARBON_FOOTPRINT,
                    ]:
                        if other_value > this_value:  # Other is worse
                            dominates = False
                            break
                        elif other_value < this_value:  # Other is better
                            better_in_any = True
                    else:
                        if other_value < this_value:  # Other is worse
                            dominates = False
                            break
                        elif other_value > this_value:  # Other is better
                            better_in_any = True

                if dominates and better_in_any:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_optimal.append(traj.material_id)

        return pareto_optimal

    def _calculate_trade_offs(
        self, comparison_metrics: Dict[ComparisonMetric, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate trade-offs between metrics."""

        trade_offs = {}
        metric_pairs = [
            (ComparisonMetric.DURABILITY, ComparisonMetric.RECYCLABILITY_SCORE),
            (ComparisonMetric.CARBON_FOOTPRINT, ComparisonMetric.DURABILITY),
            (
                ComparisonMetric.DEGRADATION_RATE,
                ComparisonMetric.OVERALL_SUSTAINABILITY,
            ),
        ]

        for metric1, metric2 in metric_pairs:
            if metric1 in comparison_metrics and metric2 in comparison_metrics:
                values1 = list(comparison_metrics[metric1].values())
                values2 = list(comparison_metrics[metric2].values())

                if len(values1) > 1 and len(values2) > 1:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    trade_offs[f"{metric1.value}_vs_{metric2.value}"] = correlation

        return trade_offs

    def _generate_recommendations(
        self,
        comparison_metrics: Dict[ComparisonMetric, Dict[str, float]],
        rankings: Dict[ComparisonMetric, List[str]],
        pareto_optimal: List[str],
    ) -> List[str]:
        """Generate material recommendations."""

        recommendations = []

        # Overall best material
        if ComparisonMetric.OVERALL_SUSTAINABILITY in rankings:
            best_overall = rankings[ComparisonMetric.OVERALL_SUSTAINABILITY][0]
            recommendations.append(f"Best overall sustainability: {best_overall}")

        # Best for specific applications
        if ComparisonMetric.DURABILITY in rankings:
            most_durable = rankings[ComparisonMetric.DURABILITY][0]
            recommendations.append(f"Most durable material: {most_durable}")

        if ComparisonMetric.RECYCLABILITY_SCORE in rankings:
            most_recyclable = rankings[ComparisonMetric.RECYCLABILITY_SCORE][0]
            recommendations.append(f"Most recyclable material: {most_recyclable}")

        if ComparisonMetric.CARBON_FOOTPRINT in rankings:
            lowest_carbon = rankings[ComparisonMetric.CARBON_FOOTPRINT][0]
            recommendations.append(f"Lowest carbon footprint: {lowest_carbon}")

        # Pareto optimal recommendations
        if pareto_optimal:
            recommendations.append(
                f"Pareto optimal materials: {', '.join(pareto_optimal)}"
            )

        return recommendations

    def _prepare_visualization_data(
        self,
        comparison_metrics: Dict[ComparisonMetric, Dict[str, float]],
        rankings: Dict[ComparisonMetric, List[str]],
    ) -> Dict[str, Any]:
        """Prepare data for visualization."""

        return {
            "metrics_data": comparison_metrics,
            "rankings_data": rankings,
            "prepared_for_plotly": PLOTLY_AVAILABLE,
            "prepared_for_pandas": PANDAS_AVAILABLE,
        }


class SustainabilityDashboard:
    """Interactive dashboard for sustainability analysis."""

    def __init__(self, use_dash: bool = True):
        self.use_dash = use_dash and DASH_AVAILABLE
        self.simulator = None
        self.visualizer = LifecycleVisualizer()
        self.comparator = MaterialComparator()

        # Dashboard state
        self.current_trajectories = []
        self.current_comparison = None

        if self.use_dash:
            self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
            self._setup_dashboard_layout()
            self._setup_dashboard_callbacks()

    def _setup_dashboard_layout(self):
        """Setup dashboard layout."""

        if not self.use_dash:
            return

        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Sustainability Analysis Dashboard",
                                    className="text-center mb-4",
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Material Selection"),
                                        dbc.CardBody(
                                            [
                                                dbc.Label("Select Materials:"),
                                                dcc.Dropdown(
                                                    id="material-dropdown",
                                                    options=[
                                                        {
                                                            "label": "PET Bottle",
                                                            "value": "PET",
                                                        },
                                                        {
                                                            "label": "HDPE Container",
                                                            "value": "HDPE",
                                                        },
                                                        {
                                                            "label": "PLA Film",
                                                            "value": "PLA",
                                                        },
                                                        {
                                                            "label": "PP Packaging",
                                                            "value": "PP",
                                                        },
                                                    ],
                                                    value=["PET"],
                                                    multi=True,
                                                ),
                                                html.Br(),
                                                dbc.Label(
                                                    "Simulation Duration (hours):"
                                                ),
                                                dcc.Slider(
                                                    id="duration-slider",
                                                    min=168,
                                                    max=8760,
                                                    step=168,
                                                    value=2000,
                                                    marks={
                                                        168: "1 week",
                                                        2000: "~3 months",
                                                        8760: "1 year",
                                                    },
                                                ),
                                                html.Br(),
                                                dbc.Button(
                                                    "Run Simulation",
                                                    id="run-simulation",
                                                    color="primary",
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dcc.Tabs(
                                    id="main-tabs",
                                    value="timeline-tab",
                                    children=[
                                        dcc.Tab(
                                            label="Timeline Analysis",
                                            value="timeline-tab",
                                        ),
                                        dcc.Tab(
                                            label="Material Comparison",
                                            value="comparison-tab",
                                        ),
                                        dcc.Tab(
                                            label="Sustainability Metrics",
                                            value="metrics-tab",
                                        ),
                                        dcc.Tab(
                                            label="Phase Breakdown", value="phases-tab"
                                        ),
                                    ],
                                ),
                                html.Div(id="tab-content"),
                            ],
                            width=9,
                        ),
                    ]
                ),
                dbc.Row([dbc.Col([html.Div(id="status-output", className="mt-3")])]),
            ],
            fluid=True,
        )

    def _setup_dashboard_callbacks(self):
        """Setup dashboard callbacks."""

        if not self.use_dash:
            return

        @self.app.callback(
            Output("tab-content", "children"),
            [Input("main-tabs", "value")],
            [State("material-dropdown", "value")],
        )
        def render_tab_content(active_tab, selected_materials):
            if not self.current_trajectories:
                return html.Div(
                    "No simulation data available. Please run a simulation first."
                )

            if active_tab == "timeline-tab":
                return self._render_timeline_tab()
            elif active_tab == "comparison-tab":
                return self._render_comparison_tab()
            elif active_tab == "metrics-tab":
                return self._render_metrics_tab()
            elif active_tab == "phases-tab":
                return self._render_phases_tab()

            return html.Div("Select a tab to view analysis.")

        @self.app.callback(
            Output("status-output", "children"),
            [Input("run-simulation", "n_clicks")],
            [State("material-dropdown", "value"), State("duration-slider", "value")],
        )
        def run_simulation(n_clicks, selected_materials, duration):
            if n_clicks is None:
                return ""

            if not selected_materials:
                return dbc.Alert(
                    "Please select at least one material.", color="warning"
                )

            try:
                # Run simulation for selected materials
                asyncio.run(
                    self._run_dashboard_simulation(selected_materials, duration)
                )
                return dbc.Alert(
                    f"Simulation completed for {len(selected_materials)} materials.",
                    color="success",
                )

            except Exception as e:
                return dbc.Alert(f"Simulation failed: {str(e)}", color="danger")

    async def _run_dashboard_simulation(self, materials: List[str], duration: float):
        """Run simulation for dashboard."""

        config = SimulationConfig(
            total_duration=duration, use_gpu=False  # Use CPU for dashboard stability
        )

        self.simulator = LifecycleSimulator(config)
        self.current_trajectories = []

        for material in materials:
            # Create material properties
            properties = {
                "molecular_weight": 50000 if material != "PLA" else 100000,
                "crystallinity": 0.3 if material in ["PET", "HDPE"] else 0.1,
            }

            # Create usage conditions
            conditions = create_outdoor_weathering_conditions(
                "temperate", duration=duration
            )

            # Run simulation
            trajectory = await self.simulator.simulate_lifecycle(
                material_id=material,
                initial_properties=properties,
                usage_conditions=conditions,
            )

            self.current_trajectories.append(trajectory)

        # Run comparison
        if len(self.current_trajectories) > 1:
            self.current_comparison = self.comparator.compare_materials(
                self.current_trajectories
            )

    def _render_timeline_tab(self):
        """Render timeline analysis tab."""

        graphs = []
        for trajectory in self.current_trajectories:
            fig = self.visualizer.create_lifecycle_timeline(trajectory)
            if isinstance(fig, go.Figure):
                graphs.append(dcc.Graph(figure=fig))

        return html.Div(graphs)

    def _render_comparison_tab(self):
        """Render material comparison tab."""

        if not self.current_comparison:
            return html.Div("No comparison data available.")

        # Create comparison visualization
        fig = self.visualizer.create_sustainability_radar(self.current_trajectories)

        return html.Div(
            [
                (
                    dcc.Graph(figure=fig)
                    if isinstance(fig, go.Figure)
                    else html.Div(str(fig))
                ),
                html.H4("Material Rankings"),
                html.Div(
                    [
                        html.P(
                            f"{metric.value.replace('_', ' ').title()}: {', '.join(ranking)}"
                        )
                        for metric, ranking in self.current_comparison.rankings.items()
                    ]
                ),
            ]
        )

    def _render_metrics_tab(self):
        """Render sustainability metrics tab."""

        if not self.current_comparison:
            return html.Div("No metrics data available.")

        metrics_table = []
        for material in self.current_comparison.materials:
            row = [material]
            for metric, values in self.current_comparison.comparison_metrics.items():
                row.append(f"{values[material]:.3f}")
            metrics_table.append(row)

        headers = ["Material"] + [
            m.value.replace("_", " ").title()
            for m in self.current_comparison.comparison_metrics.keys()
        ]

        return html.Div(
            [
                html.H4("Sustainability Metrics Comparison"),
                (
                    dbc.Table.from_dataframe(
                        pd.DataFrame(metrics_table, columns=headers),
                        striped=True,
                        bordered=True,
                        hover=True,
                    )
                    if PANDAS_AVAILABLE
                    else html.Div("Pandas not available for table display")
                ),
            ]
        )

    def _render_phases_tab(self):
        """Render phase breakdown tab."""

        graphs = []
        for trajectory in self.current_trajectories:
            fig = self.visualizer.create_phase_breakdown(trajectory)
            if isinstance(fig, go.Figure):
                graphs.append(dcc.Graph(figure=fig))

        return html.Div(graphs)

    def run_dashboard(
        self, host: str = "127.0.0.1", port: int = 8050, debug: bool = True
    ):
        """Launch interactive dashboard."""

        if not self.use_dash:
            logger.error("Dash not available. Cannot run dashboard.")
            return

        logger.info(f"Starting dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


# Utility functions


async def simulate_material_lifecycle(
    material_id: str,
    properties: Dict[str, Any],
    conditions: Optional[EnvironmentalConditions] = None,
    duration: float = 8760.0,
    use_gpu: bool = True,
) -> LifecycleTrajectory:
    """Utility function to simulate material lifecycle."""

    if conditions is None:
        conditions = create_outdoor_weathering_conditions(
            "temperate", duration=duration
        )

    config = SimulationConfig(total_duration=duration, use_gpu=use_gpu)

    simulator = LifecycleSimulator(config)
    return await simulator.simulate_lifecycle(material_id, properties, conditions)


def create_sustainability_dashboard(
    materials: Optional[List[str]] = None,
) -> SustainabilityDashboard:
    """Create interactive sustainability dashboard."""

    dashboard = SustainabilityDashboard()

    if materials:
        # Pre-populate with materials
        logger.info(f"Dashboard created for materials: {materials}")

    return dashboard


def analyze_material_trade_offs(
    trajectories: List[LifecycleTrajectory],
) -> Dict[str, Any]:
    """Analyze trade-offs between materials."""

    comparator = MaterialComparator()
    comparison = comparator.compare_materials(trajectories)

    return {
        "comparison_results": comparison,
        "pareto_optimal": comparison.pareto_optimal,
        "trade_offs": comparison.trade_offs,
        "recommendations": comparison.recommendations,
    }


# Exception classes
class LifecycleSimulationError(Exception):
    """Base exception for lifecycle simulation errors."""

    pass


class VisualizationError(LifecycleSimulationError):
    """Raised when visualization fails."""

    pass


class DashboardError(LifecycleSimulationError):
    """Raised when dashboard operations fail."""

    pass


class SimulationConfigurationError(LifecycleSimulationError):
    """Raised when simulation configuration is invalid."""

    pass

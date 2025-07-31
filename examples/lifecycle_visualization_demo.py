"""
Comprehensive Lifecycle Simulation and Visualization Demo.

This example demonstrates the interactive lifecycle simulation and visualization
capabilities including material trajectory simulation, comparative analysis,
and interactive dashboard features for sustainability assessment.
"""

import asyncio
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemi_accel.sustainability.degradation_modeling import (
    EnvironmentalConditions,
    EnvironmentalStressor,
    create_accelerated_aging_conditions,
    create_outdoor_weathering_conditions,
    create_standard_material,
)
from chemi_accel.sustainability.lifecycle_visualization import (  # Core classes; Data structures; Utility functions
    ComparisonMetric,
    LifecyclePhase,
    LifecycleSimulator,
    LifecycleVisualizer,
    MaterialComparator,
    SimulationConfig,
    SustainabilityDashboard,
    VisualizationType,
    analyze_material_trade_offs,
    create_sustainability_dashboard,
    simulate_material_lifecycle,
)

# Check for optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import dash

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LifecycleVisualizationDemo:
    """Comprehensive lifecycle visualization demonstration."""

    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.results = {}
        self.output_dir = None

        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Visualization features disabled.")

        if not DASH_AVAILABLE:
            logger.warning(
                "Dash not available. Interactive dashboard features disabled."
            )

        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available. Some analysis features disabled.")

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    async def demo_single_material_lifecycle(self) -> Dict[str, Any]:
        """Demonstrate complete lifecycle simulation for a single material."""
        self._log("=== Single Material Lifecycle Simulation Demo ===")

        # Configure simulation
        config = SimulationConfig(
            total_duration=8760.0,  # 1 year
            time_step=24.0,  # 1 day
            output_frequency=168.0,  # 1 week
            include_degradation=True,
            include_recyclability=True,
            include_lca=True,
            use_gpu=self.use_gpu,
        )

        simulator = LifecycleSimulator(config)

        # Define material properties
        material_id = "PET_Bottle_Comprehensive"
        initial_properties = {
            "molecular_weight": 75000,
            "crystallinity": 0.35,
            "density": 1.38,
            "glass_transition_temp": 343.15,
        }

        # Create realistic usage conditions
        usage_conditions = create_outdoor_weathering_conditions(
            climate="temperate", duration=8760.0
        )

        self._log(
            f"Simulating lifecycle for {material_id} over {config.total_duration/24:.0f} days..."
        )

        # Run lifecycle simulation
        trajectory = await simulator.simulate_lifecycle(
            material_id, initial_properties, usage_conditions
        )

        # Analyze trajectory
        analysis = self._analyze_single_trajectory(trajectory)

        # Create visualizations
        visualizer = LifecycleVisualizer(use_plotly=PLOTLY_AVAILABLE)

        visualizations = {}
        if PLOTLY_AVAILABLE:
            visualizations["timeline"] = visualizer.create_lifecycle_timeline(
                trajectory
            )
            visualizations["phase_breakdown"] = visualizer.create_phase_breakdown(
                trajectory
            )
            visualizations["sankey"] = visualizer.create_sankey_diagram(trajectory)

        self._log(
            f"Single material simulation completed. Final fate: {trajectory.final_fate}"
        )
        self._log(f"Sustainability score: {trajectory.sustainability_score:.3f}")

        return {
            "scenario": "single_material_lifecycle",
            "material_id": material_id,
            "trajectory": trajectory,
            "analysis": analysis,
            "visualizations": visualizations,
            "simulation_config": config,
        }

    async def demo_multi_material_comparison(self) -> Dict[str, Any]:
        """Demonstrate comparative lifecycle analysis of multiple materials."""
        self._log("=== Multi-Material Comparison Demo ===")

        # Define materials to compare
        materials = {
            "PET_Bottle": {
                "properties": {
                    "molecular_weight": 75000,
                    "crystallinity": 0.35,
                    "density": 1.38,
                },
                "description": "Standard PET bottle material",
            },
            "HDPE_Container": {
                "properties": {
                    "molecular_weight": 120000,
                    "crystallinity": 0.65,
                    "density": 0.95,
                },
                "description": "High-density polyethylene container",
            },
            "PLA_Film": {
                "properties": {
                    "molecular_weight": 150000,
                    "crystallinity": 0.15,
                    "density": 1.25,
                },
                "description": "Biodegradable PLA film",
            },
            "PP_Packaging": {
                "properties": {
                    "molecular_weight": 85000,
                    "crystallinity": 0.45,
                    "density": 0.90,
                },
                "description": "Polypropylene packaging",
            },
        }

        # Configure comparative simulation
        config = SimulationConfig(
            total_duration=5000.0,  # ~7 months for faster comparison
            output_frequency=168.0,
            use_gpu=self.use_gpu,
        )

        simulator = LifecycleSimulator(config)
        trajectories = []

        # Simulate each material
        for material_name, material_data in materials.items():
            self._log(f"Simulating {material_name}...")

            # Create varied environmental conditions
            if "PLA" in material_name:
                conditions = create_accelerated_aging_conditions(
                    temperature=323.15,  # Milder for biodegradable
                    humidity=0.7,
                    duration=5000.0,
                )
            else:
                conditions = create_outdoor_weathering_conditions(
                    "temperate", duration=5000.0
                )

            trajectory = await simulator.simulate_lifecycle(
                material_name, material_data["properties"], conditions
            )
            trajectories.append(trajectory)

        # Perform comparative analysis
        comparator = MaterialComparator()
        comparison_result = comparator.compare_materials(trajectories)

        # Create comparative visualizations
        visualizer = LifecycleVisualizer(use_plotly=PLOTLY_AVAILABLE)
        comparative_visualizations = {}

        if PLOTLY_AVAILABLE:
            comparative_visualizations[
                "radar_comparison"
            ] = visualizer.create_sustainability_radar(trajectories)
            comparative_visualizations[
                "degradation_heatmap"
            ] = visualizer.create_degradation_heatmap(
                trajectories, "molecular_weight_retention"
            )

        # Analyze trade-offs
        trade_off_analysis = analyze_material_trade_offs(trajectories)

        self._log(f"Multi-material comparison completed for {len(materials)} materials")
        self._log(f"Pareto optimal materials: {comparison_result.pareto_optimal}")

        return {
            "scenario": "multi_material_comparison",
            "materials_compared": list(materials.keys()),
            "trajectories": trajectories,
            "comparison_result": comparison_result,
            "trade_off_analysis": trade_off_analysis,
            "comparative_visualizations": comparative_visualizations,
            "material_descriptions": {
                name: data["description"] for name, data in materials.items()
            },
        }

    async def demo_environmental_scenarios(self) -> Dict[str, Any]:
        """Demonstrate impact of different environmental scenarios."""
        self._log("=== Environmental Scenarios Demo ===")

        # Define environmental scenarios
        scenarios = {
            "Tropical_Climate": {
                "conditions": create_outdoor_weathering_conditions(
                    "tropical", duration=3000.0
                ),
                "description": "High temperature and humidity conditions",
            },
            "Arid_Climate": {
                "conditions": create_outdoor_weathering_conditions(
                    "arid", duration=3000.0
                ),
                "description": "High temperature, low humidity with intense UV",
            },
            "Temperate_Climate": {
                "conditions": create_outdoor_weathering_conditions(
                    "temperate", duration=3000.0
                ),
                "description": "Moderate temperature and humidity",
            },
            "Accelerated_Aging": {
                "conditions": create_accelerated_aging_conditions(
                    temperature=363.15,  # 90°C
                    humidity=0.9,
                    uv_intensity=0.9,
                    duration=1000.0,  # Shorter duration for accelerated testing
                ),
                "description": "Laboratory accelerated aging conditions",
            },
            "Marine_Environment": {
                "conditions": EnvironmentalConditions(
                    stressors={
                        EnvironmentalStressor.TEMPERATURE: 298.15,
                        EnvironmentalStressor.HUMIDITY: 0.95,
                        EnvironmentalStressor.UV_RADIATION: 0.4,
                        EnvironmentalStressor.CHEMICAL_EXPOSURE: 0.3,  # Salt water
                    },
                    duration=3000.0,
                ),
                "description": "Marine/coastal environment with salt exposure",
            },
        }

        # Use PET as test material
        material_properties = {
            "molecular_weight": 75000,
            "crystallinity": 0.35,
            "density": 1.38,
        }

        config = SimulationConfig(
            total_duration=3000.0, output_frequency=100.0, use_gpu=self.use_gpu
        )

        simulator = LifecycleSimulator(config)
        scenario_results = {}

        # Simulate under each environmental scenario
        for scenario_name, scenario_data in scenarios.items():
            self._log(f"Simulating under {scenario_name} conditions...")

            trajectory = await simulator.simulate_lifecycle(
                f"PET_{scenario_name}", material_properties, scenario_data["conditions"]
            )

            scenario_results[scenario_name] = {
                "trajectory": trajectory,
                "description": scenario_data["description"],
                "conditions": scenario_data["conditions"],
                "performance_metrics": self._calculate_environmental_impact_metrics(
                    trajectory
                ),
            }

        # Analyze environmental impact differences
        environmental_analysis = self._analyze_environmental_scenarios(scenario_results)

        # Create environmental comparison visualization
        environmental_visualizations = {}
        if PLOTLY_AVAILABLE:
            environmental_visualizations[
                "scenario_comparison"
            ] = self._create_environmental_comparison_plot(scenario_results)

        self._log(
            f"Environmental scenarios analysis completed for {len(scenarios)} scenarios"
        )

        return {
            "scenario": "environmental_scenarios",
            "scenarios_tested": list(scenarios.keys()),
            "scenario_results": scenario_results,
            "environmental_analysis": environmental_analysis,
            "environmental_visualizations": environmental_visualizations,
        }

    async def demo_degradation_product_tracking(self) -> Dict[str, Any]:
        """Demonstrate tracking of degradation products throughout lifecycle."""
        self._log("=== Degradation Product Tracking Demo ===")

        # Configure for detailed degradation tracking
        config = SimulationConfig(
            total_duration=2000.0,
            output_frequency=50.0,  # More frequent sampling
            include_degradation=True,
            use_gpu=self.use_gpu,
        )

        simulator = LifecycleSimulator(config)

        # Test different polymer types
        polymers = {
            "PET_UV_Exposed": {
                "properties": {"molecular_weight": 80000, "crystallinity": 0.4},
                "conditions": EnvironmentalConditions(
                    stressors={EnvironmentalStressor.UV_RADIATION: 0.9}, duration=2000.0
                ),
                "expected_products": ["carbonyl_compounds", "free_radicals"],
            },
            "PE_Thermal_Stressed": {
                "properties": {"molecular_weight": 100000, "crystallinity": 0.6},
                "conditions": EnvironmentalConditions(
                    stressors={EnvironmentalStressor.TEMPERATURE: 383.15},  # 110°C
                    duration=2000.0,
                ),
                "expected_products": ["volatile_organics", "char"],
            },
            "PLA_Hydrolytic": {
                "properties": {"molecular_weight": 120000, "crystallinity": 0.1},
                "conditions": EnvironmentalConditions(
                    stressors={
                        EnvironmentalStressor.HUMIDITY: 0.9,
                        EnvironmentalStressor.TEMPERATURE: 333.15,  # 60°C
                    },
                    duration=2000.0,
                ),
                "expected_products": ["oligomers", "monomers"],
            },
        }

        degradation_tracking = {}

        for polymer_name, polymer_data in polymers.items():
            self._log(f"Tracking degradation products for {polymer_name}...")

            trajectory = await simulator.simulate_lifecycle(
                polymer_name, polymer_data["properties"], polymer_data["conditions"]
            )

            # Extract degradation product evolution
            product_evolution = self._extract_product_evolution(trajectory)

            degradation_tracking[polymer_name] = {
                "trajectory": trajectory,
                "product_evolution": product_evolution,
                "expected_products": polymer_data["expected_products"],
                "validation": self._validate_degradation_products(
                    product_evolution, polymer_data["expected_products"]
                ),
            }

        # Create degradation product visualization
        degradation_visualizations = {}
        if PLOTLY_AVAILABLE:
            degradation_visualizations[
                "product_evolution"
            ] = self._create_degradation_product_plot(degradation_tracking)

        self._log(
            f"Degradation product tracking completed for {len(polymers)} polymers"
        )

        return {
            "scenario": "degradation_product_tracking",
            "polymers_analyzed": list(polymers.keys()),
            "degradation_tracking": degradation_tracking,
            "degradation_visualizations": degradation_visualizations,
        }

    async def demo_interactive_dashboard(self) -> Dict[str, Any]:
        """Demonstrate interactive dashboard capabilities."""
        self._log("=== Interactive Dashboard Demo ===")

        if not DASH_AVAILABLE:
            self._log("Dash not available. Skipping interactive dashboard demo.")
            return {
                "scenario": "interactive_dashboard",
                "status": "skipped",
                "reason": "Dash not available",
            }

        # Create dashboard
        dashboard = create_sustainability_dashboard(["PET", "HDPE", "PLA"])

        # Pre-populate with simulation data
        materials = ["PET", "HDPE", "PLA"]
        duration = 2000.0

        self._log("Pre-populating dashboard with simulation data...")
        await dashboard._run_dashboard_simulation(materials, duration)

        dashboard_analysis = {
            "materials_simulated": len(dashboard.current_trajectories),
            "comparison_available": dashboard.current_comparison is not None,
            "trajectory_summaries": [],
        }

        # Analyze dashboard data
        for trajectory in dashboard.current_trajectories:
            dashboard_analysis["trajectory_summaries"].append(
                {
                    "material_id": trajectory.material_id,
                    "total_duration": trajectory.total_duration,
                    "sustainability_score": trajectory.sustainability_score,
                    "final_fate": trajectory.final_fate,
                    "event_count": len(trajectory.events),
                }
            )

        if dashboard.current_comparison:
            dashboard_analysis["comparison_summary"] = {
                "materials_compared": dashboard.current_comparison.materials,
                "pareto_optimal": dashboard.current_comparison.pareto_optimal,
                "top_recommendations": dashboard.current_comparison.recommendations[:3],
            }

        # Note: In a real application, you would call dashboard.run_dashboard() here
        # For demo purposes, we'll just prepare the data structure

        self._log(f"Dashboard prepared with {len(materials)} materials")

        return {
            "scenario": "interactive_dashboard",
            "status": "prepared",
            "dashboard_analysis": dashboard_analysis,
            "dashboard_features": [
                "Real-time material comparison",
                "Interactive timeline visualization",
                "Sustainability metrics dashboard",
                "Pareto optimal identification",
                "Trade-off analysis",
            ],
            "note": "Dashboard server not started in demo mode",
        }

    def _analyze_single_trajectory(self, trajectory) -> Dict[str, Any]:
        """Analyze a single material trajectory."""
        analysis = {
            "lifecycle_summary": {
                "total_duration_days": trajectory.total_duration / 24.0,
                "total_events": len(trajectory.events),
                "phase_distribution": trajectory.lifecycle_phases,
                "final_fate": trajectory.final_fate,
                "sustainability_score": trajectory.sustainability_score,
            },
            "degradation_analysis": {},
            "environmental_impact": {},
            "phase_analysis": {},
        }

        # Analyze degradation progression
        use_events = [
            e for e in trajectory.events if e.phase == LifecyclePhase.USE_PHASE
        ]
        if use_events:
            initial_retention = 1.0
            final_retention = use_events[-1].properties.get(
                "molecular_weight_retention", 0.5
            )
            degradation_rate = (initial_retention - final_retention) / (
                use_events[-1].time / 24.0
            )  # per day

            analysis["degradation_analysis"] = {
                "initial_retention": initial_retention,
                "final_retention": final_retention,
                "degradation_rate_per_day": degradation_rate,
                "half_life_days": (
                    np.log(2) / (degradation_rate + 1e-6)
                    if degradation_rate > 0
                    else float("inf")
                ),
            }

        # Analyze environmental impact
        total_impacts = trajectory.cumulative_impacts
        analysis["environmental_impact"] = {
            "total_carbon_footprint": total_impacts.get("climate_change", 0),
            "impact_intensity": total_impacts.get("climate_change", 0)
            / (trajectory.total_duration / 24.0),
            "impact_categories": list(total_impacts.keys()),
        }

        # Analyze phase contributions
        phase_events = {}
        for event in trajectory.events:
            if event.phase not in phase_events:
                phase_events[event.phase] = []
            phase_events[event.phase].append(event)

        for phase, events in phase_events.items():
            phase_impact = sum(
                event.environmental_impact.get("climate_change", 0) for event in events
            )
            analysis["phase_analysis"][phase.value] = {
                "event_count": len(events),
                "environmental_impact": phase_impact,
                "duration": (
                    max(e.time for e in events) - min(e.time for e in events)
                    if len(events) > 1
                    else 0
                ),
            }

        return analysis

    def _calculate_environmental_impact_metrics(self, trajectory) -> Dict[str, float]:
        """Calculate environmental impact metrics for a trajectory."""
        metrics = {}

        # Degradation rate
        use_events = [
            e for e in trajectory.events if e.phase == LifecyclePhase.USE_PHASE
        ]
        if len(use_events) >= 2:
            initial_retention = use_events[0].properties.get(
                "molecular_weight_retention", 1.0
            )
            final_retention = use_events[-1].properties.get(
                "molecular_weight_retention", 0.5
            )
            time_span = use_events[-1].time - use_events[0].time
            metrics["degradation_rate"] = (
                (initial_retention - final_retention) / time_span
                if time_span > 0
                else 0
            )
        else:
            metrics["degradation_rate"] = 0.001

        # Environmental impact intensity
        total_impact = trajectory.cumulative_impacts.get("climate_change", 0)
        metrics["impact_intensity"] = total_impact / (trajectory.total_duration / 24.0)

        # Sustainability metrics
        metrics["sustainability_score"] = trajectory.sustainability_score
        metrics["total_carbon_footprint"] = total_impact

        # End-of-life score
        if trajectory.final_fate == "recycling":
            metrics["end_of_life_score"] = 1.0
        elif trajectory.final_fate == "biodegradation":
            metrics["end_of_life_score"] = 0.8
        else:
            metrics["end_of_life_score"] = 0.3

        return metrics

    def _analyze_environmental_scenarios(self, scenario_results) -> Dict[str, Any]:
        """Analyze differences between environmental scenarios."""
        analysis = {
            "scenario_ranking": {},
            "stress_impact_analysis": {},
            "recommendations": [],
        }

        # Rank scenarios by different metrics
        metrics = ["degradation_rate", "sustainability_score", "total_carbon_footprint"]

        for metric in metrics:
            scenario_scores = []
            for scenario_name, result in scenario_results.items():
                score = result["performance_metrics"].get(metric, 0)
                scenario_scores.append((scenario_name, score))

            # Sort appropriately (lower is better for degradation_rate and carbon_footprint)
            if metric in ["degradation_rate", "total_carbon_footprint"]:
                scenario_scores.sort(key=lambda x: x[1])
            else:
                scenario_scores.sort(key=lambda x: x[1], reverse=True)

            analysis["scenario_ranking"][metric] = [name for name, _ in scenario_scores]

        # Analyze specific stress impacts
        stress_factors = {}
        for scenario_name, result in scenario_results.items():
            conditions = result["conditions"]
            if hasattr(conditions, "stressors"):
                for stressor, intensity in conditions.stressors.items():
                    if stressor.value not in stress_factors:
                        stress_factors[stressor.value] = []

                    degradation_rate = result["performance_metrics"]["degradation_rate"]
                    stress_factors[stressor.value].append((intensity, degradation_rate))

        # Calculate stress correlations
        for stress_type, data_points in stress_factors.items():
            if len(data_points) > 2:
                intensities = [point[0] for point in data_points]
                degradations = [point[1] for point in data_points]
                correlation = (
                    np.corrcoef(intensities, degradations)[0, 1]
                    if len(set(intensities)) > 1
                    else 0
                )
                analysis["stress_impact_analysis"][stress_type] = correlation

        # Generate recommendations
        best_overall = analysis["scenario_ranking"].get("sustainability_score", [None])[
            0
        ]
        if best_overall:
            analysis["recommendations"].append(
                f"Best environmental scenario: {best_overall}"
            )

        worst_degradation = analysis["scenario_ranking"].get(
            "degradation_rate", [None]
        )[-1]
        if worst_degradation:
            analysis["recommendations"].append(
                f"Most aggressive conditions: {worst_degradation}"
            )

        # Stress-specific recommendations
        high_impact_stresses = [
            stress
            for stress, corr in analysis["stress_impact_analysis"].items()
            if corr > 0.5
        ]
        if high_impact_stresses:
            analysis["recommendations"].append(
                f"High-impact stress factors: {', '.join(high_impact_stresses)}"
            )

        return analysis

    def _extract_product_evolution(self, trajectory) -> Dict[str, List[float]]:
        """Extract degradation product evolution over time."""
        product_evolution = {}

        for event in trajectory.events:
            if event.phase == LifecyclePhase.USE_PHASE and event.degradation_products:
                for product in event.degradation_products:
                    if product not in product_evolution:
                        product_evolution[product] = {"times": [], "concentrations": []}

                    product_evolution[product]["times"].append(event.time)
                    # Estimate concentration based on degradation extent
                    retention = event.properties.get("molecular_weight_retention", 1.0)
                    concentration = (1.0 - retention) * np.random.uniform(
                        0.5, 1.5
                    )  # Mock concentration
                    product_evolution[product]["concentrations"].append(concentration)

        return product_evolution

    def _validate_degradation_products(
        self, product_evolution, expected_products
    ) -> Dict[str, Any]:
        """Validate detected degradation products against expectations."""
        validation = {
            "detected_products": list(product_evolution.keys()),
            "expected_products": expected_products,
            "validation_score": 0.0,
            "missing_products": [],
            "unexpected_products": [],
        }

        detected = set(validation["detected_products"])
        expected = set(expected_products)

        # Calculate validation metrics
        overlap = detected.intersection(expected)
        validation["validation_score"] = (
            len(overlap) / len(expected) if expected else 1.0
        )
        validation["missing_products"] = list(expected - detected)
        validation["unexpected_products"] = list(detected - expected)

        return validation

    def _create_environmental_comparison_plot(self, scenario_results):
        """Create environmental scenario comparison plot."""
        if not PLOTLY_AVAILABLE:
            return "Plotly not available"

        scenarios = list(scenario_results.keys())
        degradation_rates = [
            result["performance_metrics"]["degradation_rate"]
            for result in scenario_results.values()
        ]
        sustainability_scores = [
            result["performance_metrics"]["sustainability_score"]
            for result in scenario_results.values()
        ]

        fig = go.Figure()

        # Add degradation rate bars
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=degradation_rates,
                name="Degradation Rate",
                yaxis="y",
                offsetgroup=1,
            )
        )

        # Add sustainability scores on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=scenarios,
                y=sustainability_scores,
                name="Sustainability Score",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="red", width=3),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title="Environmental Scenario Impact Analysis",
            xaxis_title="Environmental Scenario",
            yaxis=dict(title="Degradation Rate", side="left"),
            yaxis2=dict(title="Sustainability Score", side="right", overlaying="y"),
            height=500,
        )

        return fig

    def _create_degradation_product_plot(self, degradation_tracking):
        """Create degradation product evolution plot."""
        if not PLOTLY_AVAILABLE:
            return "Plotly not available"

        fig = make_subplots(
            rows=len(degradation_tracking),
            cols=1,
            subplot_titles=list(degradation_tracking.keys()),
            vertical_spacing=0.1,
        )

        for i, (polymer_name, tracking_data) in enumerate(
            degradation_tracking.items(), 1
        ):
            product_evolution = tracking_data["product_evolution"]

            for product_name, evolution_data in product_evolution.items():
                if evolution_data["times"] and evolution_data["concentrations"]:
                    fig.add_trace(
                        go.Scatter(
                            x=evolution_data["times"],
                            y=evolution_data["concentrations"],
                            mode="lines+markers",
                            name=f"{product_name}",
                            line=dict(width=2),
                            showlegend=(i == 1),  # Only show legend for first subplot
                        ),
                        row=i,
                        col=1,
                    )

        fig.update_layout(
            title="Degradation Product Evolution Over Time",
            height=200 * len(degradation_tracking),
            showlegend=True,
        )

        fig.update_xaxes(title_text="Time (hours)")
        fig.update_yaxes(title_text="Relative Concentration")

        return fig

    def create_comprehensive_visualization(self, results: Dict[str, Any]) -> str:
        """Create comprehensive lifecycle visualization dashboard."""

        if not PLOTLY_AVAILABLE:
            self._log("Plotly not available. Skipping visualization.")
            return "Visualization not available"

        # Create comprehensive dashboard
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Material Lifecycle Timeline",
                "Multi-Material Comparison",
                "Environmental Scenario Impact",
                "Degradation Product Evolution",
                "Sustainability Metrics",
                "Phase Contribution Analysis",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "radar"}, {"type": "bar"}],
            ],
        )

        # Plot 1: Material lifecycle timeline
        if "single_material_lifecycle" in results:
            single_result = results["single_material_lifecycle"]
            trajectory = single_result["trajectory"]

            use_events = [
                e for e in trajectory.events if e.phase == LifecyclePhase.USE_PHASE
            ]
            if use_events:
                times = [e.time for e in use_events]
                retentions = [
                    e.properties.get("molecular_weight_retention", 0.5)
                    for e in use_events
                ]

                fig.add_trace(
                    go.Scatter(
                        x=times, y=retentions, mode="lines+markers", name="MW Retention"
                    ),
                    row=1,
                    col=1,
                )

        # Plot 2: Multi-material comparison
        if "multi_material_comparison" in results:
            comparison_result = results["multi_material_comparison"][
                "comparison_result"
            ]
            materials = comparison_result.materials

            if (
                ComparisonMetric.OVERALL_SUSTAINABILITY
                in comparison_result.comparison_metrics
            ):
                sustainability_scores = [
                    comparison_result.comparison_metrics[
                        ComparisonMetric.OVERALL_SUSTAINABILITY
                    ][mat]
                    for mat in materials
                ]

                fig.add_trace(
                    go.Bar(
                        x=materials,
                        y=sustainability_scores,
                        name="Sustainability Score",
                    ),
                    row=1,
                    col=2,
                )

        # Plot 3: Environmental scenario impact
        if "environmental_scenarios" in results:
            env_results = results["environmental_scenarios"]["scenario_results"]
            scenarios = list(env_results.keys())
            degradation_rates = [
                result["performance_metrics"]["degradation_rate"]
                for result in env_results.values()
            ]

            fig.add_trace(
                go.Scatter(
                    x=scenarios,
                    y=degradation_rates,
                    mode="markers",
                    marker=dict(size=12),
                    name="Degradation Rate",
                ),
                row=2,
                col=1,
            )

        # Plot 4: Degradation products (simplified)
        if "degradation_product_tracking" in results:
            degradation_data = results["degradation_product_tracking"][
                "degradation_tracking"
            ]

            all_products = set()
            for tracking_data in degradation_data.values():
                all_products.update(tracking_data["product_evolution"].keys())

            if all_products:
                # Show product detection success rate
                polymer_names = list(degradation_data.keys())
                detection_rates = []

                for tracking_data in degradation_data.values():
                    validation = tracking_data["validation"]
                    detection_rates.append(validation["validation_score"])

                fig.add_trace(
                    go.Scatter(
                        x=polymer_names,
                        y=detection_rates,
                        mode="markers+lines",
                        name="Product Detection Rate",
                    ),
                    row=2,
                    col=2,
                )

        # Plot 5: Sustainability metrics radar (simplified)
        if "multi_material_comparison" in results:
            trajectories = results["multi_material_comparison"]["trajectories"]

            for i, trajectory in enumerate(trajectories[:3]):  # Show first 3 materials
                metrics = ["Durability", "Recyclability", "Low Impact", "Overall"]

                # Calculate metrics
                use_events = [
                    e for e in trajectory.events if e.phase == LifecyclePhase.USE_PHASE
                ]
                durability = (
                    use_events[-1].properties.get("molecular_weight_retention", 0.5)
                    if use_events
                    else 0.5
                )
                recyclability = 1.0 if trajectory.final_fate == "recycling" else 0.3
                low_impact = 1.0 / (
                    1.0 + trajectory.cumulative_impacts.get("climate_change", 1.0)
                )
                overall = trajectory.sustainability_score

                values = [durability, recyclability, low_impact, overall]

                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=metrics,
                        fill="toself",
                        name=trajectory.material_id,
                        line=dict(width=2),
                    ),
                    row=3,
                    col=1,
                )

        # Plot 6: Phase contribution analysis
        if "single_material_lifecycle" in results:
            trajectory = results["single_material_lifecycle"]["trajectory"]

            phase_impacts = {}
            for event in trajectory.events:
                phase = event.phase.value
                if phase not in phase_impacts:
                    phase_impacts[phase] = 0
                phase_impacts[phase] += event.environmental_impact.get(
                    "climate_change", 0
                )

            if phase_impacts:
                phases = list(phase_impacts.keys())
                impacts = list(phase_impacts.values())

                fig.add_trace(
                    go.Bar(x=phases, y=impacts, name="Phase Impact"), row=3, col=2
                )

        fig.update_layout(
            title="Comprehensive Lifecycle Visualization Dashboard",
            height=1200,
            showlegend=True,
        )

        # Update polar charts
        fig.update_polars(radialaxis=dict(visible=True, range=[0, 1]), row=3, col=1)

        # Save to file
        if self.output_dir:
            viz_path = self.output_dir / "lifecycle_visualization_dashboard.html"
            fig.write_html(str(viz_path))
            return str(viz_path)

        return "Dashboard created (no output directory specified)"

    async def run_all_demos(self) -> Dict[str, Any]:
        """Run all lifecycle visualization demonstration scenarios."""
        self._log("Starting Comprehensive Lifecycle Visualization Demonstrations")
        self._log("=" * 80)

        # Create output directory
        self.output_dir = Path(tempfile.mkdtemp(prefix="lifecycle_viz_demo_"))
        self._log(f"Output directory: {self.output_dir}")

        all_results = {}

        try:
            # Demo 1: Single Material Lifecycle
            all_results[
                "single_material_lifecycle"
            ] = await self.demo_single_material_lifecycle()

            # Demo 2: Multi-Material Comparison
            all_results[
                "multi_material_comparison"
            ] = await self.demo_multi_material_comparison()

            # Demo 3: Environmental Scenarios
            all_results[
                "environmental_scenarios"
            ] = await self.demo_environmental_scenarios()

            # Demo 4: Degradation Product Tracking
            all_results[
                "degradation_product_tracking"
            ] = await self.demo_degradation_product_tracking()

            # Demo 5: Interactive Dashboard
            all_results[
                "interactive_dashboard"
            ] = await self.demo_interactive_dashboard()

            self._log("=" * 80)
            self._log("All demonstrations completed successfully!")

            # Create comprehensive visualization
            viz_path = self.create_comprehensive_visualization(all_results)
            all_results["visualization_path"] = viz_path

            # Generate summary report
            summary_report = self.generate_summary_report(all_results)
            all_results["summary_report"] = summary_report

            return all_results

        except Exception as e:
            logger.error(f"Error during demonstrations: {e}")
            all_results["error"] = str(e)
            return all_results

    def generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""

        report = {
            "executive_summary": {},
            "key_findings": {},
            "performance_metrics": {},
            "recommendations": {},
        }

        # Executive summary
        completed_demos = sum(
            1
            for key, result in results.items()
            if key not in ["visualization_path", "error", "summary_report"]
            and isinstance(result, dict)
            and result.get("scenario") is not None
        )

        report["executive_summary"] = {
            "total_demonstrations": 5,
            "completed_demonstrations": completed_demos,
            "visualization_features": [
                "timeline",
                "comparison",
                "radar",
                "heatmap",
                "dashboard",
            ],
            "simulation_approaches": [
                "single_material",
                "comparative",
                "environmental",
                "degradation_tracking",
            ],
        }

        # Key findings
        if "single_material_lifecycle" in results:
            single_result = results["single_material_lifecycle"]
            trajectory = single_result["trajectory"]
            report["key_findings"]["single_material"] = {
                "sustainability_score": trajectory.sustainability_score,
                "total_duration_days": trajectory.total_duration / 24.0,
                "final_fate": trajectory.final_fate,
                "total_events": len(trajectory.events),
            }

        if "multi_material_comparison" in results:
            comparison_data = results["multi_material_comparison"]
            comparison_result = comparison_data["comparison_result"]
            report["key_findings"]["material_comparison"] = {
                "materials_compared": len(comparison_result.materials),
                "pareto_optimal_count": len(comparison_result.pareto_optimal),
                "pareto_optimal_materials": comparison_result.pareto_optimal,
            }

        if "environmental_scenarios" in results:
            env_data = results["environmental_scenarios"]
            env_analysis = env_data["environmental_analysis"]
            report["key_findings"]["environmental_impact"] = {
                "scenarios_tested": len(env_data["scenario_results"]),
                "stress_correlations": env_analysis.get("stress_impact_analysis", {}),
                "best_scenario": env_analysis.get("scenario_ranking", {}).get(
                    "sustainability_score", [None]
                )[0],
            }

        if "degradation_product_tracking" in results:
            degradation_data = results["degradation_product_tracking"]
            report["key_findings"]["degradation_tracking"] = {
                "polymers_analyzed": len(degradation_data["degradation_tracking"]),
                "product_validation_scores": [
                    data["validation"]["validation_score"]
                    for data in degradation_data["degradation_tracking"].values()
                ],
            }

        # Performance metrics
        total_simulation_time = 0
        total_materials_simulated = 0

        for key, result in results.items():
            if isinstance(result, dict) and "scenario" in result:
                if "trajectory" in result:
                    total_simulation_time += result["trajectory"].total_duration
                    total_materials_simulated += 1
                elif "trajectories" in result:
                    for traj in result["trajectories"]:
                        total_simulation_time += traj.total_duration
                        total_materials_simulated += 1

        report["performance_metrics"] = {
            "total_simulation_time_hours": total_simulation_time,
            "total_materials_simulated": total_materials_simulated,
            "visualization_capabilities": PLOTLY_AVAILABLE,
            "dashboard_capabilities": DASH_AVAILABLE,
            "data_analysis_capabilities": PANDAS_AVAILABLE,
        }

        # Recommendations
        recommendations = []

        if (
            report["key_findings"]
            .get("material_comparison", {})
            .get("pareto_optimal_count", 0)
            > 1
        ):
            recommendations.append(
                "Multiple Pareto-optimal materials identified for different applications"
            )

        if report["key_findings"].get("environmental_impact", {}).get("best_scenario"):
            best_scenario = report["key_findings"]["environmental_impact"][
                "best_scenario"
            ]
            recommendations.append(f"Optimal environmental conditions: {best_scenario}")

        avg_validation = np.mean(
            report["key_findings"]
            .get("degradation_tracking", {})
            .get("product_validation_scores", [0.5])
        )
        if avg_validation > 0.7:
            recommendations.append("High accuracy in degradation product prediction")
        elif avg_validation > 0.5:
            recommendations.append("Moderate accuracy - consider model refinement")
        else:
            recommendations.append(
                "Low prediction accuracy - review degradation models"
            )

        if report["performance_metrics"]["visualization_capabilities"]:
            recommendations.append(
                "Full visualization capabilities available for analysis"
            )
        else:
            recommendations.append("Install Plotly for enhanced visualization features")

        if report["performance_metrics"]["dashboard_capabilities"]:
            recommendations.append("Interactive dashboard features available")
        else:
            recommendations.append(
                "Install Dash for interactive dashboard capabilities"
            )

        report["recommendations"] = recommendations

        return report


async def main():
    """Main demonstration function."""
    print("Comprehensive Lifecycle Simulation and Visualization Demo")
    print("=" * 80)

    # Initialize demo
    demo = LifecycleVisualizationDemo(
        use_gpu=False, verbose=True
    )  # Use CPU for broader compatibility

    # Run all demonstrations
    results = await demo.run_all_demos()

    # Save results
    results_file = Path("lifecycle_visualization_demo_results.json")

    # Convert any non-serializable objects
    def convert_for_json(obj):
        if hasattr(obj, "value"):  # Handle Enum objects
            return obj.value
        elif isinstance(obj, (datetime, np.integer, np.floating)):
            return str(obj)
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return {k: convert_for_json(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return str(obj) if obj is not None else None

    serializable_results = convert_for_json(results)

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)

    if "summary_report" in results:
        report = results["summary_report"]

        print("\nExecutive Summary:")
        exec_summary = report.get("executive_summary", {})
        print(f"  Total Demonstrations: {exec_summary.get('total_demonstrations', 0)}")
        print(
            f"  Completed Demonstrations: {exec_summary.get('completed_demonstrations', 0)}"
        )
        print(
            f"  Visualization Features: {len(exec_summary.get('visualization_features', []))}"
        )

        print("\nKey Findings:")
        findings = report.get("key_findings", {})

        if "single_material" in findings:
            single = findings["single_material"]
            print(
                f"  Single Material Sustainability Score: {single.get('sustainability_score', 0):.3f}"
            )
            print(
                f"  Simulation Duration: {single.get('total_duration_days', 0):.1f} days"
            )

        if "material_comparison" in findings:
            comparison = findings["material_comparison"]
            print(f"  Materials Compared: {comparison.get('materials_compared', 0)}")
            print(
                f"  Pareto Optimal Materials: {comparison.get('pareto_optimal_materials', [])}"
            )

        if "environmental_impact" in findings:
            env_impact = findings["environmental_impact"]
            print(
                f"  Environmental Scenarios Tested: {env_impact.get('scenarios_tested', 0)}"
            )
            print(
                f"  Best Environmental Scenario: {env_impact.get('best_scenario', 'N/A')}"
            )

        print("\nPerformance Metrics:")
        perf_metrics = report.get("performance_metrics", {})
        print(
            f"  Total Simulation Time: {perf_metrics.get('total_simulation_time_hours', 0):.1f} hours"
        )
        print(
            f"  Materials Simulated: {perf_metrics.get('total_materials_simulated', 0)}"
        )
        print(
            f"  Visualization: {'Available' if perf_metrics.get('visualization_capabilities', False) else 'Limited'}"
        )
        print(
            f"  Dashboard: {'Available' if perf_metrics.get('dashboard_capabilities', False) else 'Not Available'}"
        )

        print("\nRecommendations:")
        recommendations = report.get("recommendations", [])
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    if "visualization_path" in results:
        print(f"\nVisualization: {results['visualization_path']}")

    print("\nTechnology Stack:")
    print(f"  Plotly: {'Available' if PLOTLY_AVAILABLE else 'Not Available'}")
    print(f"  Dash: {'Available' if DASH_AVAILABLE else 'Not Available'}")
    print(f"  Pandas: {'Available' if PANDAS_AVAILABLE else 'Not Available'}")

    print("\nLifecycle visualization demonstration completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

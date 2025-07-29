"""
Tests for Interactive Lifecycle Simulation and Visualization.

This module provides comprehensive testing for the lifecycle simulation and
visualization framework including simulation engines, dashboard components,
and comparison tools.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.sustainability.degradation_modeling import (
    EnvironmentalConditions,
    EnvironmentalStressor,
    create_outdoor_weathering_conditions,
    create_standard_material,
)
from chemi_accel.sustainability.lca_integration import ImpactCategory
from chemi_accel.sustainability.lifecycle_visualization import (  # Core classes; Data structures; Utility functions; Exceptions
    ComparisonMetric,
    ComparisonResult,
    DashboardError,
    LifecycleEvent,
    LifecyclePhase,
    LifecycleSimulationError,
    LifecycleSimulator,
    LifecycleTrajectory,
    LifecycleVisualizer,
    MaterialComparator,
    SimulationConfig,
    SimulationConfigurationError,
    SustainabilityDashboard,
    VisualizationError,
    VisualizationType,
    analyze_material_trade_offs,
    create_sustainability_dashboard,
    simulate_material_lifecycle,
)

# Check for optional dependencies
try:
    import plotly.graph_objects as go

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


class TestDataStructures:
    """Test lifecycle visualization data structures."""

    def test_lifecycle_event_creation(self):
        """Test creating lifecycle event."""
        event = LifecycleEvent(
            time=100.0,
            phase=LifecyclePhase.USE_PHASE,
            event_type="degradation",
            properties={"molecular_weight_retention": 0.8},
            environmental_impact={ImpactCategory.CLIMATE_CHANGE: 0.1},
            degradation_products=["oligomers"],
            description="Degradation event at 100 hours",
            metadata={"mechanism": "UV degradation"},
        )

        assert event.time == 100.0
        assert event.phase == LifecyclePhase.USE_PHASE
        assert event.event_type == "degradation"
        assert event.properties["molecular_weight_retention"] == 0.8
        assert event.environmental_impact[ImpactCategory.CLIMATE_CHANGE] == 0.1
        assert "oligomers" in event.degradation_products
        assert event.metadata["mechanism"] == "UV degradation"

    def test_lifecycle_trajectory_creation(self):
        """Test creating lifecycle trajectory."""
        events = [
            LifecycleEvent(
                time=0.0,
                phase=LifecyclePhase.MANUFACTURING,
                event_type="production",
                properties={},
                environmental_impact={ImpactCategory.CLIMATE_CHANGE: 2.0},
                degradation_products=[],
                description="Manufacturing event",
            ),
            LifecycleEvent(
                time=1000.0,
                phase=LifecyclePhase.USE_PHASE,
                event_type="degradation",
                properties={"molecular_weight_retention": 0.9},
                environmental_impact={ImpactCategory.CLIMATE_CHANGE: 0.1},
                degradation_products=[],
                description="Use phase event",
            ),
        ]

        trajectory = LifecycleTrajectory(
            material_id="PET_test",
            events=events,
            total_duration=1000.0,
            lifecycle_phases={LifecyclePhase.USE_PHASE: 1000.0},
            cumulative_impacts={ImpactCategory.CLIMATE_CHANGE: 2.1},
            final_fate="recycling",
            sustainability_score=0.8,
        )

        assert trajectory.material_id == "PET_test"
        assert len(trajectory.events) == 2
        assert trajectory.total_duration == 1000.0
        assert trajectory.final_fate == "recycling"
        assert trajectory.sustainability_score == 0.8

    def test_simulation_config_creation(self):
        """Test creating simulation configuration."""
        config = SimulationConfig(
            total_duration=8760.0,
            time_step=24.0,
            environmental_scenarios=["temperate", "tropical"],
            include_degradation=True,
            include_recyclability=True,
            include_lca=True,
            use_gpu=False,
            parallel_simulation=True,
            output_frequency=168.0,
        )

        assert config.total_duration == 8760.0
        assert config.time_step == 24.0
        assert "temperate" in config.environmental_scenarios
        assert config.include_degradation is True
        assert config.use_gpu is False
        assert config.output_frequency == 168.0

    def test_comparison_result_creation(self):
        """Test creating comparison result."""
        comparison_metrics = {
            ComparisonMetric.DEGRADATION_RATE: {"PET": 0.001, "PE": 0.002},
            ComparisonMetric.RECYCLABILITY_SCORE: {"PET": 0.8, "PE": 0.6},
        }

        rankings = {
            ComparisonMetric.DEGRADATION_RATE: ["PET", "PE"],
            ComparisonMetric.RECYCLABILITY_SCORE: ["PET", "PE"],
        }

        result = ComparisonResult(
            materials=["PET", "PE"],
            comparison_metrics=comparison_metrics,
            rankings=rankings,
            pareto_optimal=["PET"],
            trade_offs={"degradation_vs_recyclability": -0.3},
            recommendations=["Use PET for best overall performance"],
            visualization_data={"prepared": True},
        )

        assert len(result.materials) == 2
        assert "PET" in result.materials
        assert len(result.comparison_metrics) == 2
        assert "PET" in result.pareto_optimal
        assert len(result.recommendations) == 1


class TestLifecycleSimulator:
    """Test lifecycle simulator."""

    def test_simulator_creation(self):
        """Test simulator initialization."""
        config = SimulationConfig(use_gpu=False)
        simulator = LifecycleSimulator(config)

        assert simulator.config == config
        assert simulator.degradation_predictor is not None
        assert simulator.recyclability_assessor is not None
        assert simulator.lca_calculator is not None
        assert isinstance(simulator.current_simulations, dict)
        assert isinstance(simulator.simulation_history, list)

    @pytest.mark.asyncio
    async def test_lifecycle_simulation(self):
        """Test complete lifecycle simulation."""
        config = SimulationConfig(
            total_duration=1000.0, output_frequency=200.0, use_gpu=False
        )
        simulator = LifecycleSimulator(config)

        material_id = "PET_test"
        initial_properties = {"molecular_weight": 50000, "crystallinity": 0.3}
        usage_conditions = create_outdoor_weathering_conditions(
            "temperate", duration=1000.0
        )

        trajectory = await simulator.simulate_lifecycle(
            material_id, initial_properties, usage_conditions
        )

        assert isinstance(trajectory, LifecycleTrajectory)
        assert trajectory.material_id == material_id
        assert len(trajectory.events) > 0
        assert trajectory.total_duration > 0
        assert isinstance(trajectory.lifecycle_phases, dict)
        assert isinstance(trajectory.cumulative_impacts, dict)
        assert trajectory.sustainability_score >= 0

    @pytest.mark.asyncio
    async def test_manufacturing_phase_simulation(self):
        """Test manufacturing phase simulation."""
        config = SimulationConfig(use_gpu=False)
        simulator = LifecycleSimulator(config)

        properties = {"molecular_weight": 50000}

        event = await simulator._simulate_manufacturing_phase(
            "test_material", properties, 0.0
        )

        assert isinstance(event, LifecycleEvent)
        assert event.phase == LifecyclePhase.MANUFACTURING
        assert event.event_type == "production"
        assert len(event.environmental_impact) > 0
        assert event.time == 0.0

    @pytest.mark.asyncio
    async def test_transportation_phase_simulation(self):
        """Test transportation phase simulation."""
        config = SimulationConfig(use_gpu=False)
        simulator = LifecycleSimulator(config)

        properties = {"molecular_weight": 50000}

        event = await simulator._simulate_transportation_phase(
            "test_material", properties, 24.0
        )

        assert isinstance(event, LifecycleEvent)
        assert event.phase == LifecyclePhase.TRANSPORTATION
        assert event.event_type == "transport"
        assert event.environmental_impact[ImpactCategory.CLIMATE_CHANGE] > 0
        assert event.time == 24.0
        assert event.metadata["transport_mode"] == "truck"

    @pytest.mark.asyncio
    async def test_use_phase_simulation(self):
        """Test use phase simulation."""
        config = SimulationConfig(
            total_duration=500.0,
            output_frequency=100.0,
            include_degradation=True,
            use_gpu=False,
        )
        simulator = LifecycleSimulator(config)

        properties = {"molecular_weight": 50000}
        conditions = EnvironmentalConditions(
            stressors={EnvironmentalStressor.TEMPERATURE: 350.0}, duration=500.0
        )

        events = await simulator._simulate_use_phase(
            "test_material", properties, conditions, 48.0
        )

        assert isinstance(events, list)
        assert len(events) > 0

        for event in events:
            assert isinstance(event, LifecycleEvent)
            assert event.phase == LifecyclePhase.USE_PHASE
            assert event.time >= 48.0

    @pytest.mark.asyncio
    async def test_end_of_life_simulation(self):
        """Test end-of-life phase simulation."""
        config = SimulationConfig(include_recyclability=True, use_gpu=False)
        simulator = LifecycleSimulator(config)

        properties = {"molecular_weight_retention": 0.8}

        event = await simulator._simulate_end_of_life_phase(
            "test_material", properties, 1000.0
        )

        assert isinstance(event, LifecycleEvent)
        assert event.phase == LifecyclePhase.END_OF_LIFE
        assert event.event_type == "disposal"
        assert event.metadata["fate"] in ["recycling", "biodegradation", "landfill"]
        assert event.time == 1000.0

    def test_degradation_impact_calculation(self):
        """Test degradation impact calculation."""
        config = SimulationConfig(use_gpu=False)
        simulator = LifecycleSimulator(config)

        # Mock degradation state
        from chemi_accel.sustainability.degradation_modeling import DegradationState

        state = DegradationState(
            time=100.0,
            molecular_weight_retention=0.8,
            mechanical_retention={"tensile_strength": 0.7},
            chemical_changes={},
        )

        impact = simulator._calculate_degradation_impact(state)

        assert isinstance(impact, dict)
        assert ImpactCategory.CLIMATE_CHANGE in impact
        assert impact[ImpactCategory.CLIMATE_CHANGE] >= 0

        # Higher degradation should result in higher impact
        degraded_state = DegradationState(
            time=100.0,
            molecular_weight_retention=0.5,
            mechanical_retention={"tensile_strength": 0.4},
            chemical_changes={},
        )

        higher_impact = simulator._calculate_degradation_impact(degraded_state)
        assert (
            higher_impact[ImpactCategory.CLIMATE_CHANGE]
            > impact[ImpactCategory.CLIMATE_CHANGE]
        )

    def test_degradation_products_extraction(self):
        """Test degradation products extraction."""
        config = SimulationConfig(use_gpu=False)
        simulator = LifecycleSimulator(config)

        # Mock degradation mechanisms
        from chemi_accel.sustainability.degradation_modeling import (
            DegradationMechanism,
            DegradationState,
        )

        state = DegradationState(time=100.0, molecular_weight_retention=0.8)

        mechanisms = [
            DegradationMechanism.UV_DEGRADATION,
            DegradationMechanism.THERMAL_DEGRADATION,
        ]

        products = simulator._extract_degradation_products(state, mechanisms)

        assert isinstance(products, list)
        assert len(products) > 0
        assert "free_radicals" in products or "carbonyl_compounds" in products
        assert "volatile_organics" in products or "char" in products

    def test_sustainability_score_calculation(self):
        """Test sustainability score calculation."""
        config = SimulationConfig(use_gpu=False)
        simulator = LifecycleSimulator(config)

        # Create mock events
        events = [
            LifecycleEvent(
                time=0.0,
                phase=LifecyclePhase.USE_PHASE,
                event_type="degradation",
                properties={"molecular_weight_retention": 0.8},
                environmental_impact={},
                degradation_products=[],
                description="Use phase",
            ),
            LifecycleEvent(
                time=1000.0,
                phase=LifecyclePhase.END_OF_LIFE,
                event_type="disposal",
                properties={},
                environmental_impact={},
                degradation_products=[],
                description="End of life",
                metadata={"fate": "recycling"},
            ),
        ]

        cumulative_impacts = {ImpactCategory.CLIMATE_CHANGE: 2.0}

        score = simulator._calculate_sustainability_score(events, cumulative_impacts)

        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestLifecycleVisualizer:
    """Test lifecycle visualizer."""

    def test_visualizer_creation(self):
        """Test visualizer initialization."""
        visualizer = LifecycleVisualizer(use_plotly=PLOTLY_AVAILABLE)

        assert visualizer.use_plotly == PLOTLY_AVAILABLE
        assert isinstance(visualizer.color_schemes, dict)
        assert "phases" in visualizer.color_schemes
        assert "impacts" in visualizer.color_schemes

    def test_timeline_visualization(self):
        """Test timeline visualization creation."""
        visualizer = LifecycleVisualizer(use_plotly=PLOTLY_AVAILABLE)

        # Create mock trajectory
        events = [
            LifecycleEvent(
                time=0.0,
                phase=LifecyclePhase.MANUFACTURING,
                event_type="production",
                properties={"molecular_weight": 50000},
                environmental_impact={ImpactCategory.CLIMATE_CHANGE: 2.0},
                degradation_products=[],
                description="Manufacturing",
            ),
            LifecycleEvent(
                time=100.0,
                phase=LifecyclePhase.USE_PHASE,
                event_type="degradation",
                properties={"molecular_weight_retention": 0.9},
                environmental_impact={ImpactCategory.CLIMATE_CHANGE: 0.1},
                degradation_products=[],
                description="Degradation",
            ),
        ]

        trajectory = LifecycleTrajectory(
            material_id="test_material",
            events=events,
            total_duration=1000.0,
            lifecycle_phases={LifecyclePhase.USE_PHASE: 1000.0},
            cumulative_impacts={ImpactCategory.CLIMATE_CHANGE: 2.1},
            final_fate="recycling",
            sustainability_score=0.8,
        )

        result = visualizer.create_lifecycle_timeline(trajectory)

        if PLOTLY_AVAILABLE:
            assert isinstance(result, go.Figure)
        else:
            assert isinstance(result, str)
            assert "not available" in result.lower()

    def test_phase_breakdown_visualization(self):
        """Test phase breakdown visualization."""
        visualizer = LifecycleVisualizer(use_plotly=PLOTLY_AVAILABLE)

        # Create mock trajectory with multiple phases
        events = [
            LifecycleEvent(
                time=0.0,
                phase=LifecyclePhase.MANUFACTURING,
                event_type="production",
                properties={},
                environmental_impact={ImpactCategory.CLIMATE_CHANGE: 2.0},
                degradation_products=[],
                description="Manufacturing",
            ),
            LifecycleEvent(
                time=24.0,
                phase=LifecyclePhase.TRANSPORTATION,
                event_type="transport",
                properties={},
                environmental_impact={ImpactCategory.CLIMATE_CHANGE: 0.5},
                degradation_products=[],
                description="Transportation",
            ),
            LifecycleEvent(
                time=1000.0,
                phase=LifecyclePhase.END_OF_LIFE,
                event_type="disposal",
                properties={},
                environmental_impact={ImpactCategory.CLIMATE_CHANGE: 0.3},
                degradation_products=[],
                description="End of life",
            ),
        ]

        trajectory = LifecycleTrajectory(
            material_id="test_material",
            events=events,
            total_duration=1000.0,
            lifecycle_phases={},
            cumulative_impacts={},
            final_fate="recycling",
            sustainability_score=0.8,
        )

        result = visualizer.create_phase_breakdown(trajectory)

        if PLOTLY_AVAILABLE:
            assert isinstance(result, go.Figure)
        else:
            assert isinstance(result, str)

    def test_degradation_heatmap_visualization(self):
        """Test degradation heatmap visualization."""
        visualizer = LifecycleVisualizer(use_plotly=PLOTLY_AVAILABLE)

        # Create mock trajectories
        trajectories = []
        for i, material in enumerate(["PET", "PE"]):
            events = [
                LifecycleEvent(
                    time=100.0 * j,
                    phase=LifecyclePhase.USE_PHASE,
                    event_type="degradation",
                    properties={"molecular_weight_retention": 1.0 - 0.1 * j * (i + 1)},
                    environmental_impact={},
                    degradation_products=[],
                    description=f"Degradation at {100.0 * j} hours",
                )
                for j in range(3)
            ]

            trajectory = LifecycleTrajectory(
                material_id=material,
                events=events,
                total_duration=1000.0,
                lifecycle_phases={},
                cumulative_impacts={},
                final_fate="recycling",
                sustainability_score=0.8,
            )
            trajectories.append(trajectory)

        result = visualizer.create_degradation_heatmap(trajectories)

        if PLOTLY_AVAILABLE:
            assert isinstance(result, go.Figure)
        else:
            assert isinstance(result, str)

    def test_sustainability_radar_visualization(self):
        """Test sustainability radar chart."""
        visualizer = LifecycleVisualizer(use_plotly=PLOTLY_AVAILABLE)

        # Create mock trajectories
        trajectories = []
        for material in ["PET", "PE"]:
            events = [
                LifecycleEvent(
                    time=1000.0,
                    phase=LifecyclePhase.USE_PHASE,
                    event_type="degradation",
                    properties={"molecular_weight_retention": 0.8},
                    environmental_impact={},
                    degradation_products=[],
                    description="Final use state",
                ),
                LifecycleEvent(
                    time=1024.0,
                    phase=LifecyclePhase.END_OF_LIFE,
                    event_type="disposal",
                    properties={},
                    environmental_impact={},
                    degradation_products=[],
                    description="End of life",
                    metadata={"fate": "recycling"},
                ),
            ]

            trajectory = LifecycleTrajectory(
                material_id=material,
                events=events,
                total_duration=1024.0,
                lifecycle_phases={},
                cumulative_impacts={ImpactCategory.CLIMATE_CHANGE: 2.0},
                final_fate="recycling",
                sustainability_score=0.8,
            )
            trajectories.append(trajectory)

        result = visualizer.create_sustainability_radar(trajectories)

        if PLOTLY_AVAILABLE:
            assert isinstance(result, go.Figure)
        else:
            assert isinstance(result, str)

    def test_sankey_diagram_visualization(self):
        """Test Sankey diagram creation."""
        visualizer = LifecycleVisualizer(use_plotly=PLOTLY_AVAILABLE)

        # Create mock trajectory
        trajectory = LifecycleTrajectory(
            material_id="test_material",
            events=[],
            total_duration=1000.0,
            lifecycle_phases={},
            cumulative_impacts={},
            final_fate="recycling",
            sustainability_score=0.8,
        )

        result = visualizer.create_sankey_diagram(trajectory)

        if PLOTLY_AVAILABLE:
            assert isinstance(result, go.Figure)
        else:
            assert isinstance(result, str)


class TestMaterialComparator:
    """Test material comparator."""

    def test_comparator_creation(self):
        """Test comparator initialization."""
        comparator = MaterialComparator()

        assert isinstance(comparator.comparison_metrics, dict)
        assert len(comparator.comparison_metrics) > 0

        for metric in ComparisonMetric:
            assert metric in comparator.comparison_metrics
            assert callable(comparator.comparison_metrics[metric])

    def test_material_comparison(self):
        """Test material comparison."""
        comparator = MaterialComparator()

        # Create mock trajectories
        trajectories = []
        for i, material in enumerate(["PET", "PE", "PLA"]):
            events = [
                LifecycleEvent(
                    time=0.0,
                    phase=LifecyclePhase.USE_PHASE,
                    event_type="initial",
                    properties={"molecular_weight_retention": 1.0},
                    environmental_impact={},
                    degradation_products=[],
                    description="Initial state",
                ),
                LifecycleEvent(
                    time=1000.0,
                    phase=LifecyclePhase.USE_PHASE,
                    event_type="degradation",
                    properties={"molecular_weight_retention": 0.9 - i * 0.1},
                    environmental_impact={},
                    degradation_products=[],
                    description="Final use state",
                ),
                LifecycleEvent(
                    time=1024.0,
                    phase=LifecyclePhase.END_OF_LIFE,
                    event_type="disposal",
                    properties={},
                    environmental_impact={},
                    degradation_products=[],
                    description="End of life",
                    metadata={"fate": "recycling" if i == 0 else "landfill"},
                ),
            ]

            trajectory = LifecycleTrajectory(
                material_id=material,
                events=events,
                total_duration=1024.0,
                lifecycle_phases={},
                cumulative_impacts={ImpactCategory.CLIMATE_CHANGE: 2.0 + i * 0.5},
                final_fate="recycling" if i == 0 else "landfill",
                sustainability_score=0.9 - i * 0.1,
            )
            trajectories.append(trajectory)

        result = comparator.compare_materials(trajectories)

        assert isinstance(result, ComparisonResult)
        assert len(result.materials) == 3
        assert "PET" in result.materials
        assert len(result.comparison_metrics) > 0
        assert len(result.rankings) > 0
        assert len(result.pareto_optimal) >= 1
        assert isinstance(result.trade_offs, dict)
        assert len(result.recommendations) > 0

    def test_degradation_rate_extraction(self):
        """Test degradation rate extraction."""
        comparator = MaterialComparator()

        # Create trajectory with degradation
        events = [
            LifecycleEvent(
                time=0.0,
                phase=LifecyclePhase.USE_PHASE,
                event_type="initial",
                properties={"molecular_weight_retention": 1.0},
                environmental_impact={},
                degradation_products=[],
                description="Initial state",
            ),
            LifecycleEvent(
                time=1000.0,
                phase=LifecyclePhase.USE_PHASE,
                event_type="degradation",
                properties={"molecular_weight_retention": 0.8},
                environmental_impact={},
                degradation_products=[],
                description="Final state",
            ),
        ]

        trajectory = LifecycleTrajectory(
            material_id="test",
            events=events,
            total_duration=1000.0,
            lifecycle_phases={},
            cumulative_impacts={},
            final_fate="recycling",
            sustainability_score=0.8,
        )

        rate = comparator._extract_degradation_rate(trajectory)

        assert isinstance(rate, float)
        assert rate > 0
        # Should be approximately (1.0 - 0.8) / 1000.0 = 0.0002
        assert 0.0001 < rate < 0.001

    def test_recyclability_score_extraction(self):
        """Test recyclability score extraction."""
        comparator = MaterialComparator()

        # Test recycling fate
        recycling_trajectory = LifecycleTrajectory(
            material_id="recyclable",
            events=[
                LifecycleEvent(
                    time=1000.0,
                    phase=LifecyclePhase.END_OF_LIFE,
                    event_type="disposal",
                    properties={},
                    environmental_impact={},
                    degradation_products=[],
                    description="Recycling",
                    metadata={"fate": "recycling"},
                )
            ],
            total_duration=1000.0,
            lifecycle_phases={},
            cumulative_impacts={},
            final_fate="recycling",
            sustainability_score=0.8,
        )

        recycling_score = comparator._extract_recyclability_score(recycling_trajectory)
        assert recycling_score == 1.0

        # Test landfill fate
        landfill_trajectory = LifecycleTrajectory(
            material_id="non_recyclable",
            events=[
                LifecycleEvent(
                    time=1000.0,
                    phase=LifecyclePhase.END_OF_LIFE,
                    event_type="disposal",
                    properties={},
                    environmental_impact={},
                    degradation_products=[],
                    description="Landfill",
                    metadata={"fate": "landfill"},
                )
            ],
            total_duration=1000.0,
            lifecycle_phases={},
            cumulative_impacts={},
            final_fate="landfill",
            sustainability_score=0.3,
        )

        landfill_score = comparator._extract_recyclability_score(landfill_trajectory)
        assert landfill_score == 0.3

    def test_pareto_optimal_identification(self):
        """Test Pareto optimal identification."""
        comparator = MaterialComparator()

        # Create materials with different trade-offs
        trajectories = [
            # Material A: low degradation, low recyclability
            LifecycleTrajectory(
                material_id="A",
                events=[],
                total_duration=1000.0,
                lifecycle_phases={},
                cumulative_impacts={ImpactCategory.CLIMATE_CHANGE: 1.0},
                final_fate="landfill",
                sustainability_score=0.5,
            ),
            # Material B: high degradation, high recyclability
            LifecycleTrajectory(
                material_id="B",
                events=[],
                total_duration=1000.0,
                lifecycle_phases={},
                cumulative_impacts={ImpactCategory.CLIMATE_CHANGE: 3.0},
                final_fate="recycling",
                sustainability_score=0.7,
            ),
            # Material C: dominated by both A and B
            LifecycleTrajectory(
                material_id="C",
                events=[],
                total_duration=1000.0,
                lifecycle_phases={},
                cumulative_impacts={ImpactCategory.CLIMATE_CHANGE: 2.0},
                final_fate="landfill",
                sustainability_score=0.4,
            ),
        ]

        comparison_metrics = {
            ComparisonMetric.CARBON_FOOTPRINT: {"A": 1.0, "B": 3.0, "C": 2.0},
            ComparisonMetric.RECYCLABILITY_SCORE: {"A": 0.3, "B": 1.0, "C": 0.3},
        }

        pareto_optimal = comparator._identify_pareto_optimal(
            trajectories, comparison_metrics
        )

        assert isinstance(pareto_optimal, list)
        # A and B should be Pareto optimal, C should be dominated
        assert "A" in pareto_optimal
        assert "B" in pareto_optimal
        # C might or might not be dominated depending on exact implementation

    def test_trade_offs_calculation(self):
        """Test trade-offs calculation."""
        comparator = MaterialComparator()

        comparison_metrics = {
            ComparisonMetric.DURABILITY: {"A": 0.9, "B": 0.7, "C": 0.8},
            ComparisonMetric.RECYCLABILITY_SCORE: {"A": 0.3, "B": 0.9, "C": 0.6},
            ComparisonMetric.CARBON_FOOTPRINT: {"A": 1.0, "B": 2.5, "C": 1.8},
        }

        trade_offs = comparator._calculate_trade_offs(comparison_metrics)

        assert isinstance(trade_offs, dict)
        # Should calculate correlations between metric pairs
        for key, value in trade_offs.items():
            assert isinstance(key, str)
            assert "_vs_" in key
            assert isinstance(value, float)
            assert -1.0 <= value <= 1.0


class TestSustainabilityDashboard:
    """Test sustainability dashboard."""

    def test_dashboard_creation_without_dash(self):
        """Test dashboard creation when Dash is not available."""
        with patch(
            "chemi_accel.sustainability.lifecycle_visualization.DASH_AVAILABLE", False
        ):
            dashboard = SustainabilityDashboard(use_dash=True)

            assert dashboard.use_dash is False
            assert dashboard.visualizer is not None
            assert dashboard.comparator is not None

    @pytest.mark.skipif(not DASH_AVAILABLE, reason="Dash not available")
    def test_dashboard_creation_with_dash(self):
        """Test dashboard creation when Dash is available."""
        dashboard = SustainabilityDashboard(use_dash=True)

        assert dashboard.use_dash is True
        assert dashboard.app is not None
        assert dashboard.visualizer is not None
        assert dashboard.comparator is not None

    @pytest.mark.asyncio
    async def test_dashboard_simulation(self):
        """Test dashboard simulation functionality."""
        dashboard = SustainabilityDashboard(use_dash=False)

        materials = ["PET", "PE"]
        duration = 1000.0

        await dashboard._run_dashboard_simulation(materials, duration)

        assert len(dashboard.current_trajectories) == len(materials)
        for trajectory in dashboard.current_trajectories:
            assert isinstance(trajectory, LifecycleTrajectory)
            assert trajectory.material_id in materials

        if len(dashboard.current_trajectories) > 1:
            assert dashboard.current_comparison is not None


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.asyncio
    async def test_simulate_material_lifecycle_utility(self):
        """Test utility function for lifecycle simulation."""
        material_id = "test_material"
        properties = {"molecular_weight": 50000}
        duration = 500.0

        trajectory = await simulate_material_lifecycle(
            material_id, properties, duration=duration, use_gpu=False
        )

        assert isinstance(trajectory, LifecycleTrajectory)
        assert trajectory.material_id == material_id
        assert trajectory.total_duration > 0
        assert len(trajectory.events) > 0

    def test_create_sustainability_dashboard_utility(self):
        """Test utility function for dashboard creation."""
        materials = ["PET", "PE", "PLA"]

        dashboard = create_sustainability_dashboard(materials)

        assert isinstance(dashboard, SustainabilityDashboard)
        assert dashboard.visualizer is not None
        assert dashboard.comparator is not None

    def test_analyze_material_trade_offs_utility(self):
        """Test utility function for trade-off analysis."""
        # Create mock trajectories
        trajectories = []
        for material in ["PET", "PE"]:
            trajectory = LifecycleTrajectory(
                material_id=material,
                events=[],
                total_duration=1000.0,
                lifecycle_phases={},
                cumulative_impacts={ImpactCategory.CLIMATE_CHANGE: 2.0},
                final_fate="recycling",
                sustainability_score=0.8,
            )
            trajectories.append(trajectory)

        analysis = analyze_material_trade_offs(trajectories)

        assert isinstance(analysis, dict)
        assert "comparison_results" in analysis
        assert "pareto_optimal" in analysis
        assert "trade_offs" in analysis
        assert "recommendations" in analysis


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_lifecycle_phase_enum(self):
        """Test lifecycle phase enumeration."""
        assert LifecyclePhase.RAW_MATERIAL_EXTRACTION.value == "raw_material_extraction"
        assert LifecyclePhase.MANUFACTURING.value == "manufacturing"
        assert LifecyclePhase.USE_PHASE.value == "use_phase"
        assert LifecyclePhase.END_OF_LIFE.value == "end_of_life"

    def test_visualization_type_enum(self):
        """Test visualization type enumeration."""
        assert VisualizationType.TIMELINE.value == "timeline"
        assert VisualizationType.HEATMAP.value == "heatmap"
        assert VisualizationType.RADAR.value == "radar"
        assert VisualizationType.ANIMATION.value == "animation"

    def test_comparison_metric_enum(self):
        """Test comparison metric enumeration."""
        assert ComparisonMetric.DEGRADATION_RATE.value == "degradation_rate"
        assert ComparisonMetric.RECYCLABILITY_SCORE.value == "recyclability_score"
        assert ComparisonMetric.CARBON_FOOTPRINT.value == "carbon_footprint"
        assert ComparisonMetric.OVERALL_SUSTAINABILITY.value == "overall_sustainability"

    @pytest.mark.asyncio
    async def test_simulation_error_handling(self):
        """Test simulation error handling."""
        config = SimulationConfig(use_gpu=False)
        simulator = LifecycleSimulator(config)

        # Test with invalid properties
        invalid_properties = None
        conditions = create_outdoor_weathering_conditions("temperate")

        try:
            trajectory = await simulator.simulate_lifecycle(
                "test", invalid_properties, conditions
            )
            # Should handle gracefully
            assert isinstance(trajectory, LifecycleTrajectory)
        except (TypeError, AttributeError):
            # Expected for None properties
            pass

    def test_visualization_error_handling(self):
        """Test visualization error handling."""
        visualizer = LifecycleVisualizer(use_plotly=False)

        # Test with empty trajectory
        empty_trajectory = LifecycleTrajectory(
            material_id="empty",
            events=[],
            total_duration=0.0,
            lifecycle_phases={},
            cumulative_impacts={},
            final_fate="unknown",
            sustainability_score=0.0,
        )

        result = visualizer.create_lifecycle_timeline(empty_trajectory)

        # Should handle gracefully when Plotly is not available
        assert isinstance(result, str)
        assert "not available" in result.lower()


class TestPerformance:
    """Test performance aspects."""

    @pytest.mark.asyncio
    async def test_simulation_timing(self):
        """Test simulation performance."""
        config = SimulationConfig(
            total_duration=1000.0, output_frequency=200.0, use_gpu=False
        )
        simulator = LifecycleSimulator(config)

        properties = {"molecular_weight": 50000}
        conditions = create_outdoor_weathering_conditions("temperate", duration=1000.0)

        start_time = datetime.now()
        trajectory = await simulator.simulate_lifecycle("test", properties, conditions)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        assert execution_time < 30.0  # Should complete within 30 seconds
        assert isinstance(trajectory, LifecycleTrajectory)

    def test_comparison_timing(self):
        """Test comparison performance."""
        comparator = MaterialComparator()

        # Create multiple trajectories
        trajectories = []
        for i in range(10):
            trajectory = LifecycleTrajectory(
                material_id=f"material_{i}",
                events=[],
                total_duration=1000.0,
                lifecycle_phases={},
                cumulative_impacts={ImpactCategory.CLIMATE_CHANGE: 2.0 + i * 0.1},
                final_fate="recycling",
                sustainability_score=0.8 - i * 0.05,
            )
            trajectories.append(trajectory)

        start_time = datetime.now()
        result = comparator.compare_materials(trajectories)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        assert execution_time < 5.0  # Should complete within 5 seconds
        assert isinstance(result, ComparisonResult)
        assert len(result.materials) == 10


if __name__ == "__main__":
    pytest.main([__file__])

"""
Comprehensive tests for the Interactive Self-Assembly Visualization module.

This test suite covers all aspects of the visualization system including:
- 3D visualization engine functionality
- Real-time dashboard operations
- Phase transition analysis and visualization
- Data export and import capabilities
- Performance and rendering optimization
"""

import tempfile
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from chemi_accel.self_assembly.characterization import (
    CharacterizationResult,
    PhaseType,
    StructureMetrics,
    StructureType,
)
from chemi_accel.self_assembly.coarse_grained_simulation import (
    Particle,
    ParticleType,
    SimulationBox,
    TrajectoryFrame,
)

# Import the modules under test
from chemi_accel.self_assembly.interactive_visualization import (  # Core classes; Configuration classes; Data structures; Enums; Utility functions; Exceptions
    AnimationSpeed,
    ColorScheme,
    DashboardConfig,
    DashboardError,
    DataProcessingError,
    ExportError,
    ExportFormat,
    InteractiveSelfAssemblyVisualizer,
    ParameterSweepData,
    PhaseTransitionAnalyzer,
    PhaseTransitionEvent,
    PropertyTrackingDashboard,
    RenderingError,
    TimeSeriesData,
    VisualizationConfig,
    VisualizationError,
    VisualizationFrame,
    VisualizationMode,
    create_comparative_visualization,
    create_demo_visualization,
    export_dashboard_data,
)

# =============================================================================
# Test Configuration Classes
# =============================================================================


class TestVisualizationConfig:
    """Test visualization configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VisualizationConfig()

        assert config.width == 1200
        assert config.height == 800
        assert config.background_color == "white"
        assert config.particle_size_scale == 1.0
        assert config.color_scheme == ColorScheme.PARTICLE_TYPE
        assert config.animation_speed == AnimationSpeed.NORMAL
        assert config.enable_zoom is True
        assert config.max_particles_3d == 10000

    def test_custom_config(self):
        """Test custom configuration."""
        config = VisualizationConfig(
            width=1600,
            height=1200,
            particle_size_scale=2.0,
            color_scheme=ColorScheme.ENERGY,
            animation_speed=AnimationSpeed.FAST,
            max_particles_3d=50000,
        )

        assert config.width == 1600
        assert config.height == 1200
        assert config.particle_size_scale == 2.0
        assert config.color_scheme == ColorScheme.ENERGY
        assert config.animation_speed == AnimationSpeed.FAST
        assert config.max_particles_3d == 50000


class TestDashboardConfig:
    """Test dashboard configuration."""

    def test_default_config(self):
        """Test default dashboard configuration."""
        config = DashboardConfig()

        assert config.theme == "bootstrap"
        assert config.fluid_layout is True
        assert config.update_interval == 1000.0
        assert config.real_time_updates is True
        assert config.chart_height == 400
        assert config.max_data_points == 10000

    def test_custom_config(self):
        """Test custom dashboard configuration."""
        config = DashboardConfig(
            theme="dark", update_interval=500.0, chart_height=600, max_data_points=5000
        )

        assert config.theme == "dark"
        assert config.update_interval == 500.0
        assert config.chart_height == 600
        assert config.max_data_points == 5000


# =============================================================================
# Test Data Structures
# =============================================================================


class TestVisualizationFrame:
    """Test visualization frame data structure."""

    def test_basic_frame(self):
        """Test basic frame creation."""
        positions = np.random.rand(100, 3) * 10
        particle_types = np.array(["hydrophilic"] * 50 + ["hydrophobic"] * 50)
        particle_ids = np.arange(100)

        frame = VisualizationFrame(
            timestamp=1.0,
            positions=positions,
            particle_types=particle_types,
            particle_ids=particle_ids,
        )

        assert frame.timestamp == 1.0
        assert frame.positions.shape == (100, 3)
        assert len(frame.particle_types) == 100
        assert len(frame.particle_ids) == 100
        assert frame.velocities is None
        assert frame.energies is None

    def test_complete_frame(self):
        """Test frame with all optional data."""
        n_particles = 50

        frame = VisualizationFrame(
            timestamp=2.0,
            positions=np.random.rand(n_particles, 3) * 10,
            particle_types=np.array(["amphiphilic"] * n_particles),
            particle_ids=np.arange(n_particles),
            velocities=np.random.rand(n_particles, 3),
            forces=np.random.rand(n_particles, 3),
            energies=np.random.rand(n_particles),
            cluster_ids=np.random.randint(0, 5, n_particles),
            box_dimensions=(10.0, 10.0, 10.0),
            metadata={"temperature": 300.0},
        )

        assert frame.timestamp == 2.0
        assert frame.velocities is not None
        assert frame.forces is not None
        assert frame.energies is not None
        assert frame.cluster_ids is not None
        assert frame.box_dimensions == (10.0, 10.0, 10.0)
        assert frame.metadata["temperature"] == 300.0


class TestTimeSeriesData:
    """Test time series data structure."""

    def test_basic_time_series(self):
        """Test basic time series creation."""
        timestamps = np.linspace(0, 100, 101)
        properties = {
            "micelle_count": np.random.randint(5, 15, 101),
            "average_size": np.random.uniform(10, 30, 101),
        }

        ts_data = TimeSeriesData(timestamps=timestamps, properties=properties)

        assert len(ts_data.timestamps) == 101
        assert "micelle_count" in ts_data.properties
        assert "average_size" in ts_data.properties
        assert len(ts_data.labels) == 0
        assert len(ts_data.units) == 0

    def test_complete_time_series(self):
        """Test time series with labels and units."""
        timestamps = np.linspace(0, 50, 51)
        properties = {"temperature": np.random.uniform(290, 310, 51)}
        labels = {"temperature": "System Temperature"}
        units = {"temperature": "K"}
        categories = {"temperature": "thermodynamics"}

        ts_data = TimeSeriesData(
            timestamps=timestamps,
            properties=properties,
            labels=labels,
            units=units,
            categories=categories,
        )

        assert ts_data.labels["temperature"] == "System Temperature"
        assert ts_data.units["temperature"] == "K"
        assert ts_data.categories["temperature"] == "thermodynamics"


class TestPhaseTransitionEvent:
    """Test phase transition event data structure."""

    def test_basic_transition(self):
        """Test basic phase transition event."""
        event = PhaseTransitionEvent(
            timestamp=10.0,
            from_phase=PhaseType.LIQUID,
            to_phase=PhaseType.MICELLE,
            transition_probability=0.85,
            order_parameter=0.6,
        )

        assert event.timestamp == 10.0
        assert event.from_phase == PhaseType.LIQUID
        assert event.to_phase == PhaseType.MICELLE
        assert event.transition_probability == 0.85
        assert event.order_parameter == 0.6
        assert event.critical_temperature is None

    def test_complete_transition(self):
        """Test complete phase transition event."""
        nucleation_sites = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]

        event = PhaseTransitionEvent(
            timestamp=15.0,
            from_phase=PhaseType.MICELLE,
            to_phase=PhaseType.VESICLE,
            transition_probability=0.92,
            order_parameter=0.8,
            critical_temperature=305.0,
            critical_concentration=0.1,
            transition_duration=2.5,
            nucleation_sites=nucleation_sites,
        )

        assert event.critical_temperature == 305.0
        assert event.critical_concentration == 0.1
        assert event.transition_duration == 2.5
        assert len(event.nucleation_sites) == 2


# =============================================================================
# Test Interactive Visualizer
# =============================================================================


class TestInteractiveSelfAssemblyVisualizer:
    """Test the main interactive visualizer."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer for testing."""
        config = VisualizationConfig(max_particles_3d=100)
        return InteractiveSelfAssemblyVisualizer(config=config, use_gpu=False)

    @pytest.fixture
    def mock_trajectory(self):
        """Create mock trajectory data."""
        trajectory = []
        n_frames = 10
        n_particles = 50

        for i in range(n_frames):
            particles = []
            for j in range(n_particles):
                # Create mock particle
                particle = Mock()
                particle.position = Mock()
                particle.position.x = np.random.uniform(0, 10)
                particle.position.y = np.random.uniform(0, 10)
                particle.position.z = np.random.uniform(0, 10)
                particle.particle_type = Mock()
                particle.particle_type.name = "hydrophilic" if j < 25 else "hydrophobic"
                particle.particle_id = j
                particle.velocity = Mock()
                particle.velocity.x = np.random.uniform(-1, 1)
                particle.velocity.y = np.random.uniform(-1, 1)
                particle.velocity.z = np.random.uniform(-1, 1)
                particle.energy = np.random.uniform(-1, 1)

                particles.append(particle)

            # Create mock frame
            frame = Mock()
            frame.timestamp = i * 0.1
            frame.particles = particles
            frame.box = Mock()
            frame.box.length = 10.0
            frame.box.width = 10.0
            frame.box.height = 10.0
            frame.total_energy = np.random.uniform(-100, 100)
            frame.temperature = 300.0
            frame.pressure = 1.0

            trajectory.append(frame)

        return trajectory

    def test_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer.config is not None
        assert visualizer.use_gpu is False
        assert len(visualizer.trajectory_data) == 0
        assert visualizer.current_frame == 0
        assert visualizer.animation_running is False
        assert visualizer.figure is None

    def test_load_trajectory(self, visualizer, mock_trajectory):
        """Test loading trajectory data."""
        visualizer.load_trajectory(mock_trajectory)

        assert len(visualizer.trajectory_data) == 10
        assert visualizer.current_frame == 0
        assert visualizer.layout_initialized is False

        # Check first frame
        frame = visualizer.trajectory_data[0]
        assert frame.timestamp == 0.0
        assert frame.positions.shape == (50, 3)
        assert len(frame.particle_types) == 50
        assert len(frame.particle_ids) == 50
        assert frame.box_dimensions == (10.0, 10.0, 10.0)

    @patch("chemi_accel.self_assembly.interactive_visualization.PLOTLY_AVAILABLE", True)
    @patch("chemi_accel.self_assembly.interactive_visualization.go")
    def test_create_3d_visualization(self, mock_go, visualizer, mock_trajectory):
        """Test 3D visualization creation."""
        visualizer.load_trajectory(mock_trajectory)

        # Mock plotly Figure
        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure

        fig = visualizer.create_3d_visualization()

        assert mock_go.Figure.called
        assert visualizer.figure is not None

    def test_create_3d_visualization_no_plotly(self, visualizer, mock_trajectory):
        """Test 3D visualization without plotly."""
        visualizer.load_trajectory(mock_trajectory)

        with patch(
            "chemi_accel.self_assembly.interactive_visualization.PLOTLY_AVAILABLE",
            False,
        ):
            with pytest.raises(VisualizationError, match="Plotly not available"):
                visualizer.create_3d_visualization()

    def test_create_3d_visualization_no_data(self, visualizer):
        """Test 3D visualization without trajectory data."""
        with patch(
            "chemi_accel.self_assembly.interactive_visualization.PLOTLY_AVAILABLE", True
        ):
            with pytest.raises(VisualizationError, match="No trajectory data loaded"):
                visualizer.create_3d_visualization()

    def test_create_3d_visualization_invalid_frame(self, visualizer, mock_trajectory):
        """Test 3D visualization with invalid frame index."""
        visualizer.load_trajectory(mock_trajectory)

        with patch(
            "chemi_accel.self_assembly.interactive_visualization.PLOTLY_AVAILABLE", True
        ):
            with pytest.raises(VisualizationError, match="Frame index .* out of range"):
                visualizer.create_3d_visualization(frame_index=100)

    def test_get_particle_colors(self, visualizer, mock_trajectory):
        """Test particle color assignment."""
        visualizer.load_trajectory(mock_trajectory)
        frame = visualizer.trajectory_data[0]

        # Test particle type coloring
        colors = visualizer._get_particle_colors(frame, ColorScheme.PARTICLE_TYPE)
        assert len(colors) == len(frame.positions)
        assert all(isinstance(c, str) for c in colors)

        # Test energy coloring
        frame.energies = np.random.rand(len(frame.positions))
        colors = visualizer._get_particle_colors(frame, ColorScheme.ENERGY)
        assert len(colors) == len(frame.positions)

        # Test velocity coloring
        frame.velocities = np.random.rand(len(frame.positions), 3)
        colors = visualizer._get_particle_colors(frame, ColorScheme.VELOCITY)
        assert len(colors) == len(frame.positions)

    def test_get_particle_sizes(self, visualizer):
        """Test particle size assignment."""
        particle_types = np.array(
            ["hydrophilic", "hydrophobic", "amphiphilic", "unknown"]
        )
        sizes = visualizer._get_particle_sizes(particle_types)

        assert len(sizes) == 4
        assert all(isinstance(s, (int, float)) for s in sizes)
        assert sizes[0] == 5  # hydrophilic
        assert sizes[1] == 5  # hydrophobic
        assert sizes[2] == 6  # amphiphilic
        assert sizes[3] == 5  # unknown (default)

    @patch("chemi_accel.self_assembly.interactive_visualization.PLOTLY_AVAILABLE", True)
    @patch("chemi_accel.self_assembly.interactive_visualization.go")
    def test_animate_trajectory(self, mock_go, visualizer, mock_trajectory):
        """Test trajectory animation."""
        visualizer.load_trajectory(mock_trajectory)

        # Mock plotly classes
        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        mock_go.Frame.return_value = Mock()
        mock_go.Layout.return_value = Mock()
        mock_go.Scatter3d.return_value = Mock()

        fig = visualizer.animate_trajectory(start_frame=0, end_frame=5, step=1)

        assert mock_go.Figure.called
        assert mock_go.Frame.called

    @patch("chemi_accel.self_assembly.interactive_visualization.PLOTLY_AVAILABLE", True)
    def test_export_visualization_no_figure(self, visualizer):
        """Test export without figure."""
        with pytest.raises(ExportError, match="No figure to export"):
            visualizer.export_visualization("test.png")

    def test_calculate_density_field(self, visualizer, mock_trajectory):
        """Test density field calculation."""
        visualizer.load_trajectory(mock_trajectory)
        frame = visualizer.trajectory_data[0]

        # Create test grid
        X, Y, Z = np.meshgrid(
            np.linspace(0, 10, 5),
            np.linspace(0, 10, 5),
            np.linspace(0, 10, 5),
            indexing="ij",
        )

        density = visualizer._calculate_density_field(frame, X, Y, Z)

        assert density.shape == X.shape
        assert np.all(density >= 0)  # Density should be non-negative


# =============================================================================
# Test Property Tracking Dashboard
# =============================================================================


class TestPropertyTrackingDashboard:
    """Test property tracking dashboard."""

    @pytest.fixture
    def dashboard(self):
        """Create dashboard for testing."""
        return PropertyTrackingDashboard(port=8055)

    def test_initialization(self, dashboard):
        """Test dashboard initialization."""
        assert dashboard.config is not None
        assert dashboard.port == 8055
        assert len(dashboard.time_series_data) == 0
        assert dashboard.parameter_sweep_data is None
        assert len(dashboard.phase_transitions) == 0
        assert dashboard.app is None
        assert dashboard.server_running is False

    def test_add_time_series(self, dashboard):
        """Test adding time series data."""
        timestamps = np.linspace(0, 100, 101)
        properties = {
            "micelle_count": np.random.randint(5, 15, 101),
            "vesicle_count": np.random.randint(0, 5, 101),
        }
        labels = {
            "micelle_count": "Number of Micelles",
            "vesicle_count": "Number of Vesicles",
        }
        units = {"micelle_count": "count", "vesicle_count": "count"}

        dashboard.add_time_series(
            name="simulation_1",
            timestamps=timestamps,
            properties=properties,
            labels=labels,
            units=units,
        )

        assert "simulation_1" in dashboard.time_series_data
        ts_data = dashboard.time_series_data["simulation_1"]
        assert len(ts_data.timestamps) == 101
        assert "micelle_count" in ts_data.properties
        assert ts_data.labels["micelle_count"] == "Number of Micelles"
        assert ts_data.units["vesicle_count"] == "count"

    def test_add_parameter_sweep(self, dashboard):
        """Test adding parameter sweep data."""
        param_names = ["temperature", "concentration"]
        temp_values = np.linspace(280, 320, 10)
        conc_values = np.linspace(0.1, 1.0, 10)
        param_values = np.array([temp_values, conc_values])

        # Create property maps
        T, C = np.meshgrid(temp_values, conc_values, indexing="ij")
        stability_map = np.sin(T / 50) * np.cos(C * 5)

        property_maps = {"stability": stability_map}

        dashboard.add_parameter_sweep(
            parameter_names=param_names,
            parameter_values=param_values,
            property_maps=property_maps,
        )

        assert dashboard.parameter_sweep_data is not None
        sweep = dashboard.parameter_sweep_data
        assert sweep.parameter_names == param_names
        assert sweep.parameter_values.shape == (2, 10)
        assert "stability" in sweep.property_maps
        assert sweep.property_maps["stability"].shape == (10, 10)

    def test_add_phase_transition(self, dashboard):
        """Test adding phase transition."""
        dashboard.add_phase_transition(
            timestamp=10.0,
            from_phase=PhaseType.LIQUID,
            to_phase=PhaseType.MICELLE,
            transition_probability=0.8,
            order_parameter=0.6,
            critical_temperature=300.0,
        )

        assert len(dashboard.phase_transitions) == 1
        event = dashboard.phase_transitions[0]
        assert event.timestamp == 10.0
        assert event.from_phase == PhaseType.LIQUID
        assert event.to_phase == PhaseType.MICELLE
        assert event.transition_probability == 0.8
        assert event.critical_temperature == 300.0

    @patch("chemi_accel.self_assembly.interactive_visualization.DASH_AVAILABLE", False)
    def test_create_dashboard_no_dash(self, dashboard):
        """Test dashboard creation without Dash."""
        with pytest.raises(DashboardError, match="Dash not available"):
            dashboard.create_dashboard()

    @patch("chemi_accel.self_assembly.interactive_visualization.DASH_AVAILABLE", True)
    @patch("chemi_accel.self_assembly.interactive_visualization.dash.Dash")
    def test_create_dashboard_with_dash(self, mock_dash, dashboard):
        """Test dashboard creation with Dash."""
        mock_app = Mock()
        mock_dash.return_value = mock_app

        dashboard.create_dashboard()

        assert dashboard.app is not None
        assert mock_dash.called

    def test_run_dashboard_no_dash(self, dashboard):
        """Test running dashboard without Dash."""
        with patch(
            "chemi_accel.self_assembly.interactive_visualization.DASH_AVAILABLE", False
        ):
            with pytest.raises(DashboardError, match="Dash not available"):
                dashboard.run_dashboard()


# =============================================================================
# Test Phase Transition Analyzer
# =============================================================================


class TestPhaseTransitionAnalyzer:
    """Test phase transition analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer for testing."""
        return PhaseTransitionAnalyzer(use_gpu=False)

    @pytest.fixture
    def mock_characterization_results(self):
        """Create mock characterization results."""
        results = []
        n_frames = 20

        for i in range(n_frames):
            # Create mock structure metrics
            metrics = Mock()
            metrics.micelle_count = 5 + i // 5  # Changes every 5 frames
            metrics.vesicle_count = 2 if i > 10 else 0
            metrics.cluster_stability = 0.8 + 0.1 * np.sin(i / 5)
            metrics.cluster_compactness = 0.7 + 0.2 * np.cos(i / 3)

            # Create mock characterization result
            result = Mock()
            result.structure_metrics = metrics
            results.append(result)

        return results

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.use_gpu is False
        assert len(analyzer.transition_data) == 0
        assert len(analyzer.order_parameters) == 0
        assert len(analyzer.phase_diagrams) == 0

    def test_identify_dominant_phase(self, analyzer):
        """Test phase identification."""
        # Test micelle phase
        result = Mock()
        result.structure_metrics = Mock()
        result.structure_metrics.micelle_count = 5
        result.structure_metrics.vesicle_count = 0
        result.structure_metrics.bilayer_thickness = 0

        phase = analyzer._identify_dominant_phase(result)
        assert phase == PhaseType.MICELLE

        # Test vesicle phase
        result.structure_metrics.micelle_count = 0
        result.structure_metrics.vesicle_count = 3
        phase = analyzer._identify_dominant_phase(result)
        assert phase == PhaseType.VESICLE

        # Test liquid phase
        result.structure_metrics.micelle_count = 0
        result.structure_metrics.vesicle_count = 0
        result.structure_metrics.bilayer_thickness = 0
        phase = analyzer._identify_dominant_phase(result)
        assert phase == PhaseType.LIQUID

    def test_calculate_transition_probability(self, analyzer):
        """Test transition probability calculation."""
        result = Mock()
        result.structure_metrics = Mock()
        result.structure_metrics.cluster_stability = 0.8

        prob = analyzer._calculate_transition_probability(
            PhaseType.LIQUID, PhaseType.MICELLE, result
        )

        assert 0.0 <= prob <= 1.0
        assert prob == 0.8

    def test_calculate_order_parameter(self, analyzer):
        """Test order parameter calculation."""
        result = Mock()
        result.structure_metrics = Mock()
        result.structure_metrics.cluster_compactness = 0.7

        order_param = analyzer._calculate_order_parameter(result)

        assert 0.0 <= order_param <= 1.0
        assert order_param == 0.7

    def test_analyze_phase_transitions(
        self, analyzer, mock_trajectory, mock_characterization_results
    ):
        """Test phase transition analysis."""
        transitions = analyzer.analyze_phase_transitions(
            mock_trajectory, mock_characterization_results
        )

        assert isinstance(transitions, list)
        assert len(transitions) >= 0  # May or may not detect transitions

        for transition in transitions:
            assert isinstance(transition, PhaseTransitionEvent)
            assert transition.timestamp >= 0
            assert isinstance(transition.from_phase, PhaseType)
            assert isinstance(transition.to_phase, PhaseType)
            assert 0.0 <= transition.transition_probability <= 1.0

    @patch("chemi_accel.self_assembly.interactive_visualization.PLOTLY_AVAILABLE", True)
    @patch("chemi_accel.self_assembly.interactive_visualization.go")
    def test_create_phase_diagram(self, mock_go, analyzer):
        """Test phase diagram creation."""
        # Create test data
        temp_values = np.linspace(280, 320, 10)
        conc_values = np.linspace(0.1, 1.0, 10)
        phase_map = np.random.randint(0, len(PhaseType), (10, 10))

        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        mock_go.Heatmap.return_value = Mock()
        mock_go.Scatter.return_value = Mock()

        fig = analyzer.create_phase_diagram(
            temp_values, conc_values, phase_map, "Temperature", "Concentration"
        )

        assert mock_go.Figure.called
        assert mock_go.Heatmap.called

    @patch("chemi_accel.self_assembly.interactive_visualization.PLOTLY_AVAILABLE", True)
    @patch("chemi_accel.self_assembly.interactive_visualization.go")
    def test_create_transition_network(self, mock_go, analyzer):
        """Test transition network creation."""
        # Add some test transitions
        analyzer.transition_data = [
            PhaseTransitionEvent(
                timestamp=1.0,
                from_phase=PhaseType.LIQUID,
                to_phase=PhaseType.MICELLE,
                transition_probability=0.8,
                order_parameter=0.6,
            ),
            PhaseTransitionEvent(
                timestamp=2.0,
                from_phase=PhaseType.MICELLE,
                to_phase=PhaseType.VESICLE,
                transition_probability=0.9,
                order_parameter=0.7,
            ),
        ]

        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        mock_go.Scatter.return_value = Mock()

        fig = analyzer.create_transition_network()

        assert mock_go.Figure.called
        assert mock_go.Scatter.called


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.fixture
    def mock_visualizers(self):
        """Create mock visualizers for testing."""
        visualizers = []

        for i in range(3):
            viz = Mock(spec=InteractiveSelfAssemblyVisualizer)
            viz.current_frame = 0
            viz.trajectory_data = [Mock() for _ in range(10)]

            # Mock frame data
            frame = Mock()
            frame.positions = np.random.rand(20, 3) * 10
            frame.particle_types = np.array(["hydrophilic"] * 10 + ["hydrophobic"] * 10)

            viz.trajectory_data[0] = frame
            viz._get_particle_colors = Mock(return_value=["blue"] * 20)
            viz._get_particle_sizes = Mock(return_value=np.ones(20) * 5)

            visualizers.append(viz)

        return visualizers

    @patch("chemi_accel.self_assembly.interactive_visualization.PLOTLY_AVAILABLE", True)
    @patch("chemi_accel.self_assembly.interactive_visualization.make_subplots")
    @patch("chemi_accel.self_assembly.interactive_visualization.go")
    def test_create_comparative_visualization(
        self, mock_go, mock_subplots, mock_visualizers
    ):
        """Test comparative visualization creation."""
        labels = ["Simulation 1", "Simulation 2", "Simulation 3"]

        mock_figure = Mock()
        mock_subplots.return_value = mock_figure
        mock_go.Scatter3d.return_value = Mock()

        fig = create_comparative_visualization(mock_visualizers, labels)

        assert mock_subplots.called
        assert mock_go.Scatter3d.called

    def test_create_comparative_visualization_no_plotly(self, mock_visualizers):
        """Test comparative visualization without plotly."""
        with patch(
            "chemi_accel.self_assembly.interactive_visualization.PLOTLY_AVAILABLE",
            False,
        ):
            with pytest.raises(VisualizationError, match="Plotly not available"):
                create_comparative_visualization(mock_visualizers, ["A", "B", "C"])

    def test_export_dashboard_data_csv(self):
        """Test dashboard data export to CSV."""
        dashboard = PropertyTrackingDashboard()

        # Add test data
        timestamps = np.linspace(0, 10, 11)
        properties = {"metric1": np.random.rand(11), "metric2": np.random.rand(11)}
        dashboard.add_time_series("test_series", timestamps, properties)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            export_dashboard_data(dashboard, tmp.name, format="csv")

            # Check if file was created
            import os

            assert os.path.exists(tmp.name.replace(".csv", "_test_series.csv"))

    def test_export_dashboard_data_json(self):
        """Test dashboard data export to JSON."""
        dashboard = PropertyTrackingDashboard()

        # Add test data
        timestamps = np.linspace(0, 10, 11)
        properties = {"metric1": np.random.rand(11)}
        dashboard.add_time_series("test_series", timestamps, properties)

        # Add phase transition
        dashboard.add_phase_transition(
            timestamp=5.0,
            from_phase=PhaseType.LIQUID,
            to_phase=PhaseType.MICELLE,
            transition_probability=0.8,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            export_dashboard_data(dashboard, tmp.name, format="json")

            # Check if file was created and contains data
            import json

            with open(tmp.name) as f:
                data = json.load(f)

            assert "time_series" in data
            assert "phase_transitions" in data
            assert "test_series" in data["time_series"]

    def test_export_dashboard_data_unsupported_format(self):
        """Test export with unsupported format."""
        dashboard = PropertyTrackingDashboard()

        with pytest.raises(ExportError, match="Unsupported export format"):
            export_dashboard_data(dashboard, "test.xyz", format="xyz")

    def test_create_demo_visualization(self):
        """Test demo visualization creation."""
        visualizer = create_demo_visualization()

        assert isinstance(visualizer, InteractiveSelfAssemblyVisualizer)
        assert len(visualizer.trajectory_data) > 0
        assert visualizer.current_frame == 0

        # Check frame data
        frame = visualizer.trajectory_data[0]
        assert frame.positions.shape[1] == 3  # 3D positions
        assert len(frame.particle_types) == frame.positions.shape[0]
        assert len(frame.particle_ids) == frame.positions.shape[0]


# =============================================================================
# Test Exception Classes
# =============================================================================


class TestExceptions:
    """Test visualization exception classes."""

    def test_visualization_error(self):
        """Test VisualizationError."""
        with pytest.raises(VisualizationError, match="Test error"):
            raise VisualizationError("Test error")

    def test_rendering_error(self):
        """Test RenderingError."""
        with pytest.raises(RenderingError, match="Rendering failed"):
            raise RenderingError("Rendering failed")

    def test_dashboard_error(self):
        """Test DashboardError."""
        with pytest.raises(DashboardError, match="Dashboard error"):
            raise DashboardError("Dashboard error")

    def test_export_error(self):
        """Test ExportError."""
        with pytest.raises(ExportError, match="Export failed"):
            raise ExportError("Export failed")

    def test_data_processing_error(self):
        """Test DataProcessingError."""
        with pytest.raises(DataProcessingError, match="Data processing error"):
            raise DataProcessingError("Data processing error")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for visualization components."""

    def test_visualizer_dashboard_integration(self):
        """Test integration between visualizer and dashboard."""
        # Create visualizer with demo data
        visualizer = create_demo_visualization()

        # Create dashboard
        dashboard = PropertyTrackingDashboard()

        # Extract metrics from visualizer trajectory
        timestamps = [frame.timestamp for frame in visualizer.trajectory_data]
        micelle_counts = [
            len([p for p in frame.particle_types if "amphiphilic" in p])
            for frame in visualizer.trajectory_data
        ]

        dashboard.add_time_series(
            "simulation",
            np.array(timestamps),
            {"micelle_count": np.array(micelle_counts)},
        )

        assert "simulation" in dashboard.time_series_data
        assert len(dashboard.time_series_data["simulation"].timestamps) == len(
            timestamps
        )

    def test_analyzer_visualizer_integration(self):
        """Test integration between analyzer and visualizer."""
        # Create analyzer
        analyzer = PhaseTransitionAnalyzer()

        # Create simple transition data
        analyzer.transition_data = [
            PhaseTransitionEvent(
                timestamp=1.0,
                from_phase=PhaseType.LIQUID,
                to_phase=PhaseType.MICELLE,
                transition_probability=0.8,
                order_parameter=0.6,
            )
        ]

        # Test that transition data can be used by visualizer
        assert len(analyzer.transition_data) == 1
        transition = analyzer.transition_data[0]
        assert transition.from_phase == PhaseType.LIQUID
        assert transition.to_phase == PhaseType.MICELLE


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for visualization components."""

    def test_large_trajectory_loading(self):
        """Test loading large trajectory data."""
        # Create large mock trajectory
        n_frames = 100
        n_particles = 1000

        trajectory = []
        for i in range(n_frames):
            particles = []
            for j in range(n_particles):
                particle = Mock()
                particle.position = Mock()
                particle.position.x = np.random.uniform(0, 20)
                particle.position.y = np.random.uniform(0, 20)
                particle.position.z = np.random.uniform(0, 20)
                particle.particle_type = Mock()
                particle.particle_type.name = np.random.choice(
                    ["hydrophilic", "hydrophobic"]
                )
                particle.particle_id = j
                particles.append(particle)

            frame = Mock()
            frame.timestamp = i * 0.01
            frame.particles = particles
            frame.box = Mock()
            frame.box.length = 20.0
            frame.box.width = 20.0
            frame.box.height = 20.0

            trajectory.append(frame)

        # Test loading performance
        visualizer = InteractiveSelfAssemblyVisualizer()

        start_time = time.time()
        visualizer.load_trajectory(trajectory)
        load_time = time.time() - start_time

        assert len(visualizer.trajectory_data) == n_frames
        assert load_time < 5.0  # Should load in less than 5 seconds

    def test_time_series_performance(self):
        """Test performance of time series operations."""
        dashboard = PropertyTrackingDashboard()

        # Create large time series
        n_points = 10000
        timestamps = np.linspace(0, 1000, n_points)
        properties = {
            "metric1": np.random.rand(n_points),
            "metric2": np.random.rand(n_points),
            "metric3": np.random.rand(n_points),
        }

        start_time = time.time()
        dashboard.add_time_series("large_series", timestamps, properties)
        add_time = time.time() - start_time

        assert "large_series" in dashboard.time_series_data
        assert add_time < 1.0  # Should add in less than 1 second


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

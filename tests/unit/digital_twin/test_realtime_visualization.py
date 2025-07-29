"""
Unit tests for real-time visualization module.
"""

import time
from collections import deque
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.chemi_accel.core.exceptions import ChemiAccelError
from src.chemi_accel.digital_twin.manufacturing_integration import ProcessConditions
from src.chemi_accel.digital_twin.realtime_process_simulation import (
    MaterialProperty,
    MicrostructureState,
    PropertyType,
    RealTimeSimulationCore,
    SimulationConfig,
    SimulationState,
)
from src.chemi_accel.digital_twin.realtime_visualization import (
    PlotConfiguration,
    PlotlyVisualizationEngine,
    PlotType,
    RealTimeVisualizer,
    VisualizationConfig,
    VisualizationMode,
    VisualizationServer,
)


class TestVisualizationConfig:
    """Test VisualizationConfig dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        config = VisualizationConfig()

        assert config.mode == VisualizationMode.PLOTLY
        assert config.update_interval == 0.1
        assert config.max_history_points == 1000
        assert config.enable_real_time_updates == True
        assert config.figure_size == (1200, 800)

    def test_custom_initialization(self):
        """Test custom initialization."""
        config = VisualizationConfig(
            mode=VisualizationMode.MATPLOTLIB,
            update_interval=0.5,
            max_history_points=500,
            enable_real_time_updates=False,
        )

        assert config.mode == VisualizationMode.MATPLOTLIB
        assert config.update_interval == 0.5
        assert config.max_history_points == 500
        assert config.enable_real_time_updates == False


class TestPlotConfiguration:
    """Test PlotConfiguration dataclass."""

    def test_initialization(self):
        """Test plot configuration initialization."""
        config = PlotConfiguration(
            title="Test Plot",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time (s)",
            y_label="Value",
            properties=["property1", "property2"],
        )

        assert config.title == "Test Plot"
        assert config.plot_type == PlotType.TIME_SERIES
        assert config.x_label == "Time (s)"
        assert config.y_label == "Value"
        assert config.properties == ["property1", "property2"]
        assert config.update_frequency == 1.0
        assert config.auto_scale == True


@pytest.mark.skipif(
    not pytest.importorskip("plotly", minversion=None), reason="Plotly not available"
)
class TestPlotlyVisualizationEngine:
    """Test PlotlyVisualizationEngine class."""

    @pytest.fixture
    def vis_config(self):
        """Create visualization configuration."""
        return VisualizationConfig(mode=VisualizationMode.PLOTLY)

    @pytest.fixture
    def engine(self, vis_config):
        """Create PlotlyVisualizationEngine instance."""
        return PlotlyVisualizationEngine(vis_config)

    def test_initialization(self, engine, vis_config):
        """Test engine initialization."""
        assert engine.config == vis_config
        assert len(engine.figures) == 0

    def test_create_time_series_figure(self, engine):
        """Test creating time series figure."""
        config = PlotConfiguration(
            title="Property Evolution",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time",
            y_label="Value",
            properties=["prop1", "prop2"],
        )

        figure = engine.create_figure(config)

        assert figure is not None
        assert len(figure.data) == 2  # Two properties
        assert figure.layout.title.text == "Property Evolution"
        assert figure.layout.xaxis.title.text == "Time"
        assert figure.layout.yaxis.title.text == "Value"

    def test_create_heatmap_figure(self, engine):
        """Test creating heatmap figure."""
        config = PlotConfiguration(
            title="Spatial Distribution",
            plot_type=PlotType.HEATMAP,
            x_label="X",
            y_label="Y",
            properties=["temperature"],
        )

        figure = engine.create_figure(config)

        assert figure is not None
        assert figure.layout.title.text == "Spatial Distribution"

    def test_create_surface_figure(self, engine):
        """Test creating surface figure."""
        config = PlotConfiguration(
            title="3D Surface",
            plot_type=PlotType.SURFACE,
            x_label="X",
            y_label="Y",
            properties=["elevation"],
        )

        figure = engine.create_figure(config)

        assert figure is not None
        assert figure.layout.title.text == "3D Surface"

    def test_update_time_series_plot(self, engine):
        """Test updating time series plot."""
        config = PlotConfiguration(
            title="Test Plot",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time",
            y_label="Value",
            properties=["prop1"],
        )

        figure = engine.create_figure(config)

        # Update with data
        data = {"time_series": {"prop1": [1, 2, 3, 4, 5]}, "times": [0, 1, 2, 3, 4]}

        engine.update_plot(figure, data)

        # Check that data was updated
        assert len(figure.data[0].x) == 5
        assert len(figure.data[0].y) == 5

    def test_save_figure(self, engine, tmp_path):
        """Test saving figure to file."""
        config = PlotConfiguration(
            title="Test Plot",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time",
            y_label="Value",
            properties=["prop1"],
        )

        figure = engine.create_figure(config)
        filepath = tmp_path / "test_plot.html"

        engine.save_figure(figure, str(filepath))

        assert filepath.exists()


class TestRealTimeVisualizer:
    """Test RealTimeVisualizer class."""

    @pytest.fixture
    def mock_simulation_core(self):
        """Create mock simulation core."""
        mock_core = Mock(spec=RealTimeSimulationCore)
        mock_core.add_update_callback = Mock()
        mock_core.get_simulation_history.return_value = []
        return mock_core

    @pytest.fixture
    def vis_config(self):
        """Create visualization configuration."""
        return VisualizationConfig(
            mode=VisualizationMode.PLOTLY, max_history_points=100
        )

    @pytest.fixture
    def visualizer(self, mock_simulation_core, vis_config):
        """Create RealTimeVisualizer instance."""
        with patch(
            "src.chemi_accel.digital_twin.realtime_visualization.PlotlyVisualizationEngine"
        ):
            return RealTimeVisualizer(mock_simulation_core, vis_config)

    def test_initialization(self, visualizer, mock_simulation_core, vis_config):
        """Test visualizer initialization."""
        assert visualizer.simulation_core == mock_simulation_core
        assert visualizer.config == vis_config
        assert len(visualizer.property_histories) == 0
        assert len(visualizer.time_history) == 0

        # Check that callback was registered
        mock_simulation_core.add_update_callback.assert_called_once()

    def test_on_simulation_update(self, visualizer):
        """Test simulation update handling."""
        # Create mock simulation state
        mock_state = SimulationState(
            time=1.0,
            process_conditions=ProcessConditions(
                temperature=298.15, pressure=1e5, material_composition={"polymer": 1.0}
            ),
            material_properties={
                "tensile_strength": MaterialProperty(
                    "tensile_strength", PropertyType.MECHANICAL, 100.0, "MPa"
                ),
                "elastic_modulus": MaterialProperty(
                    "elastic_modulus", PropertyType.MECHANICAL, 2000.0, "MPa"
                ),
            },
            microstructure=MicrostructureState(
                grain_size=5.0,
                grain_orientation=np.array([1.0, 0.0, 0.0]),
                phase_fractions={"alpha": 1.0},
                defect_density=0.001,
                crystallinity=0.8,
            ),
        )

        # Trigger update
        visualizer._on_simulation_update(mock_state)

        # Check that data was stored
        assert len(visualizer.time_history) == 1
        assert visualizer.time_history[0] == 1.0

        assert "tensile_strength" in visualizer.property_histories
        assert "elastic_modulus" in visualizer.property_histories
        assert visualizer.property_histories["tensile_strength"][0] == 100.0
        assert visualizer.property_histories["elastic_modulus"][0] == 2000.0

    def test_create_property_evolution_plot(self, visualizer):
        """Test creating property evolution plot."""
        properties = ["tensile_strength", "elastic_modulus"]

        with patch.object(visualizer.engine, "create_figure") as mock_create:
            mock_figure = Mock()
            mock_create.return_value = mock_figure

            figure = visualizer.create_property_evolution_plot(properties)

            assert figure == mock_figure
            assert "property_evolution" in visualizer.active_plots
            assert "property_evolution" in visualizer.plot_configs

            # Check configuration
            config = visualizer.plot_configs["property_evolution"]
            assert config.plot_type == PlotType.TIME_SERIES
            assert config.properties == properties

    def test_create_microstructure_evolution_plot(self, visualizer):
        """Test creating microstructure evolution plot."""
        parameters = ["grain_size", "crystallinity"]

        with patch.object(visualizer.engine, "create_figure") as mock_create:
            mock_figure = Mock()
            mock_create.return_value = mock_figure

            figure = visualizer.create_microstructure_evolution_plot(parameters)

            assert figure == mock_figure
            assert "microstructure_evolution" in visualizer.active_plots

    def test_create_spatial_distribution_plot(self, visualizer):
        """Test creating spatial distribution plot."""
        property_name = "temperature"

        with patch.object(visualizer.engine, "create_figure") as mock_create:
            mock_figure = Mock()
            mock_create.return_value = mock_figure

            figure = visualizer.create_spatial_distribution_plot(property_name)

            assert figure == mock_figure
            assert "spatial_distribution" in visualizer.active_plots

            config = visualizer.plot_configs["spatial_distribution"]
            assert config.plot_type == PlotType.HEATMAP

    def test_extract_microstructure_parameter(self, visualizer):
        """Test extracting microstructure parameters."""
        # Add microstructure history
        microstructure = MicrostructureState(
            grain_size=10.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 0.7, "beta": 0.3},
            defect_density=0.01,
            crystallinity=0.8,
        )

        visualizer.microstructure_history.append(microstructure)

        # Extract grain size
        values = visualizer._extract_microstructure_parameter("grain_size")
        assert len(values) == 1
        assert values[0] == 10.0

        # Extract phase fractions (should sum)
        values = visualizer._extract_microstructure_parameter("phase_fractions")
        assert len(values) == 1
        assert values[0] == 1.0  # 0.7 + 0.3

    def test_start_stop_animation(self, visualizer):
        """Test starting and stopping animation."""
        # Start animation
        visualizer.start_animation(interval=0.01)
        assert visualizer.is_animating == True
        assert visualizer.animation_thread is not None

        # Stop animation
        visualizer.stop_animation()
        assert visualizer.is_animating == False

    def test_save_plot(self, visualizer, tmp_path):
        """Test saving individual plot."""
        # Create a mock plot
        mock_figure = Mock()
        visualizer.active_plots["test_plot"] = mock_figure

        filepath = tmp_path / "test_plot.png"

        with patch.object(visualizer.engine, "save_figure") as mock_save:
            visualizer.save_plot("test_plot", str(filepath))
            mock_save.assert_called_once_with(mock_figure, str(filepath))

    def test_save_all_plots(self, visualizer, tmp_path):
        """Test saving all plots."""
        # Create mock plots
        mock_figure1 = Mock()
        mock_figure2 = Mock()
        visualizer.active_plots["plot1"] = mock_figure1
        visualizer.active_plots["plot2"] = mock_figure2

        with patch.object(visualizer.engine, "save_figure") as mock_save:
            visualizer.save_all_plots(str(tmp_path))

            assert mock_save.call_count == 2

    def test_export_data(self, visualizer, tmp_path):
        """Test exporting visualization data."""
        # Add some data
        visualizer.time_history.extend([0.0, 1.0, 2.0])
        visualizer.property_histories["prop1"] = deque([10, 11, 12])

        # Add plot config
        config = PlotConfiguration(
            title="Test Plot",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time",
            y_label="Value",
            properties=["prop1"],
        )
        visualizer.plot_configs["test_plot"] = config

        filepath = tmp_path / "viz_data.json"
        visualizer.export_data(str(filepath))

        assert filepath.exists()

        # Check content
        import json

        with open(filepath) as f:
            data = json.load(f)

        assert "time_history" in data
        assert "property_histories" in data
        assert "plot_configs" in data
        assert len(data["time_history"]) == 3
        assert "prop1" in data["property_histories"]

    def test_get_plot_data(self, visualizer):
        """Test getting plot data."""
        # Setup data
        visualizer.time_history.extend([0.0, 1.0, 2.0])
        visualizer.property_histories["prop1"] = deque([10, 11, 12])

        config = PlotConfiguration(
            title="Test Plot",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time",
            y_label="Value",
            properties=["prop1"],
        )
        visualizer.plot_configs["test_plot"] = config

        plot_data = visualizer.get_plot_data("test_plot")

        assert "config" in plot_data
        assert "times" in plot_data
        assert "properties" in plot_data
        assert len(plot_data["times"]) == 3
        assert "prop1" in plot_data["properties"]

    def test_get_plot_data_nonexistent(self, visualizer):
        """Test getting data for non-existent plot."""
        with pytest.raises(ValueError):
            visualizer.get_plot_data("nonexistent_plot")


class TestVisualizationServer:
    """Test VisualizationServer class."""

    @pytest.fixture
    def mock_visualizer(self):
        """Create mock visualizer."""
        mock_viz = Mock(spec=RealTimeVisualizer)
        mock_viz.property_histories = {"prop1": deque([1, 2, 3])}
        mock_viz.time_history = deque([0, 1, 2])
        return mock_viz

    @pytest.fixture
    def server(self, mock_visualizer):
        """Create VisualizationServer instance."""
        return VisualizationServer(mock_visualizer, port=8080)

    def test_initialization(self, server, mock_visualizer):
        """Test server initialization."""
        assert server.visualizer == mock_visualizer
        assert server.port == 8080
        assert server.is_running == False
        assert server.server_thread is None

    @pytest.mark.skipif(
        not pytest.importorskip("flask", minversion=None), reason="Flask not available"
    )
    def test_start_server_without_flask(self, server):
        """Test server start when Flask is not available."""
        with patch("src.chemi_accel.digital_twin.realtime_visualization.flask", None):
            with pytest.raises(ChemiAccelError):
                server.start_server()


@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", minversion=None),
    reason="Matplotlib not available",
)
class TestMatplotlibVisualizationEngine:
    """Test MatplotlibVisualizationEngine class."""

    @pytest.fixture
    def vis_config(self):
        """Create visualization configuration."""
        return VisualizationConfig(mode=VisualizationMode.MATPLOTLIB)

    @pytest.fixture
    def engine(self, vis_config):
        """Create MatplotlibVisualizationEngine instance."""
        from src.chemi_accel.digital_twin.realtime_visualization import (
            MatplotlibVisualizationEngine,
        )

        return MatplotlibVisualizationEngine(vis_config)

    def test_initialization(self, engine, vis_config):
        """Test engine initialization."""
        assert engine.config == vis_config
        assert len(engine.figures) == 0

    def test_create_time_series_figure(self, engine):
        """Test creating time series figure."""
        config = PlotConfiguration(
            title="Property Evolution",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time",
            y_label="Value",
            properties=["prop1", "prop2"],
        )

        figure = engine.create_figure(config)

        assert figure is not None
        assert hasattr(figure, "lines")
        assert len(figure.lines) == 2  # Two properties

    def test_save_figure(self, engine, tmp_path):
        """Test saving matplotlib figure."""
        config = PlotConfiguration(
            title="Test Plot",
            plot_type=PlotType.TIME_SERIES,
            x_label="Time",
            y_label="Value",
            properties=["prop1"],
        )

        figure = engine.create_figure(config)
        filepath = tmp_path / "test_plot.png"

        engine.save_figure(figure, str(filepath))

        assert filepath.exists()


if __name__ == "__main__":
    pytest.main([__file__])

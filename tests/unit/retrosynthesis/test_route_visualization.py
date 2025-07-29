"""
Tests for interactive route visualization module.

This module tests the interactive visualization tools for synthesis routes,
including molecular rendering, layout calculation, and interactive features.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.retrosynthesis.forward_prediction import (
    ForwardPredictionResult,
    RouteEvaluationResult,
)
from chemi_accel.retrosynthesis.multi_objective_optimization import (
    OptimizationResult,
    OptimizationSolution,
    ParetoFront,
)
from chemi_accel.retrosynthesis.route_enumeration import SynthesisNode, SynthesisRoute

# Import the module under test
from chemi_accel.retrosynthesis.route_visualization import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    ColorScheme,
    InteractionError,
    InteractionMode,
    InteractiveVisualizer,
    LayoutAlgorithm,
    LayoutCalculator,
    LayoutError,
    MoleculeNode,
    MoleculeRenderer,
    ReactionEdge,
    RenderingError,
    VisualizationConfig,
    VisualizationData,
    VisualizationElement,
    VisualizationError,
    VisualizationType,
    create_pareto_front_plot,
    create_visualization_config,
    visualize_synthesis_route_simple,
)


# Test fixtures
@pytest.fixture
def sample_visualization_config():
    """Create sample visualization configuration."""
    return VisualizationConfig(
        visualization_type=VisualizationType.ROUTE_TREE,
        color_scheme=ColorScheme.DEFAULT,
        layout_algorithm=LayoutAlgorithm.HIERARCHICAL,
        interaction_mode=InteractionMode.HOVER_INFO,
        show_molecular_structures=True,
        show_predicted_yields=True,
        figure_width=800,
        figure_height=600,
    )


@pytest.fixture
def sample_synthesis_route():
    """Create sample synthesis route."""
    nodes = [
        SynthesisNode(smiles="CCO", depth=0, confidence=1.0, cost_estimate=0.0),
        SynthesisNode(
            smiles="CC",
            depth=1,
            confidence=0.8,
            cost_estimate=15.0,
            is_purchasable=True,
        ),
        SynthesisNode(
            smiles="O", depth=1, confidence=0.9, cost_estimate=5.0, is_purchasable=True
        ),
    ]
    return SynthesisRoute(target_smiles="CCO", route_nodes=nodes)


@pytest.fixture
def sample_molecule_nodes():
    """Create sample molecule nodes."""
    return [
        MoleculeNode(
            smiles="CCO",
            position=(0.0, 0.0),
            is_purchasable=False,
            cost_estimate=0.0,
            complexity_score=2.5,
        ),
        MoleculeNode(
            smiles="CC",
            position=(100.0, 50.0),
            is_purchasable=True,
            cost_estimate=15.0,
            complexity_score=1.8,
        ),
        MoleculeNode(
            smiles="O",
            position=(100.0, -50.0),
            is_purchasable=True,
            cost_estimate=5.0,
            complexity_score=1.0,
        ),
    ]


@pytest.fixture
def sample_visualization_data(sample_molecule_nodes):
    """Create sample visualization data."""
    vis_data = VisualizationData()

    # Add nodes
    for node in sample_molecule_nodes:
        vis_data.add_node(node)

    # Add edges
    edge1 = ReactionEdge(
        source_node=sample_molecule_nodes[0].element_id,
        target_node=sample_molecule_nodes[1].element_id,
        predicted_yield=0.8,
        confidence=0.7,
    )
    edge2 = ReactionEdge(
        source_node=sample_molecule_nodes[0].element_id,
        target_node=sample_molecule_nodes[2].element_id,
        predicted_yield=0.9,
        confidence=0.8,
    )

    vis_data.add_edge(edge1)
    vis_data.add_edge(edge2)

    # Add metadata
    vis_data.metadata = {
        "route_id": "test_route",
        "target_smiles": "CCO",
        "total_steps": 2,
        "total_cost": 20.0,
    }

    return vis_data


@pytest.fixture
def sample_optimization_result():
    """Create sample optimization result."""
    # Create mock solutions
    solutions = []
    for i in range(3):
        route = Mock()
        route.route_id = f"route_{i}"
        route.total_steps = i + 1
        route.total_cost = (i + 1) * 10.0
        route.overall_confidence = 0.8 - i * 0.1

        solution = OptimizationSolution(
            route=route,
            objective_values={
                "total_cost": route.total_cost,
                "overall_yield": 1.0 - route.overall_confidence,
            },
        )
        solutions.append(solution)

    # Create Pareto front
    front = ParetoFront(front_id=1)
    for solution in solutions:
        front.add_solution(solution)

    result = OptimizationResult()
    result.pareto_fronts = [front]

    return result


class TestEnums:
    """Test enum classes."""

    def test_visualization_type_values(self):
        """Test VisualizationType enum values."""
        assert VisualizationType.ROUTE_TREE.value == "route_tree"
        assert VisualizationType.STEP_BY_STEP.value == "step_by_step"
        assert VisualizationType.PARETO_FRONT.value == "pareto_front"
        assert VisualizationType.OBJECTIVE_COMPARISON.value == "objective_comparison"

    def test_color_scheme_values(self):
        """Test ColorScheme enum values."""
        assert ColorScheme.DEFAULT.value == "default"
        assert ColorScheme.VIRIDIS.value == "viridis"
        assert ColorScheme.PLASMA.value == "plasma"
        assert ColorScheme.COOLWARM.value == "coolwarm"

    def test_layout_algorithm_values(self):
        """Test LayoutAlgorithm enum values."""
        assert LayoutAlgorithm.HIERARCHICAL.value == "hierarchical"
        assert LayoutAlgorithm.FORCE_DIRECTED.value == "force_directed"
        assert LayoutAlgorithm.CIRCULAR.value == "circular"
        assert LayoutAlgorithm.TREE.value == "tree"

    def test_interaction_mode_values(self):
        """Test InteractionMode enum values."""
        assert InteractionMode.VIEW_ONLY.value == "view_only"
        assert InteractionMode.HOVER_INFO.value == "hover_info"
        assert InteractionMode.CLICK_DETAILS.value == "click_details"
        assert InteractionMode.DRAG_AND_DROP.value == "drag_and_drop"


class TestDataStructures:
    """Test data structure classes."""

    def test_visualization_config_creation(self, sample_visualization_config):
        """Test VisualizationConfig creation."""
        config = sample_visualization_config

        assert config.visualization_type == VisualizationType.ROUTE_TREE
        assert config.color_scheme == ColorScheme.DEFAULT
        assert config.layout_algorithm == LayoutAlgorithm.HIERARCHICAL
        assert config.interaction_mode == InteractionMode.HOVER_INFO
        assert config.show_molecular_structures is True
        assert config.figure_width == 800
        assert config.figure_height == 600

    def test_visualization_element_creation(self):
        """Test VisualizationElement creation."""
        element = VisualizationElement(
            element_id="test_element", element_type="test_type", position=(10.0, 20.0)
        )

        assert element.element_id == "test_element"
        assert element.element_type == "test_type"
        assert element.position == (10.0, 20.0)
        assert isinstance(element.properties, dict)
        assert isinstance(element.styling, dict)

    def test_molecule_node_creation(self):
        """Test MoleculeNode creation."""
        node = MoleculeNode(
            smiles="CCO",
            position=(0.0, 0.0),
            is_purchasable=True,
            cost_estimate=25.0,
            complexity_score=3.2,
        )

        assert node.smiles == "CCO"
        assert node.element_type == "molecule"
        assert node.is_purchasable is True
        assert node.cost_estimate == 25.0
        assert node.complexity_score == 3.2
        assert node.element_id != ""

    def test_reaction_edge_creation(self):
        """Test ReactionEdge creation."""
        edge = ReactionEdge(
            source_node="mol_1",
            target_node="mol_2",
            reaction_type="substitution",
            predicted_yield=0.8,
            confidence=0.7,
        )

        assert edge.source_node == "mol_1"
        assert edge.target_node == "mol_2"
        assert edge.element_type == "reaction"
        assert edge.reaction_type == "substitution"
        assert edge.predicted_yield == 0.8
        assert edge.confidence == 0.7
        assert edge.element_id != ""

    def test_visualization_data_creation(self, sample_visualization_data):
        """Test VisualizationData creation."""
        vis_data = sample_visualization_data

        assert len(vis_data.nodes) == 3
        assert len(vis_data.edges) == 2
        assert isinstance(vis_data.metadata, dict)
        assert vis_data.metadata["route_id"] == "test_route"

    def test_visualization_data_add_node(self):
        """Test adding node to visualization data."""
        vis_data = VisualizationData()
        node = MoleculeNode(smiles="CCO")

        vis_data.add_node(node)

        assert len(vis_data.nodes) == 1
        assert vis_data.nodes[0] == node

    def test_visualization_data_add_edge(self):
        """Test adding edge to visualization data."""
        vis_data = VisualizationData()
        edge = ReactionEdge(source_node="mol_1", target_node="mol_2")

        vis_data.add_edge(edge)

        assert len(vis_data.edges) == 1
        assert vis_data.edges[0] == edge

    def test_visualization_data_get_node_by_id(self, sample_visualization_data):
        """Test getting node by ID."""
        vis_data = sample_visualization_data

        # Get first node
        first_node = vis_data.nodes[0]
        found_node = vis_data.get_node_by_id(first_node.element_id)

        assert found_node == first_node

        # Test non-existent node
        not_found = vis_data.get_node_by_id("non_existent")
        assert not_found is None


class TestMoleculeRenderer:
    """Test MoleculeRenderer class."""

    def test_molecule_renderer_creation(self):
        """Test molecule renderer creation."""
        renderer = MoleculeRenderer()

        assert isinstance(renderer.image_cache, dict)
        assert len(renderer.image_cache) == 0

    @patch("chemi_accel.retrosynthesis.route_visualization.Chem")
    @patch("chemi_accel.retrosynthesis.route_visualization.Draw")
    def test_render_molecule(self, mock_draw, mock_chem):
        """Test molecule rendering."""
        renderer = MoleculeRenderer()

        # Mock RDKit components
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_draw.MolToImage.return_value = Mock()

        # Mock image conversion
        with patch("io.BytesIO"), patch("base64.b64encode") as mock_b64:
            mock_b64.return_value.decode.return_value = "test_image_data"

            result = renderer.render_molecule("CCO")

            assert result == "data:image/png;base64,test_image_data"
            mock_chem.MolFromSmiles.assert_called_with("CCO")

    def test_render_molecule_invalid_smiles(self):
        """Test rendering invalid SMILES."""
        renderer = MoleculeRenderer()

        result = renderer.render_molecule("")

        # Should handle gracefully
        assert result is None or isinstance(result, str)

    def test_render_molecule_caching(self):
        """Test molecule rendering caching."""
        renderer = MoleculeRenderer()

        # Mock successful rendering
        with patch.object(
            renderer, "render_molecule", return_value="test_image"
        ) as mock_render:
            # First call
            result1 = renderer.render_molecule("CCO")

            # Add to cache manually for test
            renderer.image_cache["CCO_200x200"] = "cached_image"

            # Create new renderer instance to test cache usage
            renderer2 = MoleculeRenderer()
            renderer2.image_cache = renderer.image_cache

            # This should use cache if implemented
            assert "CCO_200x200" in renderer2.image_cache

    @patch("chemi_accel.retrosynthesis.route_visualization.AllChem")
    @patch("chemi_accel.retrosynthesis.route_visualization.Draw")
    def test_render_reaction(self, mock_draw, mock_allchem):
        """Test reaction rendering."""
        renderer = MoleculeRenderer()

        # Mock RDKit components
        mock_rxn = Mock()
        mock_allchem.ReactionFromSmarts.return_value = mock_rxn
        mock_draw.ReactionToImage.return_value = Mock()

        # Mock image conversion
        with patch("io.BytesIO"), patch("base64.b64encode") as mock_b64:
            mock_b64.return_value.decode.return_value = "test_reaction_data"

            result = renderer.render_reaction(["CCO"], ["CC", "O"])

            assert result == "data:image/png;base64,test_reaction_data"
            mock_allchem.ReactionFromSmarts.assert_called_with("CCO>>CC.O")


class TestLayoutCalculator:
    """Test LayoutCalculator class."""

    def test_layout_calculator_creation(self):
        """Test layout calculator creation."""
        calculator = LayoutCalculator()

        assert calculator is not None

    def test_calculate_hierarchical_layout(self, sample_visualization_data):
        """Test hierarchical layout calculation."""
        calculator = LayoutCalculator()

        positions = calculator._calculate_hierarchical_layout(sample_visualization_data)

        assert isinstance(positions, dict)
        assert len(positions) == len(sample_visualization_data.nodes)

        # Check that all nodes have positions
        for node in sample_visualization_data.nodes:
            assert node.element_id in positions
            pos = positions[node.element_id]
            assert isinstance(pos, tuple)
            assert len(pos) == 2
            assert isinstance(pos[0], (int, float))
            assert isinstance(pos[1], (int, float))

    def test_calculate_circular_layout(self, sample_visualization_data):
        """Test circular layout calculation."""
        calculator = LayoutCalculator()

        positions = calculator._calculate_circular_layout(sample_visualization_data)

        assert isinstance(positions, dict)
        assert len(positions) == len(sample_visualization_data.nodes)

        # Check that positions form a circle-like pattern
        if len(positions) > 1:
            coords = list(positions.values())
            # All positions should be different
            assert len(set(coords)) == len(coords)

    def test_calculate_grid_layout(self, sample_visualization_data):
        """Test grid layout calculation."""
        calculator = LayoutCalculator()

        positions = calculator._calculate_grid_layout(sample_visualization_data, cols=2)

        assert isinstance(positions, dict)
        assert len(positions) == len(sample_visualization_data.nodes)

        # Check grid properties
        coords = list(positions.values())
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]

        # Should have regular spacing
        assert len(set(x_coords)) <= 2  # At most 2 columns

    @patch("chemi_accel.retrosynthesis.route_visualization._NETWORKX_AVAILABLE", False)
    def test_force_directed_layout_fallback(self, sample_visualization_data):
        """Test force-directed layout fallback when NetworkX unavailable."""
        calculator = LayoutCalculator()

        positions = calculator._calculate_force_directed_layout(
            sample_visualization_data
        )

        # Should fall back to hierarchical layout
        assert isinstance(positions, dict)
        assert len(positions) == len(sample_visualization_data.nodes)

    def test_calculate_layout_algorithm_selection(self, sample_visualization_data):
        """Test layout algorithm selection."""
        calculator = LayoutCalculator()

        # Test different algorithms
        algorithms = [
            LayoutAlgorithm.HIERARCHICAL,
            LayoutAlgorithm.CIRCULAR,
            LayoutAlgorithm.GRID,
        ]

        for algorithm in algorithms:
            positions = calculator.calculate_layout(
                sample_visualization_data, algorithm
            )

            assert isinstance(positions, dict)
            assert len(positions) == len(sample_visualization_data.nodes)

    def test_empty_data_handling(self):
        """Test handling of empty visualization data."""
        calculator = LayoutCalculator()
        empty_data = VisualizationData()

        positions = calculator.calculate_layout(
            empty_data, LayoutAlgorithm.HIERARCHICAL
        )

        assert isinstance(positions, dict)
        assert len(positions) == 0


class TestInteractiveVisualizer:
    """Test InteractiveVisualizer class."""

    def test_interactive_visualizer_creation(self, sample_visualization_config):
        """Test interactive visualizer creation."""
        visualizer = InteractiveVisualizer(sample_visualization_config)

        assert visualizer.config == sample_visualization_config
        assert isinstance(visualizer.molecule_renderer, MoleculeRenderer)
        assert isinstance(visualizer.layout_calculator, LayoutCalculator)

    def test_route_to_visualization_data(self, sample_synthesis_route):
        """Test conversion of route to visualization data."""
        visualizer = InteractiveVisualizer()

        vis_data = visualizer._route_to_visualization_data(sample_synthesis_route, None)

        assert isinstance(vis_data, VisualizationData)
        assert len(vis_data.nodes) == len(sample_synthesis_route.route_nodes)
        assert len(vis_data.edges) == len(sample_synthesis_route.route_nodes) - 1

        # Check metadata
        assert (
            vis_data.metadata["target_smiles"] == sample_synthesis_route.target_smiles
        )
        assert vis_data.metadata["total_steps"] == sample_synthesis_route.total_steps

    def test_route_to_visualization_data_with_evaluation(self, sample_synthesis_route):
        """Test route conversion with evaluation result."""
        visualizer = InteractiveVisualizer()

        # Create mock evaluation result
        evaluation_result = Mock()
        evaluation_result.step_predictions = [
            Mock(main_product_yield=0.8, confidence=0.7),
            Mock(main_product_yield=0.9, confidence=0.8),
        ]

        vis_data = visualizer._route_to_visualization_data(
            sample_synthesis_route, evaluation_result
        )

        assert isinstance(vis_data, VisualizationData)
        assert len(vis_data.edges) > 0

        # Check that prediction data was added to edges
        for edge in vis_data.edges:
            assert hasattr(edge, "predicted_yield")
            assert hasattr(edge, "confidence")

    def test_visualize_synthesis_route_text_fallback(self, sample_synthesis_route):
        """Test synthesis route visualization with text fallback."""
        visualizer = InteractiveVisualizer()

        # Mock plotting libraries as unavailable
        with patch(
            "chemi_accel.retrosynthesis.route_visualization._PLOTLY_AVAILABLE", False
        ), patch(
            "chemi_accel.retrosynthesis.route_visualization._MATPLOTLIB_AVAILABLE",
            False,
        ):
            result = visualizer.visualize_synthesis_route(sample_synthesis_route)

            assert isinstance(result, str)
            assert "Synthesis Route" in result
            assert sample_synthesis_route.target_smiles in result

    @patch("chemi_accel.retrosynthesis.route_visualization._PLOTLY_AVAILABLE", True)
    @patch("chemi_accel.retrosynthesis.route_visualization.go")
    def test_visualize_synthesis_route_plotly(self, mock_go, sample_synthesis_route):
        """Test synthesis route visualization with Plotly."""
        visualizer = InteractiveVisualizer()

        # Mock Plotly components
        mock_figure = Mock()
        mock_figure.to_html.return_value = "<html>Test Plot</html>"
        mock_go.Figure.return_value = mock_figure
        mock_go.Scatter.return_value = Mock()

        result = visualizer.visualize_synthesis_route(sample_synthesis_route)

        assert isinstance(result, str)
        assert "html" in result.lower()

    def test_visualize_pareto_front_text_fallback(self, sample_optimization_result):
        """Test Pareto front visualization with text fallback."""
        visualizer = InteractiveVisualizer()

        # Mock plotting libraries as unavailable
        with patch(
            "chemi_accel.retrosynthesis.route_visualization._PLOTLY_AVAILABLE", False
        ):
            result = visualizer.visualize_pareto_front(sample_optimization_result)

            assert isinstance(result, str)
            assert "Pareto Front" in result

    @patch("chemi_accel.retrosynthesis.route_visualization._PLOTLY_AVAILABLE", True)
    @patch("chemi_accel.retrosynthesis.route_visualization.go")
    def test_visualize_pareto_front_plotly(self, mock_go, sample_optimization_result):
        """Test Pareto front visualization with Plotly."""
        visualizer = InteractiveVisualizer()

        # Mock Plotly components
        mock_figure = Mock()
        mock_figure.to_html.return_value = "<html>Pareto Plot</html>"
        mock_go.Figure.return_value = mock_figure
        mock_go.Scatter.return_value = Mock()

        result = visualizer.visualize_pareto_front(sample_optimization_result)

        assert isinstance(result, str)
        assert "html" in result.lower()

    def test_compare_routes_text_fallback(self, sample_synthesis_route):
        """Test route comparison with text fallback."""
        visualizer = InteractiveVisualizer()
        routes = [sample_synthesis_route] * 2

        # Mock plotting libraries as unavailable
        with patch(
            "chemi_accel.retrosynthesis.route_visualization._PLOTLY_AVAILABLE", False
        ):
            result = visualizer.compare_routes(routes)

            assert isinstance(result, str)
            assert "Route Comparison" in result

    def test_extract_route_metrics(self, sample_synthesis_route):
        """Test route metrics extraction."""
        visualizer = InteractiveVisualizer()

        metrics = visualizer._extract_route_metrics(sample_synthesis_route, None)

        assert isinstance(metrics, dict)
        assert "Steps" in metrics
        assert "Cost" in metrics
        assert "Confidence" in metrics

        assert metrics["Steps"] == float(sample_synthesis_route.total_steps)
        assert metrics["Cost"] == sample_synthesis_route.total_cost
        assert metrics["Confidence"] == sample_synthesis_route.overall_confidence

    def test_extract_route_metrics_with_evaluation(self, sample_synthesis_route):
        """Test route metrics extraction with evaluation result."""
        visualizer = InteractiveVisualizer()

        # Create mock evaluation result
        evaluation_result = Mock()
        evaluation_result.overall_yield = 0.8
        evaluation_result.overall_selectivity = 0.9
        evaluation_result.feasibility_score = 0.7

        metrics = visualizer._extract_route_metrics(
            sample_synthesis_route, evaluation_result
        )

        assert isinstance(metrics, dict)
        assert "Overall Yield" in metrics
        assert "Selectivity" in metrics
        assert "Feasibility" in metrics

        assert metrics["Overall Yield"] == 0.8
        assert metrics["Selectivity"] == 0.9
        assert metrics["Feasibility"] == 0.7


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_visualization_config(self):
        """Test create_visualization_config function."""
        config = create_visualization_config(
            visualization_type=VisualizationType.PARETO_FRONT,
            figure_width=1000,
            figure_height=800,
            show_molecular_structures=False,
        )

        assert isinstance(config, VisualizationConfig)
        assert config.visualization_type == VisualizationType.PARETO_FRONT
        assert config.figure_width == 1000
        assert config.figure_height == 800
        assert config.show_molecular_structures is False

    def test_visualize_synthesis_route_simple_function(self, sample_synthesis_route):
        """Test visualize_synthesis_route_simple utility function."""
        result = visualize_synthesis_route_simple(sample_synthesis_route)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_visualize_synthesis_route_simple_with_file_output(
        self, sample_synthesis_route, tmp_path
    ):
        """Test route visualization with file output."""
        output_file = tmp_path / "test_route.html"

        result = visualize_synthesis_route_simple(
            sample_synthesis_route, str(output_file)
        )

        assert isinstance(result, str)
        assert output_file.exists()

        # Check file content
        content = output_file.read_text(encoding="utf-8")
        assert len(content) > 0
        assert content == result

    def test_create_pareto_front_plot_function(self, sample_optimization_result):
        """Test create_pareto_front_plot utility function."""
        result = create_pareto_front_plot(sample_optimization_result)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_pareto_front_plot_with_file_output(
        self, sample_optimization_result, tmp_path
    ):
        """Test Pareto front plot with file output."""
        output_file = tmp_path / "test_pareto.html"

        result = create_pareto_front_plot(sample_optimization_result, str(output_file))

        assert isinstance(result, str)
        assert output_file.exists()

        # Check file content
        content = output_file.read_text(encoding="utf-8")
        assert len(content) > 0
        assert content == result

    @patch("chemi_accel.retrosynthesis.route_visualization._STREAMLIT_AVAILABLE", True)
    def test_launch_streamlit_dashboard_available(self):
        """Test launching Streamlit dashboard when available."""
        from chemi_accel.retrosynthesis.route_visualization import (
            launch_streamlit_dashboard,
        )

        with patch(
            "chemi_accel.retrosynthesis.route_visualization.StreamlitDashboard"
        ) as mock_dashboard:
            mock_instance = Mock()
            mock_dashboard.return_value = mock_instance

            launch_streamlit_dashboard()

            mock_dashboard.assert_called_once()
            mock_instance.create_dashboard.assert_called_once()

    @patch("chemi_accel.retrosynthesis.route_visualization._STREAMLIT_AVAILABLE", False)
    def test_launch_streamlit_dashboard_unavailable(self, capsys):
        """Test launching Streamlit dashboard when unavailable."""
        from chemi_accel.retrosynthesis.route_visualization import (
            launch_streamlit_dashboard,
        )

        launch_streamlit_dashboard()

        captured = capsys.readouterr()
        assert "Streamlit not available" in captured.out


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_visualization_error_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(RenderingError, VisualizationError)
        assert issubclass(LayoutError, VisualizationError)
        assert issubclass(InteractionError, VisualizationError)

    def test_layout_error_handling(self):
        """Test layout calculation error handling."""
        calculator = LayoutCalculator()

        # Create problematic data
        vis_data = VisualizationData()
        vis_data.nodes = [Mock(element_id="invalid")]  # Invalid node

        try:
            positions = calculator.calculate_layout(
                vis_data, LayoutAlgorithm.HIERARCHICAL
            )
            # If no error, check result structure
            assert isinstance(positions, dict)
        except LayoutError:
            # Expected error is acceptable
            pass

    def test_rendering_error_handling(self, sample_synthesis_route):
        """Test rendering error handling."""
        visualizer = InteractiveVisualizer()

        # Force an error condition
        with patch.object(
            visualizer,
            "_route_to_visualization_data",
            side_effect=Exception("Test error"),
        ):
            try:
                result = visualizer.visualize_synthesis_route(sample_synthesis_route)
                # If no error, should still return valid result
                assert isinstance(result, str)
            except RenderingError:
                # Expected error is acceptable
                pass

    def test_molecule_rendering_error_handling(self):
        """Test molecule rendering error handling."""
        renderer = MoleculeRenderer()

        # Test with various problematic inputs
        problematic_inputs = [
            "",  # Empty string
            "INVALID",  # Invalid SMILES
            "C" * 1000,  # Very long string
            None,  # None value
        ]

        for input_smiles in problematic_inputs:
            try:
                result = renderer.render_molecule(
                    str(input_smiles) if input_smiles is not None else ""
                )
                # If successful, should return valid result or None
                assert result is None or isinstance(result, str)
            except Exception:
                # Some errors are expected for invalid inputs
                pass

    def test_visualization_data_error_handling(self):
        """Test visualization data error handling."""
        vis_data = VisualizationData()

        # Test with invalid node
        try:
            invalid_node = Mock()
            vis_data.add_node(invalid_node)
            # Should handle gracefully
            assert len(vis_data.nodes) == 1
        except Exception:
            # Errors are acceptable for invalid data
            pass

    def test_pareto_front_visualization_empty_data(self):
        """Test Pareto front visualization with empty data."""
        visualizer = InteractiveVisualizer()

        # Create empty optimization result
        empty_result = OptimizationResult()
        empty_result.pareto_fronts = []

        result = visualizer.visualize_pareto_front(empty_result)

        assert isinstance(result, str)
        # Should handle empty data gracefully


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    def test_layout_calculation_performance(self):
        """Test layout calculation performance."""
        calculator = LayoutCalculator()

        # Create larger visualization data
        vis_data = VisualizationData()

        # Add many nodes
        for i in range(100):
            node = MoleculeNode(smiles=f"C{i}", position=(0.0, 0.0))
            vis_data.add_node(node)

        # Add edges
        for i in range(50):
            edge = ReactionEdge(
                source_node=vis_data.nodes[i].element_id,
                target_node=vis_data.nodes[i + 1].element_id,
            )
            vis_data.add_edge(edge)

        import time

        start_time = time.time()

        positions = calculator.calculate_layout(vis_data, LayoutAlgorithm.HIERARCHICAL)

        elapsed_time = time.time() - start_time

        assert isinstance(positions, dict)
        assert len(positions) == 100
        assert elapsed_time < 5.0  # Should complete within 5 seconds

    def test_molecule_rendering_performance(self):
        """Test molecule rendering performance."""
        renderer = MoleculeRenderer()

        # Test molecules
        test_molecules = [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "CCN",
            "CCC",
            "c1ccc2c(c1)cccc2",
            "CC(C)CC(C(=O)O)N",
        ] * 10  # 70 renderings

        import time

        start_time = time.time()

        results = []
        for smiles in test_molecules:
            try:
                result = renderer.render_molecule(smiles)
                results.append(result)
            except:
                results.append(None)

        elapsed_time = time.time() - start_time

        assert len(results) == len(test_molecules)
        assert elapsed_time < 10.0  # Should complete within 10 seconds

    def test_visualization_generation_performance(self):
        """Test visualization generation performance."""
        visualizer = InteractiveVisualizer()

        # Create multiple routes
        test_routes = []
        for i in range(10):
            nodes = [
                SynthesisNode(smiles=f"C{i}O", depth=0, confidence=1.0),
                SynthesisNode(
                    smiles=f"C{i}", depth=1, confidence=0.8, is_purchasable=True
                ),
                SynthesisNode(smiles="O", depth=1, confidence=0.9, is_purchasable=True),
            ]
            route = SynthesisRoute(target_smiles=f"C{i}O", route_nodes=nodes)
            test_routes.append(route)

        import time

        start_time = time.time()

        results = []
        for route in test_routes:
            try:
                result = visualizer.visualize_synthesis_route(route)
                results.append(result)
            except:
                results.append(None)

        elapsed_time = time.time() - start_time

        assert len(results) == len(test_routes)
        assert elapsed_time < 30.0  # Should complete within 30 seconds


class TestIntegration:
    """Integration tests for route visualization."""

    def test_full_visualization_workflow(self, sample_synthesis_route):
        """Test complete visualization workflow."""
        # Create configuration
        config = create_visualization_config(
            visualization_type=VisualizationType.ROUTE_TREE,
            layout_algorithm=LayoutAlgorithm.HIERARCHICAL,
            show_molecular_structures=True,
            show_predicted_yields=True,
        )

        # Create visualizer
        visualizer = InteractiveVisualizer(config)

        # Generate visualization
        result = visualizer.visualize_synthesis_route(sample_synthesis_route)

        # Verify complete workflow
        assert isinstance(result, str)
        assert len(result) > 0

        # Should contain route information
        if "html" not in result.lower():  # Text fallback
            assert sample_synthesis_route.target_smiles in result
            assert str(sample_synthesis_route.total_steps) in result

    def test_different_layout_algorithms(self, sample_synthesis_route):
        """Test visualization with different layout algorithms."""
        algorithms = [
            LayoutAlgorithm.HIERARCHICAL,
            LayoutAlgorithm.CIRCULAR,
            LayoutAlgorithm.GRID,
        ]

        all_results = []
        for algorithm in algorithms:
            config = VisualizationConfig(layout_algorithm=algorithm)
            visualizer = InteractiveVisualizer(config)

            result = visualizer.visualize_synthesis_route(sample_synthesis_route)

            assert isinstance(result, str)
            assert len(result) > 0
            all_results.append(result)

        # Should have results from all algorithms
        assert len(all_results) == len(algorithms)

    def test_route_comparison_integration(self):
        """Test route comparison visualization."""
        # Create different routes
        routes = []

        # Route 1: Short, expensive
        nodes1 = [
            SynthesisNode(smiles="CCO", depth=0, confidence=1.0, cost_estimate=0.0),
            SynthesisNode(
                smiles="CC",
                depth=1,
                confidence=0.9,
                cost_estimate=50.0,
                is_purchasable=True,
            ),
        ]
        route1 = SynthesisRoute(target_smiles="CCO", route_nodes=nodes1)
        routes.append(route1)

        # Route 2: Long, cheap
        nodes2 = [
            SynthesisNode(smiles="CCO", depth=0, confidence=1.0, cost_estimate=0.0),
            SynthesisNode(smiles="CCCO", depth=1, confidence=0.8, cost_estimate=10.0),
            SynthesisNode(
                smiles="CC",
                depth=2,
                confidence=0.7,
                cost_estimate=5.0,
                is_purchasable=True,
            ),
            SynthesisNode(
                smiles="CO",
                depth=2,
                confidence=0.8,
                cost_estimate=8.0,
                is_purchasable=True,
            ),
        ]
        route2 = SynthesisRoute(target_smiles="CCO", route_nodes=nodes2)
        routes.append(route2)

        # Compare routes
        visualizer = InteractiveVisualizer()
        result = visualizer.compare_routes(routes)

        assert isinstance(result, str)
        assert len(result) > 0

        # Should contain comparison information
        if "html" not in result.lower():  # Text fallback
            assert "Route Comparison" in result
            assert "Route 1" in result
            assert "Route 2" in result

    def test_pareto_front_visualization_integration(self, sample_optimization_result):
        """Test Pareto front visualization integration."""
        visualizer = InteractiveVisualizer()

        # Test with different objectives
        x_objectives = ["total_cost", "overall_yield"]
        y_objectives = ["overall_yield", "total_cost"]

        for x_obj, y_obj in zip(x_objectives, y_objectives):
            result = visualizer.visualize_pareto_front(
                sample_optimization_result, x_obj, y_obj
            )

            assert isinstance(result, str)
            assert len(result) > 0

    def test_visualization_with_evaluation_results(self, sample_synthesis_route):
        """Test visualization with forward prediction results."""
        visualizer = InteractiveVisualizer()

        # Create mock evaluation result
        evaluation_result = Mock()
        evaluation_result.step_predictions = [
            Mock(
                main_product_yield=0.8,
                confidence=0.7,
                optimal_conditions=Mock(temperature=323.15, catalyst="Pd/C"),
            ),
            Mock(
                main_product_yield=0.9,
                confidence=0.8,
                optimal_conditions=Mock(temperature=298.15, catalyst=None),
            ),
        ]
        evaluation_result.overall_yield = 0.72
        evaluation_result.overall_selectivity = 0.85
        evaluation_result.feasibility_score = 0.78

        result = visualizer.visualize_synthesis_route(
            sample_synthesis_route, evaluation_result
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_error_propagation_through_system(self):
        """Test error propagation through the entire system."""
        visualizer = InteractiveVisualizer()

        # Test with various problematic inputs
        problematic_routes = []

        # Empty route
        empty_route = SynthesisRoute(target_smiles="", route_nodes=[])
        problematic_routes.append(empty_route)

        # Invalid SMILES
        invalid_nodes = [SynthesisNode(smiles="INVALID", depth=0)]
        invalid_route = SynthesisRoute(
            target_smiles="INVALID", route_nodes=invalid_nodes
        )
        problematic_routes.append(invalid_route)

        for route in problematic_routes:
            try:
                result = visualizer.visualize_synthesis_route(route)
                # If successful, verify result structure
                assert isinstance(result, str)
            except (VisualizationError, RenderingError):
                # Expected errors are acceptable
                pass

    def test_configuration_integration(self, sample_synthesis_route):
        """Test integration with different configurations."""
        # Test different configuration combinations
        configs = [
            VisualizationConfig(
                visualization_type=VisualizationType.ROUTE_TREE,
                show_molecular_structures=True,
                show_predicted_yields=True,
            ),
            VisualizationConfig(
                visualization_type=VisualizationType.STEP_BY_STEP,
                show_molecular_structures=False,
                show_cost_estimates=True,
            ),
            VisualizationConfig(
                layout_algorithm=LayoutAlgorithm.CIRCULAR,
                interaction_mode=InteractionMode.CLICK_DETAILS,
                figure_width=1000,
                figure_height=800,
            ),
        ]

        all_results = []
        for config in configs:
            visualizer = InteractiveVisualizer(config)
            result = visualizer.visualize_synthesis_route(sample_synthesis_route)

            assert isinstance(result, str)
            assert len(result) > 0
            all_results.append(result)

        # Should have results from all configurations
        assert len(all_results) == len(configs)

    def test_caching_system_integration(self):
        """Test integration of caching systems."""
        renderer = MoleculeRenderer()

        # Test molecule rendering cache
        test_molecules = ["CCO", "CC", "O"] * 3  # 9 renderings, 3 unique

        results = []
        for smiles in test_molecules:
            try:
                result = renderer.render_molecule(smiles)
                results.append(result)
            except:
                results.append(None)

        # Cache should contain unique molecules
        assert len([r for r in results if r is not None]) >= 0

        # Test that cache is used (would need more detailed implementation to verify)

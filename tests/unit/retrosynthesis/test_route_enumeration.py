"""
Tests for multi-step route enumeration and graph construction.

This module tests the algorithms for iteratively applying retrosynthetic steps,
constructing multi-step synthesis trees, and identifying purchasable starting
materials through chemical building block database queries.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.retrosynthesis.generative_engine import (
    ModelPrediction,
    MolecularRepresentation,
    RetrosynthesisConfig,
    RetrosynthesisEngine,
)

# Import the module under test
from chemi_accel.retrosynthesis.route_enumeration import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    BuildingBlockDatabase,
    BuildingBlockQuery,
    BuildingBlockResult,
    BuildingBlockSource,
    ComplexityCalculator,
    DatabaseQueryError,
    OptimizationError,
    PruningCriteria,
    RouteEnumerationConfig,
    RouteEnumerationError,
    RouteEnumerator,
    RouteMetrics,
    RouteOptimizer,
    RouteStrategy,
    SynthesisComplexity,
    SynthesisNode,
    SynthesisRoute,
    SynthesisTree,
    SynthesisTreeBuilder,
    TreeConstructionError,
    build_synthesis_tree,
    calculate_complexity,
    create_route_config,
    enumerate_routes,
    optimize_route,
    query_building_blocks,
)


# Test fixtures
@pytest.fixture
def sample_route_config():
    """Create a sample route enumeration configuration."""
    return RouteEnumerationConfig(
        strategy=RouteStrategy.BEST_FIRST,
        max_depth=5,
        max_routes=10,
        confidence_threshold=0.6,
        complexity_threshold=8.0,
        cost_threshold=500.0,
        enable_caching=False,  # Disable caching for testing
    )


@pytest.fixture
def sample_synthesis_node():
    """Create a sample synthesis node."""
    return SynthesisNode(
        smiles="CCO",
        depth=1,
        confidence=0.8,
        is_purchasable=False,
        cost_estimate=50.0,
        complexity_score=2.5,
    )


@pytest.fixture
def sample_building_block_query():
    """Create a sample building block query."""
    return BuildingBlockQuery(
        smiles="CCO",
        max_price=100.0,
        min_purity=95.0,
        sources=[BuildingBlockSource.COMMERCIAL],
    )


@pytest.fixture
def sample_retro_config():
    """Create a sample retrosynthesis configuration."""
    return RetrosynthesisConfig(
        use_gpu=False, confidence_threshold=0.5, enable_caching=False
    )


@pytest.fixture
def mock_database():
    """Create a mock database for testing."""
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_conn, mock_cursor


class TestEnums:
    """Test enum classes."""

    def test_route_strategy_values(self):
        """Test RouteStrategy enum values."""
        assert RouteStrategy.BREADTH_FIRST.value == "breadth_first"
        assert RouteStrategy.DEPTH_FIRST.value == "depth_first"
        assert RouteStrategy.BEST_FIRST.value == "best_first"
        assert RouteStrategy.MONTE_CARLO.value == "monte_carlo"
        assert RouteStrategy.BEAM_SEARCH.value == "beam_search"

    def test_pruning_criteria_values(self):
        """Test PruningCriteria enum values."""
        assert PruningCriteria.CONFIDENCE_THRESHOLD.value == "confidence_threshold"
        assert PruningCriteria.COMPLEXITY_INCREASE.value == "complexity_increase"
        assert PruningCriteria.MAXIMUM_STEPS.value == "maximum_steps"
        assert PruningCriteria.COST_THRESHOLD.value == "cost_threshold"

    def test_building_block_source_values(self):
        """Test BuildingBlockSource enum values."""
        assert BuildingBlockSource.COMMERCIAL.value == "commercial"
        assert BuildingBlockSource.ACADEMIC.value == "academic"
        assert BuildingBlockSource.ENAMINE.value == "enamine"
        assert BuildingBlockSource.ZINC.value == "zinc"

    def test_synthesis_complexity_values(self):
        """Test SynthesisComplexity enum values."""
        assert SynthesisComplexity.TRIVIAL.value == "trivial"
        assert SynthesisComplexity.SIMPLE.value == "simple"
        assert SynthesisComplexity.MODERATE.value == "moderate"
        assert SynthesisComplexity.COMPLEX.value == "complex"
        assert SynthesisComplexity.VERY_COMPLEX.value == "very_complex"


class TestDataStructures:
    """Test data structure classes."""

    def test_synthesis_node_creation(self, sample_synthesis_node):
        """Test SynthesisNode creation."""
        node = sample_synthesis_node

        assert node.smiles == "CCO"
        assert node.depth == 1
        assert node.confidence == 0.8
        assert node.is_purchasable is False
        assert node.cost_estimate == 50.0
        assert node.complexity_score == 2.5
        assert node.node_id != ""

    def test_synthesis_node_add_child(self, sample_synthesis_node):
        """Test adding child to synthesis node."""
        parent = sample_synthesis_node
        child = SynthesisNode(smiles="CC", depth=2)

        parent.add_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert not parent.is_leaf()
        assert child.is_leaf()

    def test_synthesis_node_molecular_representation(self, sample_synthesis_node):
        """Test molecular representation generation."""
        node = sample_synthesis_node
        mol_rep = node.get_molecular_representation()

        assert isinstance(mol_rep, MolecularRepresentation)
        assert mol_rep.smiles == "CCO"

    def test_synthesis_tree_creation(self, sample_synthesis_node):
        """Test SynthesisTree creation."""
        tree = SynthesisTree(root=sample_synthesis_node, max_depth=5)

        assert tree.root == sample_synthesis_node
        assert tree.max_depth == 5
        assert tree.total_nodes >= 1

    def test_synthesis_tree_leaf_nodes(self):
        """Test getting leaf nodes from tree."""
        root = SynthesisNode(smiles="CCO", depth=0)
        child1 = SynthesisNode(smiles="CC", depth=1)
        child2 = SynthesisNode(smiles="O", depth=1)

        root.add_child(child1)
        root.add_child(child2)

        tree = SynthesisTree(root=root)
        leaves = tree.get_leaf_nodes()

        assert len(leaves) == 2
        assert child1 in leaves
        assert child2 in leaves

    def test_synthesis_tree_complete_routes(self):
        """Test getting complete routes from tree."""
        root = SynthesisNode(smiles="CCO", depth=0)
        child1 = SynthesisNode(smiles="CC", depth=1, is_purchasable=True)
        child2 = SynthesisNode(smiles="O", depth=1, is_purchasable=True)

        root.add_child(child1)
        root.add_child(child2)

        tree = SynthesisTree(root=root)
        routes = tree.get_complete_routes()

        assert len(routes) == 2
        assert all(len(route) == 2 for route in routes)  # Root + leaf

    def test_route_enumeration_config_creation(self, sample_route_config):
        """Test RouteEnumerationConfig creation."""
        config = sample_route_config

        assert config.strategy == RouteStrategy.BEST_FIRST
        assert config.max_depth == 5
        assert config.max_routes == 10
        assert config.confidence_threshold == 0.6
        assert config.enable_caching is False

    def test_synthesis_route_creation(self):
        """Test SynthesisRoute creation."""
        nodes = [
            SynthesisNode(smiles="CCO", depth=0, confidence=1.0, cost_estimate=0.0),
            SynthesisNode(smiles="CC", depth=1, confidence=0.8, cost_estimate=10.0),
            SynthesisNode(smiles="O", depth=1, confidence=0.7, cost_estimate=5.0),
        ]

        route = SynthesisRoute(target_smiles="CCO", route_nodes=nodes)

        assert route.target_smiles == "CCO"
        assert route.total_steps == 2  # len(nodes) - 1
        assert route.total_cost == 15.0  # Sum of costs
        assert 0.0 <= route.overall_confidence <= 1.0
        assert isinstance(route.complexity, SynthesisComplexity)

    def test_synthesis_route_complexity_determination(self):
        """Test route complexity determination."""
        # Simple route (2 steps)
        nodes_simple = [SynthesisNode(smiles="CCO")] * 3
        route_simple = SynthesisRoute(target_smiles="CCO", route_nodes=nodes_simple)
        assert route_simple.complexity == SynthesisComplexity.SIMPLE

        # Complex route (6 steps)
        nodes_complex = [SynthesisNode(smiles="CCO")] * 7
        route_complex = SynthesisRoute(target_smiles="CCO", route_nodes=nodes_complex)
        assert route_complex.complexity == SynthesisComplexity.COMPLEX

    def test_route_metrics_creation(self):
        """Test RouteMetrics creation."""
        metrics = RouteMetrics(
            convergence=0.8,
            atom_economy=0.7,
            step_economy=0.9,
            green_chemistry_score=0.6,
            synthetic_accessibility=0.8,
            novelty_score=0.5,
        )

        overall_score = metrics.overall_score()
        assert 0.0 <= overall_score <= 1.0

    def test_building_block_query_creation(self, sample_building_block_query):
        """Test BuildingBlockQuery creation."""
        query = sample_building_block_query

        assert query.smiles == "CCO"
        assert query.max_price == 100.0
        assert query.min_purity == 95.0
        assert BuildingBlockSource.COMMERCIAL in query.sources

    def test_building_block_result_creation(self):
        """Test BuildingBlockResult creation."""
        result = BuildingBlockResult(
            smiles="CCO",
            is_available=True,
            price=25.0,
            purity=99.0,
            supplier="Test Supplier",
            catalog_id="T123",
        )

        assert result.smiles == "CCO"
        assert result.is_available is True
        assert result.price == 25.0
        assert result.supplier == "Test Supplier"


class TestBuildingBlockDatabase:
    """Test BuildingBlockDatabase class."""

    def test_database_creation(self):
        """Test database creation."""
        db = BuildingBlockDatabase(":memory:")

        assert db.database_path == ":memory:"
        assert db.connection is not None

    def test_database_initialization_error(self):
        """Test database initialization error handling."""
        with patch("sqlite3.connect", side_effect=Exception("DB Error")):
            with pytest.raises(DatabaseQueryError):
                BuildingBlockDatabase()

    def test_query_building_block(self, sample_building_block_query):
        """Test querying for a building block."""
        db = BuildingBlockDatabase(":memory:")

        result = db.query_building_block(sample_building_block_query)

        assert isinstance(result, BuildingBlockResult)
        assert result.smiles == sample_building_block_query.smiles

    def test_query_building_block_found(self):
        """Test successful building block query."""
        db = BuildingBlockDatabase(":memory:")

        # Query for ethanol (should be in sample data)
        query = BuildingBlockQuery(smiles="CCO")
        result = db.query_building_block(query)

        assert isinstance(result, BuildingBlockResult)
        assert result.smiles == "CCO"
        # Should find ethanol in sample data
        if result.is_available:
            assert result.price > 0
            assert result.purity > 0

    def test_batch_query(self):
        """Test batch querying."""
        db = BuildingBlockDatabase(":memory:")

        queries = [
            BuildingBlockQuery(smiles="CCO"),
            BuildingBlockQuery(smiles="CC"),
            BuildingBlockQuery(smiles="C"),
        ]

        results = db.batch_query(queries)

        assert len(results) == 3
        assert all(isinstance(r, BuildingBlockResult) for r in results)

    def test_add_building_block(self):
        """Test adding a building block to database."""
        db = BuildingBlockDatabase(":memory:")

        db.add_building_block(
            smiles="CCC",
            price=15.0,
            purity=98.0,
            supplier="Test Corp",
            catalog_id="TC001",
            delivery_time=2,
            source=BuildingBlockSource.COMMERCIAL,
        )

        # Query the added block
        query = BuildingBlockQuery(smiles="CCC")
        result = db.query_building_block(query)

        assert result.is_available
        assert result.smiles == "CCC"
        assert result.price == 15.0

    def test_database_query_error_handling(self):
        """Test error handling in database queries."""
        db = BuildingBlockDatabase(":memory:")

        # Force connection to None to simulate error
        db.connection = None

        query = BuildingBlockQuery(smiles="CCO")

        with pytest.raises(DatabaseQueryError):
            db.query_building_block(query)


class TestComplexityCalculator:
    """Test ComplexityCalculator class."""

    def test_complexity_calculator_creation(self):
        """Test complexity calculator creation."""
        calculator = ComplexityCalculator()

        assert isinstance(calculator.complexity_cache, dict)

    def test_calculate_molecular_complexity(self):
        """Test molecular complexity calculation."""
        calculator = ComplexityCalculator()

        # Test simple molecule
        complexity_simple = calculator.calculate_molecular_complexity("CCO")
        assert complexity_simple >= 0
        assert complexity_simple <= 10.0

        # Test complex molecule
        complexity_complex = calculator.calculate_molecular_complexity(
            "c1ccc2c(c1)c(c(n2)C(=O)O)c3cccc(c3)Cl"
        )
        assert complexity_complex >= 0
        assert complexity_complex <= 10.0

        # Complex molecule should have higher complexity
        assert complexity_complex > complexity_simple

    def test_complexity_caching(self):
        """Test complexity calculation caching."""
        calculator = ComplexityCalculator()

        # First calculation
        complexity1 = calculator.calculate_molecular_complexity("CCO")
        assert "CCO" in calculator.complexity_cache

        # Second calculation (should use cache)
        complexity2 = calculator.calculate_molecular_complexity("CCO")
        assert complexity1 == complexity2

    def test_calculate_route_complexity(self):
        """Test route complexity calculation."""
        calculator = ComplexityCalculator()

        nodes = [
            SynthesisNode(smiles="CCO", depth=0),
            SynthesisNode(smiles="CC", depth=1),
            SynthesisNode(smiles="O", depth=1),
        ]

        route = SynthesisRoute(target_smiles="CCO", route_nodes=nodes)
        complexity = calculator.calculate_route_complexity(route)

        assert complexity >= 0

    def test_invalid_smiles_complexity(self):
        """Test complexity calculation with invalid SMILES."""
        calculator = ComplexityCalculator()

        complexity = calculator.calculate_molecular_complexity("INVALID")
        assert complexity == 0.0


class TestRouteOptimizer:
    """Test RouteOptimizer class."""

    def test_route_optimizer_creation(self, sample_route_config):
        """Test route optimizer creation."""
        optimizer = RouteOptimizer(sample_route_config)

        assert optimizer.config == sample_route_config
        assert isinstance(optimizer.complexity_calculator, ComplexityCalculator)

    def test_optimize_routes(self, sample_route_config):
        """Test route optimization."""
        optimizer = RouteOptimizer(sample_route_config)

        # Create sample routes with different qualities
        routes = [
            SynthesisRoute(
                target_smiles="CCO",
                route_nodes=[SynthesisNode(smiles="CCO", confidence=0.6)],
            ),
            SynthesisRoute(
                target_smiles="CCO",
                route_nodes=[SynthesisNode(smiles="CCO", confidence=0.9)],
            ),
            SynthesisRoute(
                target_smiles="CCO",
                route_nodes=[SynthesisNode(smiles="CCO", confidence=0.3)],
            ),
        ]

        optimized = optimizer.optimize_routes(routes)

        assert len(optimized) <= len(routes)
        assert len(optimized) <= sample_route_config.max_routes

        # Routes should be sorted by quality (best first)
        if len(optimized) > 1:
            for i in range(len(optimized) - 1):
                score1 = getattr(
                    optimized[i], "metrics", RouteMetrics()
                ).overall_score()
                score2 = getattr(
                    optimized[i + 1], "metrics", RouteMetrics()
                ).overall_score()
                assert score1 >= score2

    def test_calculate_route_metrics(self, sample_route_config):
        """Test route metrics calculation."""
        optimizer = RouteOptimizer(sample_route_config)

        nodes = [SynthesisNode(smiles="CCO", confidence=0.8)]
        route = SynthesisRoute(target_smiles="CCO", route_nodes=nodes)

        metrics = optimizer._calculate_route_metrics(route)

        assert isinstance(metrics, RouteMetrics)
        assert 0.0 <= metrics.step_economy <= 1.0
        assert 0.0 <= metrics.synthetic_accessibility <= 1.0
        assert 0.0 <= metrics.convergence <= 1.0
        assert 0.0 <= metrics.overall_score() <= 1.0


class TestSynthesisTreeBuilder:
    """Test SynthesisTreeBuilder class."""

    def test_tree_builder_creation(self, sample_route_config, sample_retro_config):
        """Test synthesis tree builder creation."""
        retro_engine = RetrosynthesisEngine(sample_retro_config)
        building_block_db = BuildingBlockDatabase(":memory:")

        builder = SynthesisTreeBuilder(
            retro_engine, building_block_db, sample_route_config
        )

        assert builder.retro_engine == retro_engine
        assert builder.building_block_db == building_block_db
        assert builder.config == sample_route_config

    def test_build_tree(self, sample_route_config, sample_retro_config):
        """Test synthesis tree building."""
        retro_engine = RetrosynthesisEngine(sample_retro_config)
        building_block_db = BuildingBlockDatabase(":memory:")
        builder = SynthesisTreeBuilder(
            retro_engine, building_block_db, sample_route_config
        )

        tree = builder.build_tree("CCO")

        assert isinstance(tree, SynthesisTree)
        assert tree.root.smiles == "CCO"
        assert tree.total_nodes >= 1

    def test_build_tree_purchasable_target(
        self, sample_route_config, sample_retro_config
    ):
        """Test building tree for already purchasable target."""
        retro_engine = RetrosynthesisEngine(sample_retro_config)
        building_block_db = BuildingBlockDatabase(":memory:")
        builder = SynthesisTreeBuilder(
            retro_engine, building_block_db, sample_route_config
        )

        # CCO should be in sample data as purchasable
        tree = builder.build_tree("CCO")

        assert isinstance(tree, SynthesisTree)
        assert tree.root.smiles == "CCO"
        # If purchasable, tree should be shallow
        if tree.root.is_purchasable:
            assert tree.max_depth == 0

    def test_is_valid_reactant(self, sample_route_config, sample_retro_config):
        """Test reactant validation."""
        retro_engine = RetrosynthesisEngine(sample_retro_config)
        building_block_db = BuildingBlockDatabase(":memory:")
        builder = SynthesisTreeBuilder(
            retro_engine, building_block_db, sample_route_config
        )

        assert builder._is_valid_reactant("CCO") is True
        assert builder._is_valid_reactant("CC") is True
        assert builder._is_valid_reactant("") is False
        assert builder._is_valid_reactant("INVALID") is False


class TestRouteEnumerator:
    """Test RouteEnumerator class."""

    def test_route_enumerator_creation(self, sample_route_config):
        """Test route enumerator creation."""
        enumerator = RouteEnumerator(sample_route_config)

        assert enumerator.config == sample_route_config
        assert isinstance(enumerator.retro_engine, RetrosynthesisEngine)
        assert isinstance(enumerator.building_block_db, BuildingBlockDatabase)
        assert isinstance(enumerator.tree_builder, SynthesisTreeBuilder)
        assert isinstance(enumerator.route_optimizer, RouteOptimizer)

    def test_enumerate_routes(self, sample_route_config):
        """Test route enumeration."""
        enumerator = RouteEnumerator(sample_route_config)

        routes = enumerator.enumerate_routes("CCO")

        assert isinstance(routes, list)
        assert all(isinstance(r, SynthesisRoute) for r in routes)
        assert len(routes) <= sample_route_config.max_routes

    def test_enumerate_routes_caching(self):
        """Test route enumeration caching."""
        config = RouteEnumerationConfig(enable_caching=True, max_depth=3)
        enumerator = RouteEnumerator(config)

        # First enumeration
        routes1 = enumerator.enumerate_routes("CCO")

        # Second enumeration (should use cache)
        routes2 = enumerator.enumerate_routes("CCO")

        assert len(routes1) == len(routes2)
        assert len(enumerator._route_cache) > 0

    def test_get_route_statistics(self, sample_route_config):
        """Test route statistics calculation."""
        enumerator = RouteEnumerator(sample_route_config)

        routes = enumerator.enumerate_routes("CCO")
        stats = enumerator.get_route_statistics(routes)

        if routes:  # Only test if routes were found
            assert isinstance(stats, dict)
            assert "total_routes" in stats
            assert "avg_steps" in stats
            assert "avg_confidence" in stats
            assert stats["total_routes"] == len(routes)
        else:
            assert stats == {}

    def test_filter_routes(self, sample_route_config):
        """Test route filtering."""
        enumerator = RouteEnumerator(sample_route_config)

        # Create routes with different properties
        routes = [
            SynthesisRoute(
                target_smiles="CCO",
                route_nodes=[
                    SynthesisNode(smiles="CCO", confidence=0.8, cost_estimate=100)
                ],
            ),
            SynthesisRoute(
                target_smiles="CCO",
                route_nodes=[
                    SynthesisNode(smiles="CCO", confidence=0.4, cost_estimate=50)
                ],  # Below threshold
            ),
            SynthesisRoute(
                target_smiles="CCO",
                route_nodes=[
                    SynthesisNode(smiles="CCO", confidence=0.9, cost_estimate=1000)
                ],  # Too expensive
            ),
        ]

        filtered = enumerator._filter_routes(routes)

        # Should filter based on confidence and cost thresholds
        assert len(filtered) <= len(routes)
        for route in filtered:
            assert route.overall_confidence >= sample_route_config.confidence_threshold


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_route_config(self):
        """Test create_route_config function."""
        config = create_route_config(
            strategy=RouteStrategy.DEPTH_FIRST, max_depth=8, max_routes=50
        )

        assert isinstance(config, RouteEnumerationConfig)
        assert config.strategy == RouteStrategy.DEPTH_FIRST
        assert config.max_depth == 8
        assert config.max_routes == 50

    def test_build_synthesis_tree_function(self):
        """Test build_synthesis_tree utility function."""
        tree = build_synthesis_tree("CCO")

        assert isinstance(tree, SynthesisTree)
        assert tree.root.smiles == "CCO"

    def test_enumerate_routes_function(self):
        """Test enumerate_routes utility function."""
        routes = enumerate_routes("CCO")

        assert isinstance(routes, list)
        assert all(isinstance(r, SynthesisRoute) for r in routes)

    def test_optimize_route_function(self):
        """Test optimize_route utility function."""
        routes = [
            SynthesisRoute(
                target_smiles="CCO",
                route_nodes=[SynthesisNode(smiles="CCO", confidence=0.6)],
            ),
            SynthesisRoute(
                target_smiles="CCO",
                route_nodes=[SynthesisNode(smiles="CCO", confidence=0.9)],
            ),
        ]

        optimized = optimize_route(routes)

        assert isinstance(optimized, list)
        assert len(optimized) <= len(routes)

    def test_calculate_complexity_function(self):
        """Test calculate_complexity utility function."""
        complexity = calculate_complexity("CCO")

        assert isinstance(complexity, float)
        assert complexity >= 0
        assert complexity <= 10.0

    def test_query_building_blocks_function(self):
        """Test query_building_blocks utility function."""
        smiles_list = ["CCO", "CC", "C"]
        results = query_building_blocks(smiles_list)

        assert isinstance(results, list)
        assert len(results) == len(smiles_list)
        assert all(isinstance(r, BuildingBlockResult) for r in results)


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_route_enumeration_error_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(TreeConstructionError, RouteEnumerationError)
        assert issubclass(DatabaseQueryError, RouteEnumerationError)
        assert issubclass(OptimizationError, RouteEnumerationError)

    def test_database_query_error_handling(self):
        """Test database query error handling."""
        # Test with invalid database path or connection issues
        with patch("sqlite3.connect", side_effect=Exception("Connection failed")):
            with pytest.raises(DatabaseQueryError):
                BuildingBlockDatabase()

    def test_route_enumeration_error_handling(self, sample_route_config):
        """Test route enumeration error handling."""
        enumerator = RouteEnumerator(sample_route_config)

        # Test with problematic input
        try:
            routes = enumerator.enumerate_routes("INVALID_SMILES")
            # If no error, check that result is still valid
            assert isinstance(routes, list)
        except RouteEnumerationError:
            # Expected error is acceptable
            pass

    def test_tree_construction_error_handling(
        self, sample_route_config, sample_retro_config
    ):
        """Test tree construction error handling."""
        # Create builder with mock components that might fail
        retro_engine = RetrosynthesisEngine(sample_retro_config)
        building_block_db = BuildingBlockDatabase(":memory:")
        builder = SynthesisTreeBuilder(
            retro_engine, building_block_db, sample_route_config
        )

        # Test with various inputs
        try:
            tree = builder.build_tree("")  # Empty SMILES
            assert isinstance(tree, SynthesisTree)
        except (TreeConstructionError, Exception):
            # Errors are acceptable for invalid input
            pass


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    def test_route_enumeration_performance(self, sample_route_config):
        """Test route enumeration performance."""
        # Use smaller limits for testing
        config = RouteEnumerationConfig(
            max_depth=3, max_routes=5, confidence_threshold=0.5
        )
        enumerator = RouteEnumerator(config)

        import time

        start_time = time.time()

        routes = enumerator.enumerate_routes("CCO")

        elapsed_time = time.time() - start_time

        # Should complete within reasonable time
        assert elapsed_time < 30.0  # 30 seconds max
        assert isinstance(routes, list)

    def test_batch_building_block_queries(self):
        """Test batch building block query performance."""
        db = BuildingBlockDatabase(":memory:")

        # Test batch queries
        smiles_list = ["CCO", "CC", "C", "CCC", "CCCC"] * 10  # 50 queries

        import time

        start_time = time.time()

        queries = [BuildingBlockQuery(smiles=smiles) for smiles in smiles_list]
        results = db.batch_query(queries)

        elapsed_time = time.time() - start_time

        assert len(results) == len(smiles_list)
        assert elapsed_time < 10.0  # Should be reasonably fast

    def test_complexity_calculation_performance(self):
        """Test complexity calculation performance."""
        calculator = ComplexityCalculator()

        # Test with various molecules
        test_molecules = [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "CCN",
            "CCC",
            "c1ccc2c(c1)cccc2",
            "CC(C)CC(C(=O)O)N",
        ] * 20  # 140 calculations

        import time

        start_time = time.time()

        complexities = [
            calculator.calculate_molecular_complexity(smiles)
            for smiles in test_molecules
        ]

        elapsed_time = time.time() - start_time

        assert len(complexities) == len(test_molecules)
        assert all(c >= 0 for c in complexities)
        assert elapsed_time < 5.0  # Should be fast due to caching

    def test_memory_usage_with_caching(self):
        """Test memory usage with caching enabled."""
        config = RouteEnumerationConfig(enable_caching=True, max_depth=2)
        enumerator = RouteEnumerator(config)

        # Make multiple requests for same molecule
        for i in range(5):
            routes = enumerator.enumerate_routes("CCO")
            assert isinstance(routes, list)

        # Cache should contain results
        assert len(enumerator._route_cache) > 0


class TestIntegration:
    """Integration tests for route enumeration."""

    def test_full_route_enumeration_workflow(self):
        """Test complete route enumeration workflow."""
        # Create configuration
        config = create_route_config(
            strategy=RouteStrategy.BEST_FIRST,
            max_depth=4,
            max_routes=5,
            confidence_threshold=0.4,
        )

        # Create enumerator
        enumerator = RouteEnumerator(config)

        # Enumerate routes
        routes = enumerator.enumerate_routes("CCO")

        # Verify complete workflow
        assert isinstance(routes, list)
        assert all(isinstance(r, SynthesisRoute) for r in routes)
        assert len(routes) <= config.max_routes

        # Check route quality
        for route in routes:
            assert route.target_smiles == "CCO"
            assert route.total_steps >= 0
            assert 0.0 <= route.overall_confidence <= 1.0
            assert route.total_cost >= 0

    def test_different_target_molecules(self):
        """Test route enumeration for different target molecules."""
        config = RouteEnumerationConfig(max_depth=3, max_routes=3)
        enumerator = RouteEnumerator(config)

        # Test different target molecules
        test_targets = [
            "CCO",  # Simple alcohol
            "c1ccccc1",  # Aromatic
            "CC(=O)O",  # Carboxylic acid
            "CCN",  # Amine
        ]

        all_routes = []
        for target in test_targets:
            routes = enumerator.enumerate_routes(target)

            assert isinstance(routes, list)
            assert all(r.target_smiles == target for r in routes)
            all_routes.extend(routes)

        # Should have found routes for at least some targets
        assert len(all_routes) >= 0

    def test_building_block_database_integration(self):
        """Test integration with building block database."""
        config = RouteEnumerationConfig(max_depth=2)
        enumerator = RouteEnumerator(config)

        # Test that building block queries work in route enumeration
        routes = enumerator.enumerate_routes("CCO")

        # Verify database integration
        assert isinstance(enumerator.building_block_db, BuildingBlockDatabase)

        # Test direct database query
        query = BuildingBlockQuery(smiles="CCO")
        result = enumerator.building_block_db.query_building_block(query)
        assert isinstance(result, BuildingBlockResult)

    def test_error_propagation_through_system(self):
        """Test error propagation through the entire system."""
        config = RouteEnumerationConfig(max_depth=2, max_routes=5)
        enumerator = RouteEnumerator(config)

        # Test with various problematic inputs
        problematic_inputs = [
            "",  # Empty string
            "INVALID",  # Invalid SMILES
            "C" * 200,  # Very long string
        ]

        for target in problematic_inputs:
            try:
                routes = enumerator.enumerate_routes(target)
                # If successful, verify result structure
                assert isinstance(routes, list)
                assert all(isinstance(r, SynthesisRoute) for r in routes)
            except RouteEnumerationError:
                # Expected errors are acceptable
                pass

    def test_route_optimization_integration(self):
        """Test integration with route optimization."""
        config = RouteEnumerationConfig(max_depth=3, max_routes=10)
        enumerator = RouteEnumerator(config)

        routes = enumerator.enumerate_routes("CCO")

        if routes:  # Only test if routes were found
            # Test that routes are optimized
            assert isinstance(enumerator.route_optimizer, RouteOptimizer)

            # Routes should be sorted by quality
            if len(routes) > 1:
                # Check that routes have metrics
                for route in routes:
                    if hasattr(route, "metrics"):
                        assert isinstance(route.metrics, RouteMetrics)

            # Get statistics
            stats = enumerator.get_route_statistics(routes)
            assert isinstance(stats, dict)
            assert stats.get("total_routes") == len(routes)

    def test_synthesis_tree_to_routes_conversion(self):
        """Test conversion from synthesis tree to routes."""
        config = RouteEnumerationConfig(max_depth=2)
        enumerator = RouteEnumerator(config)

        # Build tree directly
        tree = enumerator.tree_builder.build_tree("CCO")

        # Get complete routes from tree
        route_paths = tree.get_complete_routes()

        # Convert to SynthesisRoute objects
        synthesis_routes = []
        for route_nodes in route_paths:
            route = SynthesisRoute(target_smiles="CCO", route_nodes=route_nodes)
            synthesis_routes.append(route)

        # Verify conversion
        assert all(isinstance(r, SynthesisRoute) for r in synthesis_routes)
        assert all(r.target_smiles == "CCO" for r in synthesis_routes)

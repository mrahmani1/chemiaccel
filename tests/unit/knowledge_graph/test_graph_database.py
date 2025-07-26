"""
Tests for Graph Database Integration

Comprehensive testing of multi-backend graph database system
including Neo4j, Jena, GPU, and memory backends.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from chemi_accel.knowledge_graph.graph_database import (
    BulkLoadConfig,
    GraphDatabaseBackend,
    GraphDatabaseManager,
    GraphDatabaseType,
    GraphQuery,
    GraphQueryResult,
    IndexConfig,
    get_graph_database,
)
from chemi_accel.knowledge_graph.memory_backend import MemoryGraphBackend


class TestGraphDatabaseFactory:
    """Test graph database factory functions."""

    def test_get_memory_backend(self):
        """Test getting memory backend."""
        backend = get_graph_database("memory", test_config=True)

        assert isinstance(backend, GraphDatabaseBackend)
        assert backend.backend_type == GraphDatabaseType.MEMORY
        assert backend.is_connected()

    def test_get_invalid_backend(self):
        """Test getting invalid backend raises error."""
        with pytest.raises(ValueError, match="Unsupported graph database backend"):
            get_graph_database("invalid_backend")

    @patch("chemi_accel.knowledge_graph.neo4j_backend.neo4j")
    def test_get_neo4j_backend_available(self, mock_neo4j):
        """Test getting Neo4j backend when available."""
        mock_neo4j.GraphDatabase.driver.return_value = Mock()

        backend = get_graph_database("neo4j")
        assert backend.backend_type == GraphDatabaseType.NEO4J

    @patch("chemi_accel.knowledge_graph.jena_backend.rdflib")
    def test_get_jena_backend_available(self, mock_rdflib):
        """Test getting Jena backend when available."""
        mock_rdflib.Graph.return_value = Mock()

        backend = get_graph_database("jena")
        assert backend.backend_type == GraphDatabaseType.JENA

    @patch("chemi_accel.knowledge_graph.gpu_graph_backend.cudf")
    @patch("chemi_accel.knowledge_graph.gpu_graph_backend.cugraph")
    def test_get_gpu_backend_available(self, mock_cugraph, mock_cudf):
        """Test getting GPU backend when available."""
        mock_cudf.DataFrame.return_value = Mock()
        mock_cugraph.Graph.return_value = Mock()

        backend = get_graph_database("gpu_graph")
        assert backend.backend_type == GraphDatabaseType.GPU_GRAPH


class TestMemoryBackend:
    """Test memory graph backend implementation."""

    @pytest.fixture
    def backend(self):
        """Create a memory backend for testing."""
        backend = MemoryGraphBackend({})
        backend.connect()
        return backend

    def test_backend_initialization(self, backend):
        """Test backend initialization."""
        assert backend.backend_type == GraphDatabaseType.MEMORY
        assert backend.is_connected()
        assert len(backend.nodes) == 0
        assert len(backend.relationships) == 0

    def test_create_node(self, backend):
        """Test node creation."""
        labels = ["Chemical", "Compound"]
        properties = {"name": "aspirin", "formula": "C9H8O4"}

        node_id = backend.create_node(labels, properties)

        assert node_id is not None
        assert len(backend.nodes) == 1

        # Verify node data
        node = backend.get_node(node_id)
        assert node is not None
        assert node["labels"] == labels
        assert node["name"] == "aspirin"
        assert node["formula"] == "C9H8O4"
        assert "_created_at" in node

    def test_create_relationship(self, backend):
        """Test relationship creation."""
        # Create nodes first
        node1_id = backend.create_node(["Drug"], {"name": "aspirin"})
        node2_id = backend.create_node(["Condition"], {"name": "pain"})

        # Create relationship
        rel_id = backend.create_relationship(
            node1_id, node2_id, "TREATS", {"efficacy": "high"}
        )

        assert rel_id is not None
        assert len(backend.relationships) == 1

        # Verify relationship data
        relationship = backend.relationships[rel_id]
        assert relationship["source_id"] == node1_id
        assert relationship["target_id"] == node2_id
        assert relationship["relationship_type"] == "TREATS"
        assert relationship["efficacy"] == "high"

    def test_find_nodes_by_labels(self, backend):
        """Test finding nodes by labels."""
        # Create test nodes
        backend.create_node(["Chemical", "Drug"], {"name": "aspirin"})
        backend.create_node(["Chemical", "Solvent"], {"name": "water"})
        backend.create_node(["Protein"], {"name": "enzyme"})

        # Find by single label
        chemicals = backend.find_nodes(labels=["Chemical"])
        assert len(chemicals) == 2

        # Find by multiple labels
        drugs = backend.find_nodes(labels=["Drug"])
        assert len(drugs) == 1
        assert drugs[0]["name"] == "aspirin"

    def test_find_nodes_by_properties(self, backend):
        """Test finding nodes by properties."""
        # Create test nodes
        backend.create_node(["Drug"], {"name": "aspirin", "type": "nsaid"})
        backend.create_node(["Drug"], {"name": "ibuprofen", "type": "nsaid"})
        backend.create_node(["Drug"], {"name": "morphine", "type": "opioid"})

        # Find by property
        nsaids = backend.find_nodes(properties={"type": "nsaid"})
        assert len(nsaids) == 2

        # Find by specific name
        aspirin = backend.find_nodes(properties={"name": "aspirin"})
        assert len(aspirin) == 1
        assert aspirin[0]["type"] == "nsaid"

    def test_update_node(self, backend):
        """Test node updates."""
        # Create node
        node_id = backend.create_node(["Drug"], {"name": "aspirin", "dosage": "500mg"})

        # Update properties
        success = backend.update_node(node_id, {"dosage": "325mg", "form": "tablet"})
        assert success

        # Verify updates
        node = backend.get_node(node_id)
        assert node["dosage"] == "325mg"
        assert node["form"] == "tablet"
        assert node["name"] == "aspirin"  # Original property preserved
        assert "_updated_at" in node

    def test_delete_node(self, backend):
        """Test node deletion."""
        # Create nodes and relationship
        node1_id = backend.create_node(["Drug"], {"name": "aspirin"})
        node2_id = backend.create_node(["Condition"], {"name": "pain"})
        rel_id = backend.create_relationship(node1_id, node2_id, "TREATS")

        # Delete node
        success = backend.delete_node(node1_id)
        assert success

        # Verify deletion
        assert backend.get_node(node1_id) is None
        assert len(backend.nodes) == 1
        assert len(backend.relationships) == 0  # Relationship should be deleted too

    def test_execute_query_find_nodes(self, backend):
        """Test query execution for finding nodes."""
        # Create test data
        backend.create_node(["Drug"], {"name": "aspirin", "type": "nsaid"})
        backend.create_node(["Drug"], {"name": "morphine", "type": "opioid"})

        # Execute query
        query = GraphQuery(
            query="find_nodes",
            parameters={"labels": ["Drug"], "properties": {"type": "nsaid"}},
            query_type="find_nodes",
        )

        result = backend.execute_query(query)

        assert isinstance(result, GraphQueryResult)
        assert len(result.records) == 1
        assert result.records[0]["name"] == "aspirin"
        assert result.summary["query_type"] == "find_nodes"
        assert result.execution_time > 0

    def test_execute_query_neighbors(self, backend):
        """Test neighbor query execution."""
        # Create test graph
        drug_id = backend.create_node(["Drug"], {"name": "aspirin"})
        condition1_id = backend.create_node(["Condition"], {"name": "pain"})
        condition2_id = backend.create_node(["Condition"], {"name": "fever"})

        backend.create_relationship(drug_id, condition1_id, "TREATS")
        backend.create_relationship(drug_id, condition2_id, "REDUCES")

        # Query neighbors
        query = GraphQuery(
            query="neighbors",
            parameters={"node_id": drug_id, "direction": "out"},
            query_type="neighbors",
        )

        result = backend.execute_query(query)

        assert len(result.records) == 2
        for record in result.records:
            assert record["direction"] == "outgoing"
            assert record["node_id"] in [condition1_id, condition2_id]

    def test_bulk_load_nodes(self, backend):
        """Test bulk node loading."""
        nodes = [
            {"labels": ["Drug"], "properties": {"name": "aspirin", "type": "nsaid"}},
            {"labels": ["Drug"], "properties": {"name": "ibuprofen", "type": "nsaid"}},
            {"labels": ["Drug"], "properties": {"name": "morphine", "type": "opioid"}},
        ]

        config = BulkLoadConfig(batch_size=2, validation=False)
        result = backend.bulk_load_nodes(nodes, config)

        assert result["total_nodes"] == 3
        assert result["created_nodes"] == 3
        assert result["failed_nodes"] == 0
        assert len(backend.nodes) == 3

    def test_bulk_load_relationships(self, backend):
        """Test bulk relationship loading."""
        # Create nodes first
        node1_id = backend.create_node(["Drug"], {"name": "aspirin"})
        node2_id = backend.create_node(["Condition"], {"name": "pain"})
        node3_id = backend.create_node(["Condition"], {"name": "fever"})

        relationships = [
            {
                "source_id": node1_id,
                "target_id": node2_id,
                "relationship_type": "TREATS",
                "properties": {"efficacy": "high"},
            },
            {
                "source_id": node1_id,
                "target_id": node3_id,
                "relationship_type": "REDUCES",
                "properties": {"efficacy": "medium"},
            },
        ]

        config = BulkLoadConfig(batch_size=1)
        result = backend.bulk_load_relationships(relationships, config)

        assert result["total_relationships"] == 2
        assert result["created_relationships"] == 2
        assert result["failed_relationships"] == 0
        assert len(backend.relationships) == 2

    def test_create_index(self, backend):
        """Test index creation."""
        index_config = IndexConfig(
            entity_type="Drug", properties=["name"], index_type="btree", unique=False
        )

        success = backend.create_index(index_config)
        assert success

        # Verify index exists
        indexes = backend.list_indexes()
        assert len(indexes) == 1
        assert indexes[0]["entity_type"] == "Drug"
        assert indexes[0]["properties"] == ["name"]

    def test_get_statistics(self, backend):
        """Test statistics retrieval."""
        # Create some test data
        backend.create_node(["Drug"], {"name": "aspirin"})
        backend.create_node(["Condition"], {"name": "pain"})

        stats = backend.get_statistics()

        assert "database" in stats
        assert "memory" in stats
        assert "connection" in stats

        assert stats["database"]["total_nodes"] == 2
        assert stats["memory"]["nodes_in_memory"] == 2
        assert stats["connection"]["backend"] == "memory"
        assert stats["connection"]["connected"] == True

    def test_clear_database(self, backend):
        """Test database clearing."""
        # Create test data
        backend.create_node(["Drug"], {"name": "aspirin"})
        backend.create_node(["Condition"], {"name": "pain"})

        assert len(backend.nodes) == 2

        # Clear database
        success = backend.clear_database()
        assert success

        # Verify clearing
        assert len(backend.nodes) == 0
        assert len(backend.relationships) == 0
        assert len(backend.label_index) == 0

    def test_connection_validation(self, backend):
        """Test connection validation."""
        validation = backend.validate_connection()

        assert validation["connected"] == True
        assert validation["read_access"] == True
        assert validation["write_access"] == True
        assert validation["index_support"] == True
        assert validation["bulk_load_support"] == True
        assert len(validation["errors"]) == 0

    def test_error_handling(self, backend):
        """Test error handling in various operations."""
        # Test relationship creation with non-existent nodes
        with pytest.raises(ValueError, match="Source or target node not found"):
            backend.create_relationship("nonexistent1", "nonexistent2", "RELATES")

        # Test updating non-existent node
        success = backend.update_node("nonexistent", {"prop": "value"})
        assert not success

        # Test deleting non-existent node
        success = backend.delete_node("nonexistent")
        assert not success

        # Test getting non-existent node
        node = backend.get_node("nonexistent")
        assert node is None


class TestGraphDatabaseManager:
    """Test graph database manager functionality."""

    @pytest.fixture
    def manager(self):
        """Create a database manager for testing."""
        return GraphDatabaseManager()

    def test_register_backend(self, manager):
        """Test backend registration."""
        backend = MemoryGraphBackend({})
        backend.connect()

        manager.register_backend("test_memory", backend, is_default=True)

        assert "test_memory" in manager.backends
        assert manager.default_backend == "test_memory"

    def test_get_backend(self, manager):
        """Test backend retrieval."""
        backend = MemoryGraphBackend({})
        backend.connect()

        manager.register_backend("test_memory", backend)

        retrieved = manager.get_backend("test_memory")
        assert retrieved is backend

        # Test default backend
        manager.default_backend = "test_memory"
        default = manager.get_backend()
        assert default is backend

    def test_get_nonexistent_backend(self, manager):
        """Test getting non-existent backend raises error."""
        with pytest.raises(ValueError, match="Backend 'nonexistent' not registered"):
            manager.get_backend("nonexistent")

    def test_execute_distributed_query(self, manager):
        """Test distributed query execution."""
        # Register multiple backends
        backend1 = MemoryGraphBackend({})
        backend1.connect()
        backend2 = MemoryGraphBackend({})
        backend2.connect()

        manager.register_backend("backend1", backend1)
        manager.register_backend("backend2", backend2)

        # Create test data in both backends
        backend1.create_node(["Drug"], {"name": "aspirin"})
        backend2.create_node(["Drug"], {"name": "ibuprofen"})

        # Execute distributed query
        query = GraphQuery(
            query="find_nodes", parameters={"labels": ["Drug"]}, query_type="find_nodes"
        )

        results = manager.execute_distributed_query(query)

        assert len(results) == 2
        assert all(isinstance(result, GraphQueryResult) for result in results)
        assert all(
            result.metadata["backend"] in ["backend1", "backend2"] for result in results
        )


class TestGraphQueryTypes:
    """Test different graph query types and formats."""

    @pytest.fixture
    def backend(self):
        """Create backend with test data."""
        backend = MemoryGraphBackend({})
        backend.connect()

        # Create test graph
        drug_id = backend.create_node(["Drug"], {"name": "aspirin", "type": "nsaid"})
        condition1_id = backend.create_node(["Condition"], {"name": "pain"})
        condition2_id = backend.create_node(["Condition"], {"name": "fever"})

        backend.create_relationship(drug_id, condition1_id, "TREATS", {"efficacy": 0.8})
        backend.create_relationship(
            drug_id, condition2_id, "REDUCES", {"efficacy": 0.6}
        )

        return backend

    def test_find_paths_query(self, backend):
        """Test path finding queries."""
        # Get node IDs for path query
        drugs = backend.find_nodes(labels=["Drug"])
        conditions = backend.find_nodes(labels=["Condition"])

        drug_id = drugs[0]["_id"]
        condition_id = conditions[0]["_id"]

        query = GraphQuery(
            query="find_paths",
            parameters={"source": drug_id, "target": condition_id, "max_depth": 2},
            query_type="find_paths",
        )

        result = backend.execute_query(query)

        assert len(result.records) >= 1
        assert all("path" in record for record in result.records)
        assert all("length" in record for record in result.records)

    def test_node_degrees_query(self, backend):
        """Test node degree calculations."""
        query = GraphQuery(
            query="node_degrees", parameters={}, query_type="node_degrees"
        )

        result = backend.execute_query(query)

        assert len(result.records) >= 3  # All nodes should be returned

        for record in result.records:
            assert "node_id" in record
            assert "in_degree" in record
            assert "out_degree" in record
            assert "total_degree" in record

    def test_count_queries(self, backend):
        """Test count queries."""
        # Count nodes
        node_count_query = GraphQuery(
            query="count_nodes", parameters={}, query_type="count_nodes"
        )

        result = backend.execute_query(node_count_query)
        assert result.records[0]["count"] >= 3

        # Count relationships
        rel_count_query = GraphQuery(
            query="count_relationships", parameters={}, query_type="count_relationships"
        )

        result = backend.execute_query(rel_count_query)
        assert result.records[0]["count"] >= 2

    def test_invalid_query_type(self, backend):
        """Test handling of invalid query types."""
        query = GraphQuery(
            query="invalid_operation", parameters={}, query_type="invalid_type"
        )

        result = backend.execute_query(query)

        assert result.metadata.get("error") == True
        assert "Unsupported query type" in result.summary["error"]


class TestBulkOperations:
    """Test bulk operations and performance."""

    @pytest.fixture
    def backend(self):
        """Create backend for bulk testing."""
        backend = MemoryGraphBackend({})
        backend.connect()
        return backend

    def test_large_bulk_load(self, backend):
        """Test loading large number of nodes."""
        # Create large dataset
        nodes = []
        for i in range(1000):
            nodes.append(
                {
                    "labels": ["TestNode"],
                    "properties": {"id": i, "name": f"node_{i}", "value": i * 2},
                }
            )

        config = BulkLoadConfig(batch_size=100, validation=False)
        result = backend.bulk_load_nodes(nodes, config)

        assert result["total_nodes"] == 1000
        assert result["created_nodes"] == 1000
        assert result["failed_nodes"] == 0
        assert len(backend.nodes) == 1000

    def test_bulk_load_with_progress_callback(self, backend):
        """Test bulk loading with progress reporting."""
        progress_updates = []

        def progress_callback(progress, message):
            progress_updates.append((progress, message))

        nodes = [
            {"labels": ["Drug"], "properties": {"name": f"drug_{i}"}} for i in range(50)
        ]

        config = BulkLoadConfig(batch_size=10, progress_callback=progress_callback)

        backend.bulk_load_nodes(nodes, config)

        # Should have received progress updates
        assert len(progress_updates) > 0
        assert any(update[0] == 1.0 for update in progress_updates)  # Final progress

    def test_bulk_load_error_handling(self, backend):
        """Test bulk loading with some invalid data."""
        # Mix of valid and invalid relationships
        node_id = backend.create_node(["Drug"], {"name": "aspirin"})

        relationships = [
            {
                "source_id": node_id,
                "target_id": "nonexistent",  # Invalid target
                "relationship_type": "TREATS",
            },
            {
                "source_id": "nonexistent",  # Invalid source
                "target_id": node_id,
                "relationship_type": "CAUSED_BY",
            },
        ]

        result = backend.bulk_load_relationships(relationships)

        assert result["total_relationships"] == 2
        assert result["failed_relationships"] == 2
        assert len(result["errors"]) > 0


class TestPerformanceMetrics:
    """Test performance characteristics and metrics."""

    @pytest.fixture
    def backend(self):
        """Create backend with performance test data."""
        backend = MemoryGraphBackend({})
        backend.connect()

        # Create medium-sized test dataset
        nodes = []
        for i in range(100):
            nodes.append(
                {"labels": ["TestNode"], "properties": {"id": i, "name": f"node_{i}"}}
            )

        backend.bulk_load_nodes(nodes)
        return backend

    def test_query_performance_metrics(self, backend):
        """Test query performance measurement."""
        query = GraphQuery(
            query="find_nodes",
            parameters={"labels": ["TestNode"]},
            query_type="find_nodes",
        )

        result = backend.execute_query(query)

        # Check performance metrics
        assert result.execution_time > 0
        assert result.execution_time < 1.0  # Should be fast for memory backend
        assert len(result.records) == 100

        # Check that statistics were updated
        stats = backend.get_statistics()
        assert stats["queries_executed"] > 0
        assert stats["last_activity"] is not None

    def test_index_performance_impact(self, backend):
        """Test performance impact of indexes."""
        # Measure query time without index
        query = GraphQuery(
            query="find_nodes",
            parameters={"properties": {"name": "node_50"}},
            query_type="find_nodes",
        )

        result1 = backend.execute_query(query)
        time_without_index = result1.execution_time

        # Create index
        index_config = IndexConfig(
            entity_type="TestNode", properties=["name"], index_type="btree"
        )
        backend.create_index(index_config)

        # Measure query time with index
        result2 = backend.execute_query(query)
        time_with_index = result2.execution_time

        # Both should find the same result
        assert len(result1.records) == len(result2.records) == 1

        # With small dataset, difference might not be significant,
        # but index should not make performance worse
        assert time_with_index <= time_without_index * 2  # Allow some overhead


class TestErrorConditions:
    """Test various error conditions and edge cases."""

    @pytest.fixture
    def backend(self):
        """Create backend for error testing."""
        backend = MemoryGraphBackend({})
        backend.connect()
        return backend

    def test_empty_operations(self, backend):
        """Test operations with empty inputs."""
        # Empty bulk loads
        result = backend.bulk_load_nodes([])
        assert result["total_nodes"] == 0
        assert result["created_nodes"] == 0

        result = backend.bulk_load_relationships([])
        assert result["total_relationships"] == 0
        assert result["created_relationships"] == 0

        # Empty find operations
        nodes = backend.find_nodes(labels=["NonExistent"])
        assert len(nodes) == 0

    def test_malformed_data_handling(self, backend):
        """Test handling of malformed data."""
        # Node without required fields
        with pytest.raises(Exception):
            backend.create_node([], None)  # None properties should cause error

        # Bulk load with malformed nodes
        malformed_nodes = [
            {"invalid": "structure"},  # Missing labels and properties
            {"labels": ["Drug"]},  # Missing properties
            {"properties": {"name": "test"}},  # Missing labels
        ]

        # Should handle gracefully - some might succeed with defaults
        result = backend.bulk_load_nodes(malformed_nodes)
        assert result["total_nodes"] == len(malformed_nodes)
        # Some may fail depending on implementation

    def test_concurrent_access_safety(self, backend):
        """Test thread safety of concurrent operations."""
        import threading
        import time

        results = []
        errors = []

        def create_nodes(thread_id):
            try:
                for i in range(10):
                    node_id = backend.create_node(
                        ["TestNode"], {"thread": thread_id, "index": i}
                    )
                    results.append(node_id)
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_nodes, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 50  # 5 threads × 10 nodes each
        assert len(set(results)) == 50  # All node IDs should be unique
        assert len(backend.nodes) == 50

"""
Tests for Graph Query Engine

Testing query optimization, caching, and specialized chemical queries.
"""

import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest
from chemi_accel.knowledge_graph.graph_database import GraphQuery, GraphQueryResult
from chemi_accel.knowledge_graph.memory_backend import MemoryGraphBackend
from chemi_accel.knowledge_graph.ontology import EntityType, RelationshipType
from chemi_accel.knowledge_graph.query_engine import (
    ChemicalQuery,
    GraphQueryEngine,
    QueryBuilder,
    QueryCache,
    QueryOptimizer,
    QueryPlan,
    QueryStats,
    SemanticQuery,
)


class TestQueryCache:
    """Test query result caching functionality."""

    @pytest.fixture
    def cache(self):
        """Create test cache."""
        return QueryCache(max_size=5, default_ttl=10)

    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_size == 5
        assert cache.default_ttl == 10
        assert len(cache.cache) == 0

    def test_cache_put_and_get(self, cache):
        """Test basic cache operations."""
        # Create mock result
        result = GraphQueryResult(
            records=[{"test": "data"}],
            summary={"query_type": "test"},
            execution_time=0.1,
            query=GraphQuery("test", {}, "test"),
        )

        # Put in cache
        cache.put("test_query", result)

        # Get from cache
        cached_result = cache.get("test_query")

        assert cached_result is not None
        assert cached_result.records == [{"test": "data"}]
        assert cached_result.summary["query_type"] == "test"

    def test_cache_expiry(self, cache):
        """Test cache TTL expiry."""
        result = GraphQueryResult(
            records=[{"test": "data"}],
            summary={},
            execution_time=0.1,
            query=GraphQuery("test", {}, "test"),
        )

        # Put with very short TTL
        cache.put("test_query", result, ttl=0.01)  # 10ms

        # Should be available immediately
        assert cache.get("test_query") is not None

        # Wait for expiry
        time.sleep(0.02)

        # Should be expired now
        assert cache.get("test_query") is None

    def test_cache_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache to max size
        for i in range(5):
            result = GraphQueryResult(
                records=[{"id": i}],
                summary={},
                execution_time=0.1,
                query=GraphQuery(f"query_{i}", {}, "test"),
            )
            cache.put(f"query_{i}", result)

        # All should be cached
        assert len(cache.cache) == 5

        # Access first query to make it recently used
        cache.get("query_0")

        # Add one more to trigger eviction
        new_result = GraphQueryResult(
            records=[{"id": "new"}],
            summary={},
            execution_time=0.1,
            query=GraphQuery("new_query", {}, "test"),
        )
        cache.put("new_query", new_result)

        # Should still be max size
        assert len(cache.cache) == 5

        # query_0 should still be there (recently accessed)
        assert cache.get("query_0") is not None

        # Some other query should have been evicted
        assert cache.get("new_query") is not None

    def test_cache_invalidation(self, cache):
        """Test cache invalidation."""
        # Add some entries
        for i in range(3):
            result = GraphQueryResult(
                records=[{"id": i}],
                summary={},
                execution_time=0.1,
                query=GraphQuery(f"query_{i}", {}, "test"),
            )
            cache.put(f"query_{i}", result)

        assert len(cache.cache) == 3

        # Invalidate all
        cache.invalidate()

        assert len(cache.cache) == 0

    def test_cache_statistics(self, cache):
        """Test cache statistics."""
        # Add some entries
        result = GraphQueryResult(
            records=[{"test": "data"}],
            summary={},
            execution_time=0.1,
            query=GraphQuery("test", {}, "test"),
        )
        cache.put("test_query", result)

        stats = cache.get_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert "hit_rate" in stats
        assert stats["size"] == 1
        assert stats["max_size"] == 5


class TestQueryOptimizer:
    """Test query optimization functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create test optimizer."""
        return QueryOptimizer()

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert len(optimizer.optimization_rules) > 0
        assert isinstance(optimizer.statistics, dict)

    def test_query_plan_generation(self, optimizer):
        """Test query plan generation."""
        query = GraphQuery(
            query="MATCH (n:Drug) WHERE n.name = 'aspirin' RETURN n",
            parameters={"drug_name": "aspirin"},
            query_type="cypher",
        )

        plan = optimizer.optimize_query(query, "neo4j")

        assert isinstance(plan, QueryPlan)
        assert plan.query_id is not None
        assert plan.original_query == query.query
        assert plan.estimated_cost > 0
        assert plan.cache_strategy in ["aggressive", "normal", "minimal", "none"]

    def test_cost_estimation(self, optimizer):
        """Test query cost estimation."""
        # Simple query
        simple_query = GraphQuery("MATCH (n) RETURN n LIMIT 10", {}, "cypher")
        simple_cost = optimizer._estimate_query_cost(simple_query)

        # Complex query
        complex_query = GraphQuery(
            "MATCH (a)-[r]-(b) WHERE a.prop > 100 ORDER BY b.score GROUP BY a.type",
            {},
            "cypher",
        )
        complex_cost = optimizer._estimate_query_cost(complex_query)

        # Complex query should have higher cost
        assert complex_cost > simple_cost

    def test_cache_strategy_determination(self, optimizer):
        """Test cache strategy determination."""
        # High cost query should get aggressive caching
        high_cost_query = GraphQuery("COMPLEX QUERY WITH JOINS", {}, "cypher")
        plan = optimizer.optimize_query(high_cost_query, "neo4j")

        # Low cost query should get minimal caching
        low_cost_query = GraphQuery("MATCH (n) RETURN n LIMIT 1", {}, "cypher")
        simple_plan = optimizer.optimize_query(low_cost_query, "neo4j")

        # Strategy should reflect cost differences
        assert plan.cache_strategy != "none"
        assert simple_plan.cache_strategy in ["minimal", "none"]

    def test_parallelization_detection(self, optimizer):
        """Test parallel execution detection."""
        # GPU backend should support parallelization
        gpu_query = GraphQuery("FIND NODES", {}, "custom")
        gpu_plan = optimizer.optimize_query(gpu_query, "gpu_graph")

        # Memory backend with small results might not need parallelization
        memory_query = GraphQuery("MATCH (n) RETURN n LIMIT 5", {}, "cypher")
        memory_plan = optimizer.optimize_query(memory_query, "memory")

        # GPU should be more likely to parallelize
        assert gpu_plan.parallel_execution == True


class TestGraphQueryEngine:
    """Test main query engine functionality."""

    @pytest.fixture
    def backend(self):
        """Create test backend with data."""
        backend = MemoryGraphBackend({})
        backend.connect()

        # Add test data
        drug_id = backend.create_node(["Drug"], {"name": "aspirin", "type": "nsaid"})
        condition_id = backend.create_node(["Condition"], {"name": "pain"})
        backend.create_relationship(drug_id, condition_id, "TREATS", {"efficacy": 0.8})

        return backend

    @pytest.fixture
    def query_engine(self, backend):
        """Create test query engine."""
        return GraphQueryEngine(backend, enable_caching=True)

    def test_query_engine_initialization(self, query_engine):
        """Test query engine initialization."""
        assert query_engine.backend is not None
        assert query_engine.enable_caching == True
        assert query_engine.cache is not None
        assert isinstance(query_engine.optimizer, QueryOptimizer)

    def test_execute_query_basic(self, query_engine):
        """Test basic query execution."""
        query = GraphQuery(
            query="find_nodes", parameters={"labels": ["Drug"]}, query_type="find_nodes"
        )

        result = query_engine.execute_query(query)

        assert isinstance(result, GraphQueryResult)
        assert len(result.records) >= 1
        assert result.records[0]["name"] == "aspirin"
        assert result.execution_time > 0

    def test_execute_query_with_caching(self, query_engine):
        """Test query execution with caching."""
        query = GraphQuery(
            query="find_nodes", parameters={"labels": ["Drug"]}, query_type="find_nodes"
        )

        # First execution
        result1 = query_engine.execute_query(query, use_cache=True)

        # Second execution should use cache
        result2 = query_engine.execute_query(query, use_cache=True)

        # Results should be the same
        assert result1.records == result2.records

        # Check statistics show cache hit
        stats = query_engine.get_query_statistics()
        assert stats["cache_hit_rate"] > 0

    def test_execute_query_without_optimization(self, query_engine):
        """Test query execution without optimization."""
        query = GraphQuery(
            query="find_nodes", parameters={"labels": ["Drug"]}, query_type="find_nodes"
        )

        result = query_engine.execute_query(query, optimize=False)

        assert isinstance(result, GraphQueryResult)
        assert len(result.records) >= 1

    def test_execute_batch_queries(self, query_engine):
        """Test batch query execution."""
        queries = [
            GraphQuery("find_nodes", {"labels": ["Drug"]}, "find_nodes"),
            GraphQuery("find_nodes", {"labels": ["Condition"]}, "find_nodes"),
            GraphQuery("count_nodes", {}, "count_nodes"),
        ]

        results = query_engine.execute_batch_queries(queries, parallel=False)

        assert len(results) == 3
        assert all(isinstance(r, GraphQueryResult) for r in results)
        assert len(results[0].records) >= 1  # Should find drug
        assert len(results[1].records) >= 1  # Should find condition
        assert results[2].records[0]["count"] >= 2  # Should count nodes

    def test_execute_parallel_batch_queries(self, query_engine):
        """Test parallel batch query execution."""
        queries = [
            GraphQuery("find_nodes", {"labels": ["Drug"]}, "find_nodes"),
            GraphQuery("find_nodes", {"labels": ["Condition"]}, "find_nodes"),
        ]

        results = query_engine.execute_batch_queries(queries, parallel=True)

        assert len(results) == 2
        assert all(isinstance(r, GraphQueryResult) for r in results)

    def test_query_statistics(self, query_engine):
        """Test query statistics collection."""
        # Execute some queries
        query1 = GraphQuery("find_nodes", {"labels": ["Drug"]}, "find_nodes")
        query2 = GraphQuery("count_nodes", {}, "count_nodes")

        query_engine.execute_query(query1)
        query_engine.execute_query(query2)

        stats = query_engine.get_query_statistics()

        assert "total_queries" in stats
        assert "average_execution_time" in stats
        assert "cache_hit_rate" in stats
        assert "optimization_stats" in stats
        assert "recent_queries" in stats

        assert stats["total_queries"] == 2
        assert stats["average_execution_time"] > 0

    def test_cache_clearing(self, query_engine):
        """Test cache clearing functionality."""
        # Execute and cache a query
        query = GraphQuery("find_nodes", {"labels": ["Drug"]}, "find_nodes")
        query_engine.execute_query(query, use_cache=True)

        # Verify cache has content
        cache_stats = query_engine.cache.get_stats()
        assert cache_stats["size"] > 0

        # Clear cache
        query_engine.clear_cache()

        # Verify cache is empty
        cache_stats = query_engine.cache.get_stats()
        assert cache_stats["size"] == 0

    def test_query_error_handling(self, query_engine):
        """Test error handling in query execution."""
        # Create invalid query
        invalid_query = GraphQuery(
            query="invalid_operation", parameters={}, query_type="invalid_type"
        )

        result = query_engine.execute_query(invalid_query)

        # Should return error result, not crash
        assert isinstance(result, GraphQueryResult)
        assert result.metadata.get("error") == True
        assert "error" in result.summary


class TestQueryBuilder:
    """Test query builder functionality."""

    def test_query_builder_initialization(self):
        """Test query builder initialization."""
        builder = QueryBuilder("neo4j")

        assert builder.backend_type == "neo4j"
        assert builder._query_parts == []
        assert builder._parameters == {}

    def test_build_node_query_memory(self):
        """Test building node query for memory backend."""
        builder = QueryBuilder("memory")

        query = (
            builder.find_nodes(labels=["Drug"], properties={"type": "nsaid"})
            .limit(10)
            .build()
        )

        assert isinstance(query, GraphQuery)
        assert query.query_type == "find_nodes"
        assert query.parameters["labels"] == ["Drug"]
        assert query.parameters["properties"]["type"] == "nsaid"

    def test_build_cypher_query(self):
        """Test building Cypher query for Neo4j."""
        builder = QueryBuilder("neo4j")

        query = (
            builder.find_nodes(labels=["Drug"])
            .where({"name": "aspirin"})
            .limit(5)
            .build()
        )

        assert isinstance(query, GraphQuery)
        assert query.query_type == "cypher"
        assert "MATCH" in query.query
        assert "WHERE" in query.query
        assert "LIMIT 5" in query.query

    def test_build_relationship_query(self):
        """Test building relationship query."""
        builder = QueryBuilder("neo4j")

        query = builder.find_relationships(
            source_labels=["Drug"],
            target_labels=["Condition"],
            relationship_types=["TREATS"],
        ).build()

        assert isinstance(query, GraphQuery)
        assert "MATCH" in query.query
        assert "-[r:TREATS]->" in query.query

    def test_query_builder_reset(self):
        """Test query builder reset functionality."""
        builder = QueryBuilder("neo4j")

        # Build a query
        builder.find_nodes(labels=["Drug"]).limit(10)

        # Reset
        builder.reset()

        # State should be cleared
        assert builder._query_parts == []
        assert builder._parameters == {}
        assert builder._limit is None

    def test_order_by_functionality(self):
        """Test ORDER BY clause building."""
        builder = QueryBuilder("neo4j")

        query = (
            builder.find_nodes(labels=["Drug"])
            .order_by("name", ascending=True)
            .order_by("molecular_weight", ascending=False)
            .build()
        )

        assert "ORDER BY" in query.query
        assert "ASC" in query.query
        assert "DESC" in query.query


class TestChemicalQuery:
    """Test specialized chemical query builder."""

    def test_chemical_query_initialization(self):
        """Test chemical query initialization."""
        chem_query = ChemicalQuery()

        assert chem_query.entity_types == []
        assert chem_query.properties == {}
        assert chem_query.relationships == []
        assert chem_query.similarity_constraints == []
        assert chem_query.substructure_constraints == []

    def test_find_compounds(self):
        """Test compound finding functionality."""
        chem_query = ChemicalQuery().find_compounds(
            molecular_formula="C9H8O4",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            molecular_weight_range=(150, 200),
        )

        assert EntityType.SMALL_MOLECULE in chem_query.entity_types
        assert chem_query.properties["molecular_formula"] == "C9H8O4"
        assert chem_query.properties["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"
        assert chem_query.properties["molecular_weight_min"] == 150
        assert chem_query.properties["molecular_weight_max"] == 200

    def test_find_reactions(self):
        """Test reaction finding functionality."""
        chem_query = ChemicalQuery().find_reactions(
            reaction_type="synthesis", catalyst="Pd(PPh3)4", yield_threshold=0.8
        )

        assert EntityType.REACTION in chem_query.entity_types
        assert chem_query.properties["reaction_type"] == "synthesis"
        assert chem_query.properties["catalyst"] == "Pd(PPh3)4"
        assert chem_query.properties["yield_min"] == 0.8

    def test_relationship_constraints(self):
        """Test relationship constraint addition."""
        chem_query = (
            ChemicalQuery()
            .find_compounds(molecular_formula="C6H6")
            .with_relationship(RelationshipType.DISSOLVES_IN, EntityType.SOLVENT)
        )

        assert len(chem_query.relationships) == 1
        rel = chem_query.relationships[0]
        assert rel["type"] == RelationshipType.DISSOLVES_IN
        assert rel["target_type"] == EntityType.SOLVENT

    def test_similarity_constraints(self):
        """Test molecular similarity constraints."""
        reference_smiles = "CCO"

        chem_query = (
            ChemicalQuery()
            .find_compounds()
            .similar_to(reference_smiles, similarity_threshold=0.7)
        )

        assert len(chem_query.similarity_constraints) == 1
        constraint = chem_query.similarity_constraints[0]
        assert constraint["reference_smiles"] == reference_smiles
        assert constraint["threshold"] == 0.7

    def test_substructure_constraints(self):
        """Test substructure search constraints."""
        substructure = "c1ccccc1"  # Benzene ring

        chem_query = (
            ChemicalQuery().find_compounds().contains_substructure(substructure)
        )

        assert len(chem_query.substructure_constraints) == 1
        constraint = chem_query.substructure_constraints[0]
        assert constraint["substructure"] == substructure

    def test_to_cypher_query(self):
        """Test conversion to Cypher query."""
        chem_query = ChemicalQuery().find_compounds(molecular_formula="C6H6")

        # Mock backend type
        mock_backend_type = Mock()
        mock_backend_type.value = "neo4j"

        graph_query = chem_query.to_graph_query(mock_backend_type)

        assert isinstance(graph_query, GraphQuery)
        assert graph_query.query_type == "cypher"
        assert "MATCH" in graph_query.query
        assert "SMALL_MOLECULE" in graph_query.query

    def test_to_sparql_query(self):
        """Test conversion to SPARQL query."""
        chem_query = ChemicalQuery().find_compounds(molecular_formula="C6H6")

        # Mock backend type
        mock_backend_type = Mock()
        mock_backend_type.value = "jena"

        graph_query = chem_query.to_graph_query(mock_backend_type)

        assert isinstance(graph_query, GraphQuery)
        assert graph_query.query_type == "sparql"
        assert "SELECT" in graph_query.query
        assert "cao:SMALL_MOLECULE" in graph_query.query

    def test_to_generic_query(self):
        """Test conversion to generic query format."""
        chem_query = (
            ChemicalQuery()
            .find_compounds(molecular_formula="C6H6")
            .similar_to("CCO", 0.8)
        )

        # Mock backend type
        mock_backend_type = Mock()
        mock_backend_type.value = "memory"

        graph_query = chem_query.to_graph_query(mock_backend_type)

        assert isinstance(graph_query, GraphQuery)
        assert graph_query.query_type == "chemical_search"

        query_data = json.loads(graph_query.query)
        assert "entity_types" in query_data
        assert "properties" in query_data
        assert "similarity_constraints" in query_data


class TestSemanticQuery:
    """Test semantic query builder."""

    def test_semantic_query_initialization(self):
        """Test semantic query initialization."""
        sem_query = SemanticQuery()

        assert sem_query.ontology_uri == "http://chemi-accel.org/ontology/"
        assert sem_query.concepts == []
        assert sem_query.semantic_relationships == []
        assert sem_query.inference_rules == []

    def test_find_concept(self):
        """Test concept finding functionality."""
        concept_uri = "http://example.org/ontology/Drug"

        sem_query = SemanticQuery().find_concept(concept_uri, include_subclasses=True)

        assert len(sem_query.concepts) == 1
        concept = sem_query.concepts[0]
        assert concept["uri"] == concept_uri
        assert concept["include_subclasses"] == True

    def test_semantic_relationships(self):
        """Test semantic relationship constraints."""
        relationship_uri = "http://example.org/ontology/treats"
        target_concept = "http://example.org/ontology/Disease"

        sem_query = (
            SemanticQuery()
            .find_concept("http://example.org/ontology/Drug")
            .with_semantic_relationship(relationship_uri, target_concept)
        )

        assert len(sem_query.semantic_relationships) == 1
        rel = sem_query.semantic_relationships[0]
        assert rel["relationship"] == relationship_uri
        assert rel["target_concept"] == target_concept

    def test_inference_rules(self):
        """Test inference rule addition."""
        rule_name = "transitivity"
        parameters = {"property": "subClassOf"}

        sem_query = (
            SemanticQuery()
            .find_concept("http://example.org/ontology/Chemical")
            .with_inference(rule_name, parameters)
        )

        assert len(sem_query.inference_rules) == 1
        rule = sem_query.inference_rules[0]
        assert rule["rule"] == rule_name
        assert rule["parameters"] == parameters

    def test_to_sparql(self):
        """Test conversion to SPARQL query."""
        sem_query = (
            SemanticQuery()
            .find_concept("http://example.org/ontology/Drug", include_subclasses=True)
            .with_semantic_relationship(
                "http://example.org/ontology/treats",
                "http://example.org/ontology/Disease",
            )
        )

        sparql_query = sem_query.to_sparql()

        assert isinstance(sparql_query, GraphQuery)
        assert sparql_query.query_type == "sparql"
        assert "SELECT DISTINCT ?resource WHERE" in sparql_query.query
        assert "rdfs:subClassOf*" in sparql_query.query  # Should include subclasses
        assert "treats" in sparql_query.query


class TestQueryEngineIntegration:
    """Integration tests for query engine components."""

    @pytest.fixture
    def backend_with_data(self):
        """Create backend with comprehensive test data."""
        backend = MemoryGraphBackend({})
        backend.connect()

        # Create a small chemical knowledge graph
        aspirin_id = backend.create_node(
            ["Chemical", "Drug"],
            {"name": "aspirin", "formula": "C9H8O4", "molecular_weight": 180.16},
        )
        ibuprofen_id = backend.create_node(
            ["Chemical", "Drug"],
            {"name": "ibuprofen", "formula": "C13H18O2", "molecular_weight": 206.29},
        )
        pain_id = backend.create_node(
            ["Condition"], {"name": "pain", "severity": "moderate"}
        )
        fever_id = backend.create_node(
            ["Condition"], {"name": "fever", "temperature": "high"}
        )

        # Create relationships
        backend.create_relationship(aspirin_id, pain_id, "TREATS", {"efficacy": 0.8})
        backend.create_relationship(aspirin_id, fever_id, "REDUCES", {"efficacy": 0.7})
        backend.create_relationship(ibuprofen_id, pain_id, "TREATS", {"efficacy": 0.9})

        return backend

    def test_chemical_query_execution(self, backend_with_data):
        """Test executing chemical queries on real data."""
        query_engine = GraphQueryEngine(backend_with_data)

        # Build chemical query
        chem_query = ChemicalQuery().find_compounds(molecular_weight_range=(150, 200))

        # Mock backend type for conversion
        mock_backend_type = Mock()
        mock_backend_type.value = "memory"

        result = query_engine.execute_chemical_query(chem_query)

        assert isinstance(result, GraphQueryResult)
        # Should find compounds in the weight range
        assert len(result.records) >= 1

    def test_query_builder_with_real_backend(self, backend_with_data):
        """Test query builder with real backend execution."""
        query_engine = GraphQueryEngine(backend_with_data)

        # Build query
        builder = QueryBuilder("memory")
        query = builder.find_nodes(labels=["Drug"]).limit(5).build()

        result = query_engine.execute_query(query)

        assert isinstance(result, GraphQueryResult)
        assert len(result.records) >= 2  # Should find aspirin and ibuprofen

        # Verify drug properties
        drug_names = [record["name"] for record in result.records]
        assert "aspirin" in drug_names
        assert "ibuprofen" in drug_names

    def test_optimization_with_real_queries(self, backend_with_data):
        """Test query optimization with real queries."""
        query_engine = GraphQueryEngine(backend_with_data)

        # Create a complex query that could benefit from optimization
        query = GraphQuery(
            query="find_nodes",
            parameters={"labels": ["Drug"], "properties": {"molecular_weight": 180.16}},
            query_type="find_nodes",
        )

        # Execute with optimization
        optimized_result = query_engine.execute_query(query, optimize=True)

        # Execute without optimization
        unoptimized_result = query_engine.execute_query(query, optimize=False)

        # Both should return same results
        assert len(optimized_result.records) == len(unoptimized_result.records)

        # Check that optimization was applied
        stats = query_engine.get_query_statistics()
        assert stats["optimization_stats"] is not None

    def test_caching_effectiveness(self, backend_with_data):
        """Test caching effectiveness with repeated queries."""
        query_engine = GraphQueryEngine(backend_with_data, enable_caching=True)

        query = GraphQuery(
            query="find_nodes", parameters={"labels": ["Drug"]}, query_type="find_nodes"
        )

        # Execute multiple times
        results = []
        for _ in range(5):
            result = query_engine.execute_query(query, use_cache=True)
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert result.records == results[0].records

        # Cache hit rate should be high
        stats = query_engine.get_query_statistics()
        assert stats["cache_hit_rate"] >= 0.8  # 4 out of 5 should be cache hits

    def test_error_recovery_in_complex_scenario(self, backend_with_data):
        """Test error recovery in complex query scenarios."""
        query_engine = GraphQueryEngine(backend_with_data)

        # Mix of valid and invalid queries
        queries = [
            GraphQuery("find_nodes", {"labels": ["Drug"]}, "find_nodes"),  # Valid
            GraphQuery("invalid_op", {}, "invalid_type"),  # Invalid
            GraphQuery("count_nodes", {}, "count_nodes"),  # Valid
            GraphQuery("malformed", {"bad": "params"}, "nonexistent"),  # Invalid
        ]

        results = query_engine.execute_batch_queries(queries, parallel=False)

        assert len(results) == 4

        # First and third should succeed
        assert not results[0].metadata.get("error", False)
        assert not results[2].metadata.get("error", False)

        # Second and fourth should fail gracefully
        assert results[1].metadata.get("error", False)
        assert results[3].metadata.get("error", False)

        # But should have error information, not crash
        assert "error" in results[1].summary
        assert "error" in results[3].summary

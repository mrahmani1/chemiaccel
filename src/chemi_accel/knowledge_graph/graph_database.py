"""
Graph Database Integration and Management

Multi-backend graph database system supporting Neo4j, Apache Jena, and
custom GPU-optimized graph storage for the IC-KG.
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class GraphDatabaseType(Enum):
    """Supported graph database types."""

    NEO4J = "neo4j"
    JENA = "jena"
    GPU_GRAPH = "gpu_graph"
    MEMORY = "memory"


@dataclass
class GraphQuery:
    """Graph query specification."""

    query: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    query_type: str = "cypher"  # cypher, sparql, custom
    timeout: int = 30
    explain: bool = False
    profiling: bool = False


@dataclass
class GraphQueryResult:
    """Graph query execution result."""

    records: List[Dict[str, Any]]
    summary: Dict[str, Any]
    execution_time: float
    query: GraphQuery
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexConfig:
    """Configuration for graph database indexes."""

    entity_type: str
    properties: List[str]
    index_type: str = "btree"  # btree, fulltext, composite
    unique: bool = False
    background: bool = True


@dataclass
class BulkLoadConfig:
    """Configuration for bulk data loading."""

    batch_size: int = 1000
    parallel_workers: int = 4
    validation: bool = True
    create_missing_nodes: bool = True
    merge_duplicates: bool = True
    constraint_checking: bool = True
    progress_callback: Optional[callable] = None


class GraphDatabaseBackend(ABC):
    """Abstract base class for graph database backends."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend_type = self._get_backend_type()
        self.connection = None
        self.indexes = {}
        self.statistics = {
            "queries_executed": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "last_activity": None,
        }

    @abstractmethod
    def _get_backend_type(self) -> GraphDatabaseType:
        """Get the backend type."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the graph database."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to the graph database."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if database is connected."""
        pass

    @abstractmethod
    def execute_query(self, query: GraphQuery) -> GraphQueryResult:
        """Execute a graph query."""
        pass

    @abstractmethod
    def create_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
        """Create a node with labels and properties."""
        pass

    @abstractmethod
    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
    ) -> str:
        """Create a relationship between nodes."""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID."""
        pass

    @abstractmethod
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """Delete node by ID."""
        pass

    @abstractmethod
    def find_nodes(
        self,
        labels: List[str] = None,
        properties: Dict[str, Any] = None,
        limit: int = None,
    ) -> List[Dict[str, Any]]:
        """Find nodes matching criteria."""
        pass

    @abstractmethod
    def create_index(self, config: IndexConfig) -> bool:
        """Create database index."""
        pass

    @abstractmethod
    def drop_index(self, index_name: str) -> bool:
        """Drop database index."""
        pass

    @abstractmethod
    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all database indexes."""
        pass

    @abstractmethod
    def bulk_load_nodes(
        self, nodes: List[Dict[str, Any]], config: BulkLoadConfig = None
    ) -> Dict[str, Any]:
        """Bulk load nodes into database."""
        pass

    @abstractmethod
    def bulk_load_relationships(
        self, relationships: List[Dict[str, Any]], config: BulkLoadConfig = None
    ) -> Dict[str, Any]:
        """Bulk load relationships into database."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass

    @abstractmethod
    def clear_database(self) -> bool:
        """Clear all data from database."""
        pass

    def batch_process(
        self, operations: List[Dict[str, Any]], batch_size: int = 100
    ) -> List[Any]:
        """Process operations in batches."""
        results = []

        for i in range(0, len(operations), batch_size):
            batch = operations[i : i + batch_size]
            batch_results = []

            for op in batch:
                try:
                    result = self._process_operation(op)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Operation failed: {e}")
                    batch_results.append({"error": str(e)})

            results.extend(batch_results)

        return results

    def _process_operation(self, operation: Dict[str, Any]) -> Any:
        """Process a single operation."""
        op_type = operation.get("type")

        if op_type == "create_node":
            return self.create_node(operation["labels"], operation["properties"])
        elif op_type == "create_relationship":
            return self.create_relationship(
                operation["source_id"],
                operation["target_id"],
                operation["relationship_type"],
                operation.get("properties", {}),
            )
        elif op_type == "update_node":
            return self.update_node(operation["node_id"], operation["properties"])
        elif op_type == "delete_node":
            return self.delete_node(operation["node_id"])
        else:
            raise ValueError(f"Unsupported operation type: {op_type}")

    def validate_connection(self) -> Dict[str, Any]:
        """Validate database connection and basic functionality."""
        validation_results = {
            "connected": False,
            "read_access": False,
            "write_access": False,
            "index_support": False,
            "bulk_load_support": False,
            "errors": [],
        }

        try:
            # Test connection
            validation_results["connected"] = self.is_connected()

            if validation_results["connected"]:
                # Test read access
                try:
                    stats = self.get_statistics()
                    validation_results["read_access"] = True
                except Exception as e:
                    validation_results["errors"].append(f"Read access failed: {e}")

                # Test write access
                try:
                    test_id = str(uuid.uuid4())
                    node_id = self.create_node(["Test"], {"test_id": test_id})
                    if node_id:
                        self.delete_node(node_id)
                        validation_results["write_access"] = True
                except Exception as e:
                    validation_results["errors"].append(f"Write access failed: {e}")

                # Test index support
                try:
                    indexes = self.list_indexes()
                    validation_results["index_support"] = True
                except Exception as e:
                    validation_results["errors"].append(f"Index support failed: {e}")

                # Test bulk load support
                try:
                    result = self.bulk_load_nodes([], BulkLoadConfig(batch_size=1))
                    validation_results["bulk_load_support"] = True
                except Exception as e:
                    validation_results["errors"].append(
                        f"Bulk load support failed: {e}"
                    )

        except Exception as e:
            validation_results["errors"].append(f"Connection validation failed: {e}")

        return validation_results

    def optimize_for_workload(self, workload_type: str) -> Dict[str, Any]:
        """Optimize database configuration for specific workload."""
        optimization_results = {
            "workload_type": workload_type,
            "optimizations_applied": [],
            "recommendations": [],
            "performance_impact": "unknown",
        }

        if workload_type == "read_heavy":
            optimization_results["recommendations"].extend(
                [
                    "Create indexes on frequently queried properties",
                    "Enable query result caching",
                    "Use read replicas if available",
                ]
            )
        elif workload_type == "write_heavy":
            optimization_results["recommendations"].extend(
                [
                    "Use bulk loading for large datasets",
                    "Disable non-essential constraints during loading",
                    "Batch write operations",
                ]
            )
        elif workload_type == "mixed":
            optimization_results["recommendations"].extend(
                [
                    "Balance read and write optimizations",
                    "Monitor query performance",
                    "Use appropriate batch sizes",
                ]
            )

        return optimization_results


def get_graph_database(backend: str = "memory", **config) -> GraphDatabaseBackend:
    """
    Get a graph database backend instance.

    Args:
        backend: Backend type ("neo4j", "jena", "gpu_graph", "memory")
        **config: Backend-specific configuration

    Returns:
        GraphDatabaseBackend instance
    """
    backend = backend.lower()

    if backend == "neo4j":
        from .neo4j_backend import Neo4jBackend

        return Neo4jBackend(config)
    elif backend == "jena":
        from .jena_backend import JenaBackend

        return JenaBackend(config)
    elif backend == "gpu_graph":
        from .gpu_graph_backend import GPUGraphBackend

        return GPUGraphBackend(config)
    elif backend == "memory":
        from .memory_backend import MemoryGraphBackend

        return MemoryGraphBackend(config)
    else:
        raise ValueError(f"Unsupported graph database backend: {backend}")


def _detect_best_backend() -> str:
    """Detect the best available graph database backend."""
    # Check for Neo4j availability
    try:
        import neo4j

        return "neo4j"
    except ImportError:
        pass

    # Check for GPU capabilities
    try:
        import cugraph
        import cupy

        return "gpu_graph"
    except ImportError:
        pass

    # Check for Jena availability
    try:
        import rdflib

        return "jena"
    except ImportError:
        pass

    # Fallback to memory backend
    return "memory"


class GraphDatabaseManager:
    """Manages multiple graph database backends and provides unified interface."""

    def __init__(self):
        self.backends = {}
        self.default_backend = None
        self.routing_rules = {}

    def register_backend(
        self, name: str, backend: GraphDatabaseBackend, is_default: bool = False
    ):
        """Register a graph database backend."""
        self.backends[name] = backend
        if is_default or self.default_backend is None:
            self.default_backend = name

    def get_backend(self, name: str = None) -> GraphDatabaseBackend:
        """Get a registered backend."""
        if name is None:
            name = self.default_backend

        if name not in self.backends:
            raise ValueError(f"Backend '{name}' not registered")

        return self.backends[name]

    def route_query(self, query: GraphQuery) -> str:
        """Route query to appropriate backend based on routing rules."""
        # Simple routing based on query type for now
        if query.query_type == "sparql":
            for name, backend in self.backends.items():
                if backend.backend_type == GraphDatabaseType.JENA:
                    return name

        return self.default_backend

    def execute_distributed_query(self, query: GraphQuery) -> List[GraphQueryResult]:
        """Execute query across multiple backends."""
        results = []

        for name, backend in self.backends.items():
            try:
                result = backend.execute_query(query)
                result.metadata["backend"] = name
                results.append(result)
            except Exception as e:
                logger.error(f"Query failed on backend {name}: {e}")
                # Create error result
                error_result = GraphQueryResult(
                    records=[],
                    summary={"error": str(e)},
                    execution_time=0.0,
                    query=query,
                    metadata={"backend": name, "error": True},
                )
                results.append(error_result)

        return results

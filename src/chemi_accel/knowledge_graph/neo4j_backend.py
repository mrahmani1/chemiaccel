"""
Neo4j Graph Database Backend

High-performance Neo4j integration for the IC-KG with optimized
CRUD operations, batch processing, and indexing.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .graph_database import (
    BulkLoadConfig,
    GraphDatabaseBackend,
    GraphDatabaseType,
    GraphQuery,
    GraphQueryResult,
    IndexConfig,
)

logger = logging.getLogger(__name__)


class Neo4jBackend(GraphDatabaseBackend):
    """Neo4j database backend implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.driver = None
        self.session = None
        self._setup_connection_config()

    def _get_backend_type(self) -> GraphDatabaseType:
        return GraphDatabaseType.NEO4J

    def _setup_connection_config(self):
        """Setup Neo4j connection configuration."""
        self.uri = self.config.get("uri", "bolt://localhost:7687")
        self.username = self.config.get("username", "neo4j")
        self.password = self.config.get("password", "password")
        self.database = self.config.get("database", "neo4j")
        self.max_connection_lifetime = self.config.get("max_connection_lifetime", 3600)
        self.max_connection_pool_size = self.config.get("max_connection_pool_size", 100)
        self.connection_timeout = self.config.get("connection_timeout", 30)

    def connect(self) -> bool:
        """Establish connection to Neo4j database."""
        try:
            import neo4j

            self.driver = neo4j.GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=self.max_connection_lifetime,
                max_connection_pool_size=self.max_connection_pool_size,
                connection_timeout=self.connection_timeout,
            )

            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1 as test")

            logger.info(f"Connected to Neo4j at {self.uri}")
            return True

        except ImportError:
            logger.error("Neo4j driver not available. Install with: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def disconnect(self) -> bool:
        """Close connection to Neo4j database."""
        try:
            if self.driver:
                self.driver.close()
                self.driver = None
            logger.info("Disconnected from Neo4j")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Neo4j: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if database is connected."""
        try:
            if not self.driver:
                return False

            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False

    def execute_query(self, query: GraphQuery) -> GraphQueryResult:
        """Execute a Cypher query."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Neo4j database")

        start_time = time.time()

        try:
            with self.driver.session(database=self.database) as session:
                if query.explain:
                    # Execute with EXPLAIN
                    explain_query = f"EXPLAIN {query.query}"
                    result = session.run(explain_query, query.parameters)
                    records = [dict(record) for record in result]
                    summary = result.consume()
                else:
                    # Normal execution
                    result = session.run(
                        query.query, query.parameters, timeout=query.timeout
                    )
                    records = [dict(record) for record in result]
                    summary = result.consume()

                execution_time = time.time() - start_time

                # Convert summary to dict
                summary_dict = {
                    "query_type": summary.query_type,
                    "counters": {
                        "nodes_created": summary.counters.nodes_created,
                        "nodes_deleted": summary.counters.nodes_deleted,
                        "relationships_created": summary.counters.relationships_created,
                        "relationships_deleted": summary.counters.relationships_deleted,
                        "properties_set": summary.counters.properties_set,
                        "labels_added": summary.counters.labels_added,
                        "labels_removed": summary.counters.labels_removed,
                        "indexes_added": summary.counters.indexes_added,
                        "indexes_removed": summary.counters.indexes_removed,
                        "constraints_added": summary.counters.constraints_added,
                        "constraints_removed": summary.counters.constraints_removed,
                    },
                    "result_available_after": summary.result_available_after,
                    "result_consumed_after": summary.result_consumed_after,
                }

                if query.profiling and hasattr(summary, "profile"):
                    summary_dict["profile"] = summary.profile

                # Update statistics
                self.statistics["queries_executed"] += 1
                self.statistics["nodes_created"] += summary.counters.nodes_created
                self.statistics[
                    "relationships_created"
                ] += summary.counters.relationships_created
                self.statistics["last_activity"] = datetime.now()

                return GraphQueryResult(
                    records=records,
                    summary=summary_dict,
                    execution_time=execution_time,
                    query=query,
                )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed: {e}")

            return GraphQueryResult(
                records=[],
                summary={"error": str(e)},
                execution_time=execution_time,
                query=query,
                metadata={"error": True},
            )

    def create_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
        """Create a node with labels and properties."""
        node_id = str(uuid.uuid4())
        properties["_id"] = node_id
        properties["_created_at"] = datetime.now().isoformat()

        labels_str = ":".join(labels)
        query = f"CREATE (n:{labels_str}) SET n = $properties RETURN n._id as id"

        result = self.execute_query(GraphQuery(query, {"properties": properties}))

        if result.records:
            return result.records[0]["id"]
        else:
            raise RuntimeError("Failed to create node")

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
    ) -> str:
        """Create a relationship between nodes."""
        if properties is None:
            properties = {}

        rel_id = str(uuid.uuid4())
        properties["_id"] = rel_id
        properties["_created_at"] = datetime.now().isoformat()

        query = """
        MATCH (a {_id: $source_id}), (b {_id: $target_id})
        CREATE (a)-[r:{}]->(b)
        SET r = $properties
        RETURN r._id as id
        """.format(
            relationship_type
        )

        result = self.execute_query(
            GraphQuery(
                query,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "properties": properties,
                },
            )
        )

        if result.records:
            return result.records[0]["id"]
        else:
            raise RuntimeError("Failed to create relationship")

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID."""
        query = "MATCH (n {_id: $node_id}) RETURN n, labels(n) as labels"
        result = self.execute_query(GraphQuery(query, {"node_id": node_id}))

        if result.records:
            record = result.records[0]
            node_data = dict(record["n"])
            node_data["labels"] = record["labels"]
            return node_data

        return None

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        properties["_updated_at"] = datetime.now().isoformat()

        query = """
        MATCH (n {_id: $node_id})
        SET n += $properties
        RETURN count(n) as updated
        """

        result = self.execute_query(
            GraphQuery(query, {"node_id": node_id, "properties": properties})
        )

        return result.records and result.records[0]["updated"] > 0

    def delete_node(self, node_id: str) -> bool:
        """Delete node by ID."""
        query = """
        MATCH (n {_id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """

        result = self.execute_query(GraphQuery(query, {"node_id": node_id}))
        return result.records and result.records[0]["deleted"] > 0

    def find_nodes(
        self,
        labels: List[str] = None,
        properties: Dict[str, Any] = None,
        limit: int = None,
    ) -> List[Dict[str, Any]]:
        """Find nodes matching criteria."""
        query_parts = ["MATCH (n"]
        params = {}

        if labels:
            query_parts.append(":" + ":".join(labels))

        if properties:
            conditions = []
            for key, value in properties.items():
                param_name = f"prop_{key}"
                conditions.append(f"n.{key} = ${param_name}")
                params[param_name] = value

            if conditions:
                query_parts.append(") WHERE " + " AND ".join(conditions))
            else:
                query_parts.append(")")
        else:
            query_parts.append(")")

        query_parts.append(" RETURN n, labels(n) as labels")

        if limit:
            query_parts.append(f" LIMIT {limit}")

        query = "".join(query_parts)
        result = self.execute_query(GraphQuery(query, params))

        nodes = []
        for record in result.records:
            node_data = dict(record["n"])
            node_data["labels"] = record["labels"]
            nodes.append(node_data)

        return nodes

    def create_index(self, config: IndexConfig) -> bool:
        """Create database index."""
        try:
            if config.index_type == "btree":
                if config.unique:
                    # Create unique constraint (which includes index)
                    query = (
                        f"CREATE CONSTRAINT {config.entity_type}_{config.properties[0]}_unique "
                        f"FOR (n:{config.entity_type}) REQUIRE n.{config.properties[0]} IS UNIQUE"
                    )
                else:
                    # Create regular index
                    properties_str = ", ".join(
                        [f"n.{prop}" for prop in config.properties]
                    )
                    query = (
                        f"CREATE INDEX {config.entity_type}_{'_'.join(config.properties)}_index "
                        f"FOR (n:{config.entity_type}) ON ({properties_str})"
                    )

            elif config.index_type == "fulltext":
                properties_str = ", ".join([f"n.{prop}" for prop in config.properties])
                query = (
                    f"CREATE FULLTEXT INDEX {config.entity_type}_{'_'.join(config.properties)}_fulltext "
                    f"FOR (n:{config.entity_type}) ON EACH [{properties_str}]"
                )

            else:
                raise ValueError(f"Unsupported index type: {config.index_type}")

            result = self.execute_query(GraphQuery(query))

            # Store index configuration
            index_name = f"{config.entity_type}_{'_'.join(config.properties)}_{config.index_type}"
            self.indexes[index_name] = config

            return not result.metadata.get("error", False)

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def drop_index(self, index_name: str) -> bool:
        """Drop database index."""
        try:
            query = f"DROP INDEX {index_name}"
            result = self.execute_query(GraphQuery(query))

            if index_name in self.indexes:
                del self.indexes[index_name]

            return not result.metadata.get("error", False)

        except Exception as e:
            logger.error(f"Failed to drop index: {e}")
            return False

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all database indexes."""
        query = "SHOW INDEXES"
        result = self.execute_query(GraphQuery(query))
        return result.records

    def bulk_load_nodes(
        self, nodes: List[Dict[str, Any]], config: BulkLoadConfig = None
    ) -> Dict[str, Any]:
        """Bulk load nodes into database."""
        if config is None:
            config = BulkLoadConfig()

        results = {
            "total_nodes": len(nodes),
            "created_nodes": 0,
            "failed_nodes": 0,
            "errors": [],
        }

        if not nodes:
            return results

        # Process in batches
        for i in range(0, len(nodes), config.batch_size):
            batch = nodes[i : i + config.batch_size]

            try:
                # Prepare batch data
                batch_data = []
                for node in batch:
                    node_data = node.copy()
                    if "_id" not in node_data:
                        node_data["_id"] = str(uuid.uuid4())
                    node_data["_created_at"] = datetime.now().isoformat()
                    batch_data.append(node_data)

                # Execute batch insert
                query = """
                UNWIND $batch as item
                CREATE (n)
                SET n = item.properties
                WITH n, item.labels as labels
                CALL apoc.create.addLabels(n, labels) YIELD node
                RETURN count(node) as created
                """

                # Prepare data for Neo4j
                batch_params = []
                for node_data in batch_data:
                    labels = node_data.pop("labels", ["Node"])
                    batch_params.append({"properties": node_data, "labels": labels})

                result = self.execute_query(GraphQuery(query, {"batch": batch_params}))

                if result.records:
                    created = result.records[0]["created"]
                    results["created_nodes"] += created
                else:
                    results["failed_nodes"] += len(batch)
                    results["errors"].append(f"Batch {i//config.batch_size + 1} failed")

                # Progress callback
                if config.progress_callback:
                    progress = min((i + config.batch_size) / len(nodes), 1.0)
                    config.progress_callback(
                        progress, f"Processed {i + len(batch)} nodes"
                    )

            except Exception as e:
                results["failed_nodes"] += len(batch)
                results["errors"].append(f"Batch {i//config.batch_size + 1}: {str(e)}")
                logger.error(f"Batch load failed: {e}")

        return results

    def bulk_load_relationships(
        self, relationships: List[Dict[str, Any]], config: BulkLoadConfig = None
    ) -> Dict[str, Any]:
        """Bulk load relationships into database."""
        if config is None:
            config = BulkLoadConfig()

        results = {
            "total_relationships": len(relationships),
            "created_relationships": 0,
            "failed_relationships": 0,
            "errors": [],
        }

        if not relationships:
            return results

        # Process in batches
        for i in range(0, len(relationships), config.batch_size):
            batch = relationships[i : i + config.batch_size]

            try:
                # Prepare batch data
                batch_data = []
                for rel in batch:
                    rel_data = rel.copy()
                    if "_id" not in rel_data:
                        rel_data["_id"] = str(uuid.uuid4())
                    rel_data["_created_at"] = datetime.now().isoformat()
                    batch_data.append(rel_data)

                # Execute batch insert
                query = """
                UNWIND $batch as item
                MATCH (a {_id: item.source_id}), (b {_id: item.target_id})
                CREATE (a)-[r]->(b)
                SET r = item.properties
                WITH r, item.relationship_type as relType
                CALL apoc.refactor.setType(r, relType) YIELD output
                RETURN count(output) as created
                """

                # Prepare data for Neo4j
                batch_params = []
                for rel_data in batch_data:
                    relationship_type = rel_data.pop("relationship_type", "RELATES_TO")
                    source_id = rel_data.pop("source_id")
                    target_id = rel_data.pop("target_id")

                    batch_params.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "relationship_type": relationship_type,
                            "properties": rel_data,
                        }
                    )

                result = self.execute_query(GraphQuery(query, {"batch": batch_params}))

                if result.records:
                    created = result.records[0]["created"]
                    results["created_relationships"] += created
                else:
                    results["failed_relationships"] += len(batch)
                    results["errors"].append(f"Batch {i//config.batch_size + 1} failed")

                # Progress callback
                if config.progress_callback:
                    progress = min((i + config.batch_size) / len(relationships), 1.0)
                    config.progress_callback(
                        progress, f"Processed {i + len(batch)} relationships"
                    )

            except Exception as e:
                results["failed_relationships"] += len(batch)
                results["errors"].append(f"Batch {i//config.batch_size + 1}: {str(e)}")
                logger.error(f"Batch relationship load failed: {e}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            # Get basic stats
            stats_query = """
            MATCH (n)
            OPTIONAL MATCH ()-[r]->()
            RETURN count(DISTINCT n) as nodes, count(r) as relationships
            """

            result = self.execute_query(GraphQuery(stats_query))
            db_stats = (
                result.records[0]
                if result.records
                else {"nodes": 0, "relationships": 0}
            )

            # Get label distribution
            labels_query = """
            MATCH (n)
            UNWIND labels(n) as label
            RETURN label, count(*) as count
            ORDER BY count DESC
            """

            result = self.execute_query(GraphQuery(labels_query))
            label_distribution = {r["label"]: r["count"] for r in result.records}

            # Combine with internal statistics
            combined_stats = {
                **self.statistics,
                "database": {
                    "total_nodes": db_stats["nodes"],
                    "total_relationships": db_stats["relationships"],
                    "label_distribution": label_distribution,
                },
                "connection": {
                    "uri": self.uri,
                    "database": self.database,
                    "connected": self.is_connected(),
                },
            }

            return combined_stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {**self.statistics, "error": str(e)}

    def clear_database(self) -> bool:
        """Clear all data from database."""
        try:
            query = "MATCH (n) DETACH DELETE n"
            result = self.execute_query(GraphQuery(query))

            # Reset statistics
            self.statistics.update({"nodes_created": 0, "relationships_created": 0})

            return not result.metadata.get("error", False)

        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

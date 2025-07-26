"""
Memory Graph Backend

Simple in-memory graph database for testing and fallback scenarios.
Provides basic graph operations without external dependencies.
"""

import logging
import time
import uuid
from collections import defaultdict
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


class MemoryGraphBackend(GraphDatabaseBackend):
    """Simple in-memory graph database backend."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.nodes = {}
        self.relationships = {}
        self.adjacency_list = defaultdict(list)
        self.reverse_adjacency = defaultdict(list)
        self.label_index = defaultdict(set)
        self.property_indexes = defaultdict(dict)

    def _get_backend_type(self) -> GraphDatabaseType:
        return GraphDatabaseType.MEMORY

    def connect(self) -> bool:
        """Initialize in-memory structures."""
        try:
            # Already initialized in __init__
            logger.info("Connected to memory graph backend")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize memory backend: {e}")
            return False

    def disconnect(self) -> bool:
        """Clear memory structures."""
        try:
            self.nodes.clear()
            self.relationships.clear()
            self.adjacency_list.clear()
            self.reverse_adjacency.clear()
            self.label_index.clear()
            self.property_indexes.clear()

            logger.info("Disconnected from memory graph backend")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from memory backend: {e}")
            return False

    def is_connected(self) -> bool:
        """Memory backend is always connected if initialized."""
        return True

    def execute_query(self, query: GraphQuery) -> GraphQueryResult:
        """Execute a simple graph query."""
        start_time = time.time()

        try:
            query_type = query.query_type.lower()
            records = []

            if query_type == "find_nodes":
                records = self._query_find_nodes(query.parameters)
            elif query_type == "find_paths":
                records = self._query_find_paths(query.parameters)
            elif query_type == "neighbors":
                records = self._query_get_neighbors(query.parameters)
            elif query_type == "count_nodes":
                records = [{"count": len(self.nodes)}]
            elif query_type == "count_relationships":
                records = [{"count": len(self.relationships)}]
            elif query_type == "node_degrees":
                records = self._query_node_degrees(query.parameters)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")

            execution_time = time.time() - start_time

            summary = {
                "query_type": query_type,
                "result_count": len(records),
                "execution_time": execution_time,
            }

            # Update statistics
            self.statistics["queries_executed"] += 1
            self.statistics["last_activity"] = datetime.now()

            return GraphQueryResult(
                records=records,
                summary=summary,
                execution_time=execution_time,
                query=query,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Memory query execution failed: {e}")

            return GraphQueryResult(
                records=[],
                summary={"error": str(e)},
                execution_time=execution_time,
                query=query,
                metadata={"error": True},
            )

    def create_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
        """Create a node in memory."""
        node_id = str(uuid.uuid4())

        node_data = {
            "_id": node_id,
            "labels": labels,
            "_created_at": datetime.now().isoformat(),
            **properties,
        }

        # Store node
        self.nodes[node_id] = node_data

        # Update label index
        for label in labels:
            self.label_index[label].add(node_id)

        # Update property indexes
        self._update_property_indexes(node_id, properties)

        self.statistics["nodes_created"] += 1
        return node_id

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
    ) -> str:
        """Create a relationship in memory."""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target node not found")

        if properties is None:
            properties = {}

        rel_id = str(uuid.uuid4())

        rel_data = {
            "_id": rel_id,
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type,
            "_created_at": datetime.now().isoformat(),
            **properties,
        }

        # Store relationship
        self.relationships[rel_id] = rel_data

        # Update adjacency lists
        self.adjacency_list[source_id].append(
            {
                "target_id": target_id,
                "relationship_id": rel_id,
                "relationship_type": relationship_type,
            }
        )

        self.reverse_adjacency[target_id].append(
            {
                "source_id": source_id,
                "relationship_id": rel_id,
                "relationship_type": relationship_type,
            }
        )

        self.statistics["relationships_created"] += 1
        return rel_id

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        if node_id not in self.nodes:
            return False

        # Update properties
        self.nodes[node_id].update(properties)
        self.nodes[node_id]["_updated_at"] = datetime.now().isoformat()

        # Update property indexes
        self._update_property_indexes(node_id, properties)

        return True

    def delete_node(self, node_id: str) -> bool:
        """Delete node by ID."""
        if node_id not in self.nodes:
            return False

        node_data = self.nodes[node_id]

        # Remove from label index
        for label in node_data.get("labels", []):
            self.label_index[label].discard(node_id)

        # Remove from property indexes
        self._remove_from_property_indexes(node_id)

        # Remove associated relationships
        relationships_to_remove = []

        # Outgoing relationships
        for rel_info in self.adjacency_list[node_id]:
            relationships_to_remove.append(rel_info["relationship_id"])

        # Incoming relationships
        for rel_info in self.reverse_adjacency[node_id]:
            relationships_to_remove.append(rel_info["relationship_id"])

        # Remove relationships
        for rel_id in relationships_to_remove:
            if rel_id in self.relationships:
                rel_data = self.relationships[rel_id]
                source_id = rel_data["source_id"]
                target_id = rel_data["target_id"]

                # Remove from adjacency lists
                self.adjacency_list[source_id] = [
                    r
                    for r in self.adjacency_list[source_id]
                    if r["relationship_id"] != rel_id
                ]
                self.reverse_adjacency[target_id] = [
                    r
                    for r in self.reverse_adjacency[target_id]
                    if r["relationship_id"] != rel_id
                ]

                del self.relationships[rel_id]

        # Remove from adjacency lists
        del self.adjacency_list[node_id]
        del self.reverse_adjacency[node_id]

        # Remove node
        del self.nodes[node_id]

        return True

    def find_nodes(
        self,
        labels: List[str] = None,
        properties: Dict[str, Any] = None,
        limit: int = None,
    ) -> List[Dict[str, Any]]:
        """Find nodes matching criteria."""
        candidate_nodes = set(self.nodes.keys())

        # Filter by labels
        if labels:
            label_candidates = set()
            for label in labels:
                label_candidates.update(self.label_index.get(label, set()))
            candidate_nodes &= label_candidates

        # Filter by properties
        if properties:
            for key, value in properties.items():
                if key in self.property_indexes:
                    prop_candidates = set(self.property_indexes[key].get(value, []))
                    candidate_nodes &= prop_candidates
                else:
                    # Linear search for non-indexed properties
                    prop_candidates = set()
                    for node_id in candidate_nodes:
                        node_data = self.nodes[node_id]
                        if node_data.get(key) == value:
                            prop_candidates.add(node_id)
                    candidate_nodes &= prop_candidates

        # Get node data
        results = [self.nodes[node_id] for node_id in candidate_nodes]

        # Apply limit
        if limit:
            results = results[:limit]

        return results

    def create_index(self, config: IndexConfig) -> bool:
        """Create property index."""
        try:
            index_name = f"{config.entity_type}_{'_'.join(config.properties)}_{config.index_type}"

            # For memory backend, we create simple property indexes
            if len(config.properties) == 1:
                prop = config.properties[0]

                # Build index for existing nodes
                prop_index = defaultdict(list)
                for node_id, node_data in self.nodes.items():
                    if prop in node_data:
                        prop_index[node_data[prop]].append(node_id)

                self.property_indexes[prop] = dict(prop_index)

            # Store index configuration
            self.indexes[index_name] = config

            return True

        except Exception as e:
            logger.error(f"Failed to create memory index: {e}")
            return False

    def drop_index(self, index_name: str) -> bool:
        """Drop index."""
        if index_name in self.indexes:
            config = self.indexes[index_name]

            # Remove property index
            if len(config.properties) == 1:
                prop = config.properties[0]
                if prop in self.property_indexes:
                    del self.property_indexes[prop]

            del self.indexes[index_name]
            return True
        return False

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List memory indexes."""
        return [
            {
                "name": name,
                "entity_type": config.entity_type,
                "properties": config.properties,
                "index_type": config.index_type,
                "unique": config.unique,
                "backend": "memory",
            }
            for name, config in self.indexes.items()
        ]

    def bulk_load_nodes(
        self, nodes: List[Dict[str, Any]], config: BulkLoadConfig = None
    ) -> Dict[str, Any]:
        """Bulk load nodes into memory."""
        if config is None:
            config = BulkLoadConfig()

        results = {
            "total_nodes": len(nodes),
            "created_nodes": 0,
            "failed_nodes": 0,
            "errors": [],
        }

        for i, node in enumerate(nodes):
            try:
                labels = node.get("labels", ["Node"])
                properties = {k: v for k, v in node.items() if k != "labels"}

                node_id = self.create_node(labels, properties)
                results["created_nodes"] += 1

                # Progress callback
                if config.progress_callback and i % 100 == 0:
                    progress = (i + 1) / len(nodes)
                    config.progress_callback(progress, f"Processed {i + 1} nodes")

            except Exception as e:
                results["failed_nodes"] += 1
                results["errors"].append(f"Node {i}: {str(e)}")
                logger.error(f"Failed to create node {i}: {e}")

        return results

    def bulk_load_relationships(
        self, relationships: List[Dict[str, Any]], config: BulkLoadConfig = None
    ) -> Dict[str, Any]:
        """Bulk load relationships into memory."""
        if config is None:
            config = BulkLoadConfig()

        results = {
            "total_relationships": len(relationships),
            "created_relationships": 0,
            "failed_relationships": 0,
            "errors": [],
        }

        for i, rel in enumerate(relationships):
            try:
                source_id = rel.get("source_id")
                target_id = rel.get("target_id")
                relationship_type = rel.get("relationship_type", "RELATES_TO")
                properties = {
                    k: v
                    for k, v in rel.items()
                    if k not in ["source_id", "target_id", "relationship_type"]
                }

                rel_id = self.create_relationship(
                    source_id, target_id, relationship_type, properties
                )
                results["created_relationships"] += 1

                # Progress callback
                if config.progress_callback and i % 100 == 0:
                    progress = (i + 1) / len(relationships)
                    config.progress_callback(
                        progress, f"Processed {i + 1} relationships"
                    )

            except Exception as e:
                results["failed_relationships"] += 1
                results["errors"].append(f"Relationship {i}: {str(e)}")
                logger.error(f"Failed to create relationship {i}: {e}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory graph statistics."""
        try:
            # Label distribution
            label_distribution = {}
            for label, node_ids in self.label_index.items():
                label_distribution[label] = len(node_ids)

            # Relationship type distribution
            rel_type_distribution = defaultdict(int)
            for rel_data in self.relationships.values():
                rel_type_distribution[rel_data["relationship_type"]] += 1

            # Graph metrics
            total_degree = sum(len(edges) for edges in self.adjacency_list.values())
            avg_degree = total_degree / len(self.nodes) if self.nodes else 0

            # Combine statistics
            combined_stats = {
                **self.statistics,
                "database": {
                    "total_nodes": len(self.nodes),
                    "total_relationships": len(self.relationships),
                    "label_distribution": label_distribution,
                    "relationship_type_distribution": dict(rel_type_distribution),
                    "average_degree": avg_degree,
                },
                "memory": {
                    "nodes_in_memory": len(self.nodes),
                    "relationships_in_memory": len(self.relationships),
                    "indexes": len(self.property_indexes),
                },
                "connection": {"backend": "memory", "connected": self.is_connected()},
            }

            return combined_stats

        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {**self.statistics, "error": str(e)}

    def clear_database(self) -> bool:
        """Clear all memory data."""
        try:
            self.nodes.clear()
            self.relationships.clear()
            self.adjacency_list.clear()
            self.reverse_adjacency.clear()
            self.label_index.clear()
            self.property_indexes.clear()

            # Reset statistics
            self.statistics.update({"nodes_created": 0, "relationships_created": 0})

            return True

        except Exception as e:
            logger.error(f"Failed to clear memory database: {e}")
            return False

    def _update_property_indexes(self, node_id: str, properties: Dict[str, Any]):
        """Update property indexes for a node."""
        for prop, value in properties.items():
            if prop in self.property_indexes:
                # Remove from old value
                for old_value, node_list in self.property_indexes[prop].items():
                    if node_id in node_list:
                        node_list.remove(node_id)

                # Add to new value
                if value not in self.property_indexes[prop]:
                    self.property_indexes[prop][value] = []
                self.property_indexes[prop][value].append(node_id)

    def _remove_from_property_indexes(self, node_id: str):
        """Remove node from all property indexes."""
        for prop_index in self.property_indexes.values():
            for node_list in prop_index.values():
                if node_id in node_list:
                    node_list.remove(node_id)

    # Query implementations

    def _query_find_nodes(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find nodes query implementation."""
        labels = params.get("labels")
        properties = params.get("properties")
        limit = params.get("limit")

        return self.find_nodes(labels, properties, limit)

    def _query_find_paths(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find paths between nodes."""
        source_id = params.get("source")
        target_id = params.get("target")
        max_depth = params.get("max_depth", 3)

        if not source_id or not target_id:
            return []

        # Simple BFS path finding
        from collections import deque

        queue = deque([(source_id, [source_id])])
        visited = set()
        paths = []

        while queue and len(paths) < 100:  # Limit results
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if current_id == target_id and len(path) > 1:
                paths.append({"path": path, "length": len(path) - 1})
                continue

            if current_id in visited:
                continue
            visited.add(current_id)

            # Add neighbors
            for edge in self.adjacency_list.get(current_id, []):
                next_id = edge["target_id"]
                if next_id not in path:  # Avoid cycles
                    queue.append((next_id, path + [next_id]))

        return paths

    def _query_get_neighbors(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get neighbors of a node."""
        node_id = params.get("node_id")
        direction = params.get("direction", "out")  # "in", "out", "both"

        if not node_id:
            return []

        neighbors = []

        if direction in ["out", "both"]:
            for edge in self.adjacency_list.get(node_id, []):
                neighbors.append(
                    {
                        "node_id": edge["target_id"],
                        "relationship_id": edge["relationship_id"],
                        "relationship_type": edge["relationship_type"],
                        "direction": "outgoing",
                    }
                )

        if direction in ["in", "both"]:
            for edge in self.reverse_adjacency.get(node_id, []):
                neighbors.append(
                    {
                        "node_id": edge["source_id"],
                        "relationship_id": edge["relationship_id"],
                        "relationship_type": edge["relationship_type"],
                        "direction": "incoming",
                    }
                )

        return neighbors

    def _query_node_degrees(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get node degree statistics."""
        results = []

        for node_id in self.nodes:
            out_degree = len(self.adjacency_list.get(node_id, []))
            in_degree = len(self.reverse_adjacency.get(node_id, []))

            results.append(
                {
                    "node_id": node_id,
                    "out_degree": out_degree,
                    "in_degree": in_degree,
                    "total_degree": out_degree + in_degree,
                }
            )

        return results

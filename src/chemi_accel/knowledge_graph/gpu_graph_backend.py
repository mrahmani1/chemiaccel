"""
GPU-Optimized Graph Backend

Custom GPU-accelerated graph storage using RAPIDS/cuGraph for
high-performance graph operations on the IC-KG.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from .graph_database import (
    BulkLoadConfig,
    GraphDatabaseBackend,
    GraphDatabaseType,
    GraphQuery,
    GraphQueryResult,
    IndexConfig,
)

logger = logging.getLogger(__name__)


class GPUGraphBackend(GraphDatabaseBackend):
    """GPU-optimized graph database backend using RAPIDS/cuGraph."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.graph = None
        self.node_data = None
        self.edge_data = None
        self.node_index = {}
        self.relationship_index = {}
        self._setup_gpu_config()

    def _get_backend_type(self) -> GraphDatabaseType:
        return GraphDatabaseType.GPU_GRAPH

    def _setup_gpu_config(self):
        """Setup GPU computation configuration."""
        self.gpu_memory_limit = self.config.get("gpu_memory_limit", "8GB")
        self.use_managed_memory = self.config.get("use_managed_memory", True)
        self.graph_algorithms = self.config.get(
            "algorithms",
            ["pagerank", "betweenness_centrality", "clustering", "shortest_path"],
        )
        self.batch_processing = self.config.get("batch_processing", True)
        self.compression = self.config.get("compression", True)

    def connect(self) -> bool:
        """Initialize GPU graph storage."""
        try:
            # Check for RAPIDS availability
            import cudf
            import cugraph
            import cupy as cp

            # Initialize GPU memory pool if requested
            if self.use_managed_memory:
                import rmm

                rmm.reinitialize(
                    managed_memory=True,
                    pool_allocator=True,
                    initial_pool_size=self.gpu_memory_limit,
                )

            # Initialize empty graph structures
            self.graph = cugraph.Graph(directed=True)
            self.node_data = cudf.DataFrame()
            self.edge_data = cudf.DataFrame()

            # Initialize indexes
            self.node_index = {}
            self.relationship_index = {}

            logger.info("Connected to GPU graph backend with RAPIDS/cuGraph")
            return True

        except ImportError as e:
            logger.error(f"RAPIDS libraries not available: {e}")
            logger.error(
                "Install with: conda install -c rapidsai -c nvidia -c conda-forge rapids"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize GPU graph backend: {e}")
            return False

    def disconnect(self) -> bool:
        """Cleanup GPU resources."""
        try:
            if self.graph is not None:
                del self.graph
                self.graph = None

            if self.node_data is not None:
                del self.node_data
                self.node_data = None

            if self.edge_data is not None:
                del self.edge_data
                self.edge_data = None

            self.node_index.clear()
            self.relationship_index.clear()

            # Cleanup GPU memory
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()

            logger.info("Disconnected from GPU graph backend")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from GPU backend: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if GPU backend is initialized."""
        return self.graph is not None

    def execute_query(self, query: GraphQuery) -> GraphQueryResult:
        """Execute a graph algorithm or query."""
        if not self.is_connected():
            raise ConnectionError("GPU graph backend not connected")

        start_time = time.time()

        try:
            query_type = query.query_type.lower()
            records = []

            if query_type == "pagerank":
                records = self._execute_pagerank(query.parameters)
            elif query_type == "betweenness_centrality":
                records = self._execute_betweenness_centrality(query.parameters)
            elif query_type == "clustering":
                records = self._execute_clustering(query.parameters)
            elif query_type == "shortest_path":
                records = self._execute_shortest_path(query.parameters)
            elif query_type == "community_detection":
                records = self._execute_community_detection(query.parameters)
            elif query_type == "similarity":
                records = self._execute_similarity(query.parameters)
            elif query_type == "pattern_match":
                records = self._execute_pattern_match(query.parameters)
            elif query_type == "subgraph":
                records = self._execute_subgraph_extraction(query.parameters)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")

            execution_time = time.time() - start_time

            summary = {
                "query_type": query_type,
                "result_count": len(records),
                "execution_time": execution_time,
                "gpu_memory_used": self._get_gpu_memory_usage(),
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
            logger.error(f"GPU query execution failed: {e}")

            return GraphQueryResult(
                records=[],
                summary={"error": str(e)},
                execution_time=execution_time,
                query=query,
                metadata={"error": True},
            )

    def create_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
        """Create a node in the GPU graph."""
        import cudf

        node_id = str(uuid.uuid4())
        properties["_id"] = node_id
        properties["_labels"] = ",".join(labels)
        properties["_created_at"] = datetime.now().isoformat()

        # Add to node data
        new_node = cudf.DataFrame([properties])
        if len(self.node_data) == 0:
            self.node_data = new_node
        else:
            self.node_data = cudf.concat([self.node_data, new_node], ignore_index=True)

        # Update node index
        node_idx = len(self.node_data) - 1
        self.node_index[node_id] = node_idx

        # Update labels index
        for label in labels:
            if label not in self.node_index:
                self.node_index[f"label:{label}"] = []
            self.node_index[f"label:{label}"].append(node_idx)

        self.statistics["nodes_created"] += 1
        return node_id

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
    ) -> str:
        """Create a relationship in the GPU graph."""
        import cudf

        if source_id not in self.node_index or target_id not in self.node_index:
            raise ValueError("Source or target node not found")

        if properties is None:
            properties = {}

        rel_id = str(uuid.uuid4())
        properties["_id"] = rel_id
        properties["_type"] = relationship_type
        properties["_created_at"] = datetime.now().isoformat()

        # Get node indices
        source_idx = self.node_index[source_id]
        target_idx = self.node_index[target_id]

        # Create edge data
        edge_record = {
            "src": source_idx,
            "dst": target_idx,
            "source_id": source_id,
            "target_id": target_id,
            **properties,
        }

        # Add to edge data
        new_edge = cudf.DataFrame([edge_record])
        if len(self.edge_data) == 0:
            self.edge_data = new_edge
        else:
            self.edge_data = cudf.concat([self.edge_data, new_edge], ignore_index=True)

        # Update graph structure
        self._rebuild_graph()

        # Update relationship index
        edge_idx = len(self.edge_data) - 1
        self.relationship_index[rel_id] = edge_idx

        self.statistics["relationships_created"] += 1
        return rel_id

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID."""
        if node_id not in self.node_index:
            return None

        node_idx = self.node_index[node_id]
        node_row = self.node_data.iloc[node_idx]

        # Convert to dictionary
        node_data = node_row.to_dict()

        # Parse labels
        if "_labels" in node_data:
            node_data["labels"] = node_data["_labels"].split(",")
            del node_data["_labels"]

        return node_data

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        if node_id not in self.node_index:
            return False

        node_idx = self.node_index[node_id]
        properties["_updated_at"] = datetime.now().isoformat()

        # Update properties in GPU DataFrame
        for key, value in properties.items():
            if key not in self.node_data.columns:
                # Add new column
                self.node_data[key] = None
            self.node_data.at[node_idx, key] = value

        return True

    def delete_node(self, node_id: str) -> bool:
        """Delete node by ID."""
        if node_id not in self.node_index:
            return False

        node_idx = self.node_index[node_id]

        # Remove from node data
        mask = self.node_data["_id"] != node_id
        self.node_data = self.node_data[mask]

        # Remove associated edges
        edge_mask = (self.edge_data["source_id"] != node_id) & (
            self.edge_data["target_id"] != node_id
        )
        self.edge_data = self.edge_data[edge_mask]

        # Rebuild indexes
        self._rebuild_indexes()
        self._rebuild_graph()

        return True

    def find_nodes(
        self,
        labels: List[str] = None,
        properties: Dict[str, Any] = None,
        limit: int = None,
    ) -> List[Dict[str, Any]]:
        """Find nodes matching criteria using GPU acceleration."""
        import cudf

        # Start with all nodes
        mask = cudf.Series([True] * len(self.node_data))

        # Apply label filters
        if labels:
            label_mask = cudf.Series([False] * len(self.node_data))
            for label in labels:
                label_condition = self.node_data["_labels"].str.contains(label)
                label_mask = label_mask | label_condition
            mask = mask & label_mask

        # Apply property filters
        if properties:
            for key, value in properties.items():
                if key in self.node_data.columns:
                    mask = mask & (self.node_data[key] == value)

        # Filter and limit results
        filtered_data = self.node_data[mask]
        if limit:
            filtered_data = filtered_data.head(limit)

        # Convert to list of dictionaries
        nodes = []
        for _, row in filtered_data.iterrows():
            node_data = row.to_dict()
            if "_labels" in node_data:
                node_data["labels"] = node_data["_labels"].split(",")
                del node_data["_labels"]
            nodes.append(node_data)

        return nodes

    def create_index(self, config: IndexConfig) -> bool:
        """Create GPU-accelerated index."""
        try:
            # GPU DataFrames have built-in indexing
            # Store index configuration for optimization hints
            index_name = f"{config.entity_type}_{'_'.join(config.properties)}_{config.index_type}"
            self.indexes[index_name] = config

            # For GPU optimization, we can pre-sort by indexed columns
            if config.properties and len(config.properties) == 1:
                prop = config.properties[0]
                if prop in self.node_data.columns:
                    self.node_data = self.node_data.sort_values(by=prop)
                    self._rebuild_indexes()

            return True

        except Exception as e:
            logger.error(f"Failed to create GPU index: {e}")
            return False

    def drop_index(self, index_name: str) -> bool:
        """Drop index configuration."""
        if index_name in self.indexes:
            del self.indexes[index_name]
            return True
        return False

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List GPU index configurations."""
        return [
            {
                "name": name,
                "entity_type": config.entity_type,
                "properties": config.properties,
                "index_type": config.index_type,
                "unique": config.unique,
                "backend": "gpu",
            }
            for name, config in self.indexes.items()
        ]

    def bulk_load_nodes(
        self, nodes: List[Dict[str, Any]], config: BulkLoadConfig = None
    ) -> Dict[str, Any]:
        """GPU-accelerated bulk node loading."""
        import cudf

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

        try:
            # Prepare batch data
            batch_data = []
            for node in nodes:
                node_data = node.copy()
                if "_id" not in node_data:
                    node_data["_id"] = str(uuid.uuid4())

                labels = node_data.pop("labels", ["Node"])
                node_data["_labels"] = ",".join(labels)
                node_data["_created_at"] = datetime.now().isoformat()

                batch_data.append(node_data)

            # Create GPU DataFrame from batch
            new_nodes_df = cudf.DataFrame(batch_data)

            # Append to existing node data
            if len(self.node_data) == 0:
                self.node_data = new_nodes_df
            else:
                self.node_data = cudf.concat(
                    [self.node_data, new_nodes_df], ignore_index=True
                )

            # Rebuild indexes
            self._rebuild_indexes()

            results["created_nodes"] = len(batch_data)
            self.statistics["nodes_created"] += len(batch_data)

            # Progress callback
            if config.progress_callback:
                config.progress_callback(1.0, f"Loaded {len(batch_data)} nodes")

        except Exception as e:
            results["failed_nodes"] = len(nodes)
            results["errors"].append(str(e))
            logger.error(f"GPU bulk node load failed: {e}")

        return results

    def bulk_load_relationships(
        self, relationships: List[Dict[str, Any]], config: BulkLoadConfig = None
    ) -> Dict[str, Any]:
        """GPU-accelerated bulk relationship loading."""
        import cudf

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

        try:
            # Prepare batch data
            batch_data = []
            for rel in relationships:
                rel_data = rel.copy()
                if "_id" not in rel_data:
                    rel_data["_id"] = str(uuid.uuid4())

                source_id = rel_data.get("source_id")
                target_id = rel_data.get("target_id")

                if source_id not in self.node_index or target_id not in self.node_index:
                    results["failed_relationships"] += 1
                    continue

                rel_data["src"] = self.node_index[source_id]
                rel_data["dst"] = self.node_index[target_id]
                rel_data["_type"] = rel_data.pop("relationship_type", "RELATES_TO")
                rel_data["_created_at"] = datetime.now().isoformat()

                batch_data.append(rel_data)

            # Create GPU DataFrame from batch
            new_edges_df = cudf.DataFrame(batch_data)

            # Append to existing edge data
            if len(self.edge_data) == 0:
                self.edge_data = new_edges_df
            else:
                self.edge_data = cudf.concat(
                    [self.edge_data, new_edges_df], ignore_index=True
                )

            # Rebuild graph and indexes
            self._rebuild_graph()
            self._rebuild_relationship_index()

            results["created_relationships"] = len(batch_data)
            self.statistics["relationships_created"] += len(batch_data)

            # Progress callback
            if config.progress_callback:
                config.progress_callback(1.0, f"Loaded {len(batch_data)} relationships")

        except Exception as e:
            results["failed_relationships"] = len(relationships)
            results["errors"].append(str(e))
            logger.error(f"GPU bulk relationship load failed: {e}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get GPU graph statistics."""
        try:
            # Basic counts
            node_count = len(self.node_data) if self.node_data is not None else 0
            edge_count = len(self.edge_data) if self.edge_data is not None else 0

            # GPU memory usage
            gpu_memory = self._get_gpu_memory_usage()

            # Graph metrics (if graph is built)
            graph_metrics = {}
            if self.graph is not None and edge_count > 0:
                try:
                    import cugraph

                    # Calculate graph metrics using GPU
                    if (
                        node_count < 10000
                    ):  # Only for smaller graphs to avoid memory issues
                        try:
                            pagerank_df = cugraph.pagerank(self.graph)
                            graph_metrics["avg_pagerank"] = float(
                                pagerank_df["pagerank"].mean()
                            )
                        except:
                            pass

                        try:
                            clustering_df = cugraph.triangle_count(self.graph)
                            graph_metrics["avg_clustering"] = float(
                                clustering_df["counts"].mean()
                            )
                        except:
                            pass

                except Exception as e:
                    logger.debug(f"Could not calculate graph metrics: {e}")

            # Label distribution
            label_distribution = {}
            if self.node_data is not None and "_labels" in self.node_data.columns:
                # Extract labels and count
                all_labels = self.node_data["_labels"].str.split(",").explode()
                label_counts = all_labels.value_counts()
                label_distribution = label_counts.to_dict()

            # Combine statistics
            combined_stats = {
                **self.statistics,
                "database": {
                    "total_nodes": node_count,
                    "total_relationships": edge_count,
                    "label_distribution": label_distribution,
                    "graph_metrics": graph_metrics,
                },
                "gpu": {
                    "memory_usage": gpu_memory,
                    "algorithms_available": self.graph_algorithms,
                },
                "connection": {
                    "backend": "gpu_graph",
                    "connected": self.is_connected(),
                },
            }

            return combined_stats

        except Exception as e:
            logger.error(f"Failed to get GPU statistics: {e}")
            return {**self.statistics, "error": str(e)}

    def clear_database(self) -> bool:
        """Clear all GPU graph data."""
        try:
            import cudf

            # Clear DataFrames
            self.node_data = cudf.DataFrame()
            self.edge_data = cudf.DataFrame()

            # Clear indexes
            self.node_index.clear()
            self.relationship_index.clear()

            # Clear graph
            if self.graph is not None:
                import cugraph

                self.graph = cugraph.Graph(directed=True)

            # Reset statistics
            self.statistics.update({"nodes_created": 0, "relationships_created": 0})

            # Free GPU memory
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()

            return True

        except Exception as e:
            logger.error(f"Failed to clear GPU database: {e}")
            return False

    def _rebuild_graph(self):
        """Rebuild the cuGraph from edge data."""
        try:
            import cugraph

            if len(self.edge_data) > 0:
                # Create graph from edge list
                edge_list = self.edge_data[["src", "dst"]].copy()
                self.graph = cugraph.from_cudf_edgelist(
                    edge_list,
                    source="src",
                    destination="dst",
                    create_using=cugraph.Graph(directed=True),
                )
            else:
                self.graph = cugraph.Graph(directed=True)

        except Exception as e:
            logger.error(f"Failed to rebuild graph: {e}")

    def _rebuild_indexes(self):
        """Rebuild node indexes."""
        self.node_index.clear()

        if self.node_data is not None and len(self.node_data) > 0:
            for idx, row in self.node_data.iterrows():
                node_id = row["_id"]
                self.node_index[node_id] = idx

                # Index by labels
                if "_labels" in row:
                    labels = row["_labels"].split(",")
                    for label in labels:
                        if f"label:{label}" not in self.node_index:
                            self.node_index[f"label:{label}"] = []
                        self.node_index[f"label:{label}"].append(idx)

    def _rebuild_relationship_index(self):
        """Rebuild relationship indexes."""
        self.relationship_index.clear()

        if self.edge_data is not None and len(self.edge_data) > 0:
            for idx, row in self.edge_data.iterrows():
                rel_id = row["_id"]
                self.relationship_index[rel_id] = idx

    def _get_gpu_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage statistics."""
        try:
            import cupy as cp

            mempool = cp.get_default_memory_pool()

            return {
                "used_bytes": mempool.used_bytes(),
                "total_bytes": mempool.total_bytes(),
                "free_bytes": mempool.free_bytes(),
            }
        except Exception:
            return {"error": "Could not get GPU memory usage"}

    # GPU-specific query implementations

    def _execute_pagerank(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute PageRank algorithm on GPU."""
        import cugraph

        alpha = params.get("alpha", 0.85)
        max_iter = params.get("max_iter", 100)
        tol = params.get("tol", 1e-5)

        pagerank_df = cugraph.pagerank(
            self.graph, alpha=alpha, max_iter=max_iter, tol=tol
        )

        # Convert to records
        records = []
        for _, row in pagerank_df.iterrows():
            records.append(
                {"node": int(row["vertex"]), "pagerank": float(row["pagerank"])}
            )

        return records

    def _execute_betweenness_centrality(
        self, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute betweenness centrality on GPU."""
        import cugraph

        normalized = params.get("normalized", True)
        k = params.get("k", None)  # For approximate computation

        if k:
            bc_df = cugraph.betweenness_centrality(
                self.graph, normalized=normalized, k=k
            )
        else:
            bc_df = cugraph.betweenness_centrality(self.graph, normalized=normalized)

        # Convert to records
        records = []
        for _, row in bc_df.iterrows():
            records.append(
                {
                    "node": int(row["vertex"]),
                    "betweenness_centrality": float(row["betweenness_centrality"]),
                }
            )

        return records

    def _execute_clustering(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute clustering coefficient computation on GPU."""
        import cugraph

        clustering_df = cugraph.clustering(self.graph)

        # Convert to records
        records = []
        for _, row in clustering_df.iterrows():
            records.append(
                {
                    "node": int(row["vertex"]),
                    "clustering_coefficient": float(row["clustering_coefficient"]),
                }
            )

        return records

    def _execute_shortest_path(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute shortest path computation on GPU."""
        import cugraph

        source = params.get("source")
        target = params.get("target")

        if source is None:
            raise ValueError("Source node required for shortest path")

        if target is not None:
            # Single-source single-target
            path_df = cugraph.shortest_path(self.graph, source, target)
        else:
            # Single-source all targets
            path_df = cugraph.sssp(self.graph, source)

        # Convert to records
        records = []
        for _, row in path_df.iterrows():
            records.append(
                {
                    "vertex": int(row["vertex"]),
                    "distance": float(row["distance"]),
                    "predecessor": int(row["predecessor"])
                    if "predecessor" in row
                    else None,
                }
            )

        return records

    def _execute_community_detection(
        self, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute community detection on GPU."""
        import cugraph

        algorithm = params.get("algorithm", "louvain")

        if algorithm == "louvain":
            community_df, modularity = cugraph.louvain(self.graph)
        elif algorithm == "leiden":
            community_df = cugraph.leiden(self.graph)
            modularity = None
        else:
            raise ValueError(f"Unsupported community detection algorithm: {algorithm}")

        # Convert to records
        records = []
        for _, row in community_df.iterrows():
            record = {"node": int(row["vertex"]), "community": int(row["partition"])}
            if modularity is not None:
                record["modularity"] = float(modularity)
            records.append(record)

        return records

    def _execute_similarity(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute similarity computation on GPU."""
        import cugraph

        similarity_type = params.get("type", "jaccard")

        if similarity_type == "jaccard":
            similarity_df = cugraph.jaccard(self.graph)
        elif similarity_type == "sorensen":
            similarity_df = cugraph.sorensen(self.graph)
        elif similarity_type == "overlap":
            similarity_df = cugraph.overlap(self.graph)
        else:
            raise ValueError(f"Unsupported similarity type: {similarity_type}")

        # Convert to records
        records = []
        for _, row in similarity_df.iterrows():
            records.append(
                {
                    "source": int(row["source"]),
                    "destination": int(row["destination"]),
                    "similarity": float(row[similarity_type]),
                }
            )

        return records

    def _execute_pattern_match(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute pattern matching on GPU."""
        # Simplified pattern matching - could be extended
        pattern = params.get("pattern", {})
        node_filters = pattern.get("nodes", {})
        edge_filters = pattern.get("edges", {})

        # Find matching nodes
        matching_nodes = []
        if node_filters:
            for node_id, node_idx in self.node_index.items():
                if node_id.startswith("label:"):
                    continue

                node_data = self.get_node(node_id)
                if self._matches_filters(node_data, node_filters):
                    matching_nodes.append({"node_id": node_id, "node_idx": node_idx})

        return matching_nodes

    def _execute_subgraph_extraction(
        self, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract subgraph on GPU."""
        import cugraph

        vertices = params.get("vertices", [])

        if not vertices:
            return []

        # Extract subgraph
        subgraph = cugraph.subgraph(self.graph, vertices)

        # Get subgraph statistics
        edge_count = len(subgraph.edges())
        vertex_count = len(vertices)

        return [
            {
                "vertices": vertices,
                "vertex_count": vertex_count,
                "edge_count": edge_count,
                "density": edge_count / (vertex_count * (vertex_count - 1))
                if vertex_count > 1
                else 0,
            }
        ]

    def _matches_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if data matches filter criteria."""
        for key, value in filters.items():
            if key not in data:
                return False
            if data[key] != value:
                return False
        return True

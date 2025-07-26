"""
Graph Query Engine and Optimization

Advanced query engine with optimization, caching, and specialized
chemical query support for the IC-KG.
"""

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

from .graph_database import GraphDatabaseBackend, GraphQuery, GraphQueryResult
from .ontology import EntityType, PropertyType, RelationshipType

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """Query execution plan with optimization details."""

    query_id: str
    original_query: str
    optimized_query: str
    execution_steps: List[Dict[str, Any]]
    estimated_cost: float
    cache_strategy: str = "none"
    parallel_execution: bool = False


@dataclass
class QueryStats:
    """Query execution statistics."""

    query_id: str
    execution_time: float
    result_count: int
    cache_hit: bool = False
    optimization_applied: bool = False
    backend_used: str = ""
    memory_usage: Optional[int] = None


class QueryCache:
    """LRU cache for query results with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self._lock = threading.RLock()

    def get(self, query_hash: str) -> Optional[GraphQueryResult]:
        """Get cached query result."""
        with self._lock:
            if query_hash not in self.cache:
                return None

            # Check expiry
            if query_hash in self.expiry_times:
                if datetime.now() > self.expiry_times[query_hash]:
                    self._remove(query_hash)
                    return None

            # Update access time
            self.access_times[query_hash] = datetime.now()
            return self.cache[query_hash]

    def put(self, query_hash: str, result: GraphQueryResult, ttl: int = None):
        """Store query result in cache."""
        with self._lock:
            # Evict if necessary
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            self.cache[query_hash] = result
            self.access_times[query_hash] = datetime.now()

            if ttl is None:
                ttl = self.default_ttl

            if ttl > 0:
                self.expiry_times[query_hash] = datetime.now() + timedelta(seconds=ttl)

    def invalidate(self, pattern: str = None):
        """Invalidate cache entries matching pattern."""
        with self._lock:
            if pattern is None:
                # Clear all
                self.cache.clear()
                self.access_times.clear()
                self.expiry_times.clear()
            else:
                # Clear matching patterns
                keys_to_remove = [k for k in self.cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self._remove(key)

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(lru_key)

    def _remove(self, key: str):
        """Remove key from all cache structures."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expiry_times.pop(key, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": getattr(self, "_hit_count", 0)
                / max(getattr(self, "_total_requests", 1), 1),
                "expired_entries": sum(
                    1
                    for exp_time in self.expiry_times.values()
                    if datetime.now() > exp_time
                ),
            }


class QueryOptimizer:
    """Query optimizer with rule-based and cost-based optimization."""

    def __init__(self):
        self.optimization_rules = [
            self._optimize_filter_pushdown,
            self._optimize_join_order,
            self._optimize_index_usage,
            self._optimize_limit_early,
            self._optimize_redundant_operations,
        ]
        self.statistics = defaultdict(int)

    def optimize_query(self, query: GraphQuery, backend_type: str) -> QueryPlan:
        """Optimize query and create execution plan."""
        query_id = self._generate_query_id(query)

        optimized_query = query.query
        execution_steps = []
        estimated_cost = self._estimate_query_cost(query)
        optimization_applied = False

        # Apply optimization rules
        for rule in self.optimization_rules:
            try:
                rule_result = rule(query, backend_type)
                if rule_result:
                    optimized_query = rule_result.get("query", optimized_query)
                    execution_steps.extend(rule_result.get("steps", []))
                    optimization_applied = True
                    self.statistics[rule.__name__] += 1

            except Exception as e:
                logger.warning(f"Optimization rule {rule.__name__} failed: {e}")

        # Determine caching strategy
        cache_strategy = self._determine_cache_strategy(query, estimated_cost)

        # Determine parallel execution potential
        parallel_execution = self._can_parallelize(query, backend_type)

        plan = QueryPlan(
            query_id=query_id,
            original_query=query.query,
            optimized_query=optimized_query,
            execution_steps=execution_steps,
            estimated_cost=estimated_cost,
            cache_strategy=cache_strategy,
            parallel_execution=parallel_execution,
        )

        return plan

    def _generate_query_id(self, query: GraphQuery) -> str:
        """Generate unique query identifier."""
        query_string = f"{query.query}:{json.dumps(query.parameters, sort_keys=True)}"
        return hashlib.md5(query_string.encode()).hexdigest()[:16]

    def _estimate_query_cost(self, query: GraphQuery) -> float:
        """Estimate query execution cost."""
        # Simple cost estimation based on query complexity
        cost = 1.0

        query_lower = query.query.lower()

        # Add cost for operations
        if "join" in query_lower or "match" in query_lower:
            cost += 2.0
        if "order by" in query_lower or "sort" in query_lower:
            cost += 1.5
        if "group by" in query_lower or "aggregate" in query_lower:
            cost += 1.5

        # Add cost for result size hints
        if "limit" in query_lower:
            try:
                limit_val = int(
                    [word for word in query.query.split() if word.isdigit()][0]
                )
                cost *= min(limit_val / 1000.0, 1.0)
            except:
                pass

        return cost

    def _determine_cache_strategy(self, query: GraphQuery, cost: float) -> str:
        """Determine appropriate caching strategy."""
        if cost > 5.0:
            return "aggressive"
        elif cost > 2.0:
            return "normal"
        else:
            return "minimal"

    def _can_parallelize(self, query: GraphQuery, backend_type: str) -> bool:
        """Determine if query can be parallelized."""
        # GPU backends are naturally parallel
        if backend_type == "gpu_graph":
            return True

        # Large result sets can benefit from parallelization
        query_lower = query.query.lower()
        if "limit" not in query_lower and (
            "match" in query_lower or "find" in query_lower
        ):
            return True

        return False

    # Optimization rules

    def _optimize_filter_pushdown(
        self, query: GraphQuery, backend_type: str
    ) -> Optional[Dict[str, Any]]:
        """Push filters down to reduce intermediate result size."""
        # Implementation would depend on query language
        # This is a placeholder for actual optimization logic
        return None

    def _optimize_join_order(
        self, query: GraphQuery, backend_type: str
    ) -> Optional[Dict[str, Any]]:
        """Optimize join order based on selectivity."""
        return None

    def _optimize_index_usage(
        self, query: GraphQuery, backend_type: str
    ) -> Optional[Dict[str, Any]]:
        """Suggest index usage for better performance."""
        return None

    def _optimize_limit_early(
        self, query: GraphQuery, backend_type: str
    ) -> Optional[Dict[str, Any]]:
        """Apply limits as early as possible."""
        return None

    def _optimize_redundant_operations(
        self, query: GraphQuery, backend_type: str
    ) -> Optional[Dict[str, Any]]:
        """Remove redundant operations."""
        return None


class GraphQueryEngine:
    """Advanced graph query engine with optimization and caching."""

    def __init__(self, backend: GraphDatabaseBackend, enable_caching: bool = True):
        self.backend = backend
        self.enable_caching = enable_caching
        self.cache = QueryCache() if enable_caching else None
        self.optimizer = QueryOptimizer()
        self.query_stats = []
        self._execution_stats = defaultdict(int)

    def execute_query(
        self, query: GraphQuery, use_cache: bool = True, optimize: bool = True
    ) -> GraphQueryResult:
        """Execute query with optimization and caching."""
        start_time = time.time()

        # Generate query hash for caching
        query_hash = self._hash_query(query)

        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get(query_hash)
            if cached_result:
                self._record_query_stats(
                    query_hash,
                    time.time() - start_time,
                    len(cached_result.records),
                    cache_hit=True,
                )
                return cached_result

        # Optimize query if requested
        plan = None
        if optimize:
            plan = self.optimizer.optimize_query(query, self.backend.backend_type.value)
            if plan.optimized_query != query.query:
                query = GraphQuery(
                    query=plan.optimized_query,
                    parameters=query.parameters,
                    query_type=query.query_type,
                    timeout=query.timeout,
                )

        # Execute query
        try:
            result = self.backend.execute_query(query)

            # Cache result if appropriate
            if use_cache and self.cache and plan:
                cache_ttl = self._get_cache_ttl(plan.cache_strategy)
                if cache_ttl > 0:
                    self.cache.put(query_hash, result, cache_ttl)

            # Record statistics
            self._record_query_stats(
                query_hash,
                time.time() - start_time,
                len(result.records),
                optimization_applied=plan is not None,
            )

            return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            # Return error result
            return GraphQueryResult(
                records=[],
                summary={"error": str(e)},
                execution_time=time.time() - start_time,
                query=query,
                metadata={"error": True},
            )

    def execute_chemical_query(
        self, chemical_query: "ChemicalQuery"
    ) -> GraphQueryResult:
        """Execute specialized chemical query."""
        # Convert chemical query to backend-specific query
        graph_query = chemical_query.to_graph_query(self.backend.backend_type)

        # Execute with chemical-specific optimizations
        return self.execute_query(graph_query, optimize=True)

    def execute_batch_queries(
        self, queries: List[GraphQuery], parallel: bool = True
    ) -> List[GraphQueryResult]:
        """Execute multiple queries in batch."""
        if parallel and len(queries) > 1:
            return self._execute_parallel_queries(queries)
        else:
            return [self.execute_query(query) for query in queries]

    def get_query_statistics(self) -> Dict[str, Any]:
        """Get comprehensive query execution statistics."""
        if not self.query_stats:
            return {"message": "No queries executed yet"}

        # Calculate aggregate statistics
        total_queries = len(self.query_stats)
        avg_execution_time = (
            sum(stat.execution_time for stat in self.query_stats) / total_queries
        )
        cache_hit_rate = (
            sum(1 for stat in self.query_stats if stat.cache_hit) / total_queries
        )

        return {
            "total_queries": total_queries,
            "average_execution_time": avg_execution_time,
            "cache_hit_rate": cache_hit_rate,
            "optimization_stats": dict(self.optimizer.statistics),
            "execution_stats": dict(self._execution_stats),
            "cache_stats": self.cache.get_stats() if self.cache else None,
            "recent_queries": [
                {
                    "query_id": stat.query_id,
                    "execution_time": stat.execution_time,
                    "result_count": stat.result_count,
                    "cache_hit": stat.cache_hit,
                }
                for stat in self.query_stats[-10:]  # Last 10 queries
            ],
        }

    def clear_cache(self, pattern: str = None):
        """Clear query cache."""
        if self.cache:
            self.cache.invalidate(pattern)

    def _hash_query(self, query: GraphQuery) -> str:
        """Generate hash for query caching."""
        query_string = f"{query.query}:{json.dumps(query.parameters, sort_keys=True)}:{query.query_type}"
        return hashlib.md5(query_string.encode()).hexdigest()

    def _get_cache_ttl(self, cache_strategy: str) -> int:
        """Get cache TTL based on strategy."""
        ttl_map = {
            "aggressive": 7200,  # 2 hours
            "normal": 3600,  # 1 hour
            "minimal": 600,  # 10 minutes
            "none": 0,
        }
        return ttl_map.get(cache_strategy, 3600)

    def _record_query_stats(
        self,
        query_id: str,
        execution_time: float,
        result_count: int,
        cache_hit: bool = False,
        optimization_applied: bool = False,
    ):
        """Record query execution statistics."""
        stats = QueryStats(
            query_id=query_id,
            execution_time=execution_time,
            result_count=result_count,
            cache_hit=cache_hit,
            optimization_applied=optimization_applied,
            backend_used=self.backend.backend_type.value,
        )

        self.query_stats.append(stats)

        # Keep only recent stats to avoid memory bloat
        if len(self.query_stats) > 1000:
            self.query_stats = self.query_stats[-500:]

        # Update execution statistics
        self._execution_stats["total_queries"] += 1
        if cache_hit:
            self._execution_stats["cache_hits"] += 1
        if optimization_applied:
            self._execution_stats["optimized_queries"] += 1

    def _execute_parallel_queries(
        self, queries: List[GraphQuery]
    ) -> List[GraphQueryResult]:
        """Execute queries in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
            futures = [executor.submit(self.execute_query, query) for query in queries]
            results = [future.result() for future in futures]

        return results


class QueryBuilder:
    """Builder for constructing graph queries programmatically."""

    def __init__(self, backend_type: str = "memory"):
        self.backend_type = backend_type
        self.reset()

    def reset(self):
        """Reset builder to initial state."""
        self._query_parts = []
        self._parameters = {}
        self._query_type = "custom"
        self._limit = None
        self._order_by = []
        return self

    def find_nodes(self, labels: List[str] = None, properties: Dict[str, Any] = None):
        """Add node finding clause."""
        if self.backend_type == "neo4j":
            self._add_cypher_node_match(labels, properties)
        elif self.backend_type == "jena":
            self._add_sparql_node_match(labels, properties)
        else:
            self._query_type = "find_nodes"
            self._parameters.update({"labels": labels, "properties": properties})
        return self

    def find_relationships(
        self,
        source_labels: List[str] = None,
        target_labels: List[str] = None,
        relationship_types: List[str] = None,
    ):
        """Add relationship finding clause."""
        if self.backend_type == "neo4j":
            self._add_cypher_relationship_match(
                source_labels, target_labels, relationship_types
            )
        elif self.backend_type == "jena":
            self._add_sparql_relationship_match(
                source_labels, target_labels, relationship_types
            )
        else:
            self._query_type = "find_relationships"
            self._parameters.update(
                {
                    "source_labels": source_labels,
                    "target_labels": target_labels,
                    "relationship_types": relationship_types,
                }
            )
        return self

    def where(self, conditions: Dict[str, Any]):
        """Add WHERE conditions."""
        if self.backend_type == "neo4j":
            self._add_cypher_where(conditions)
        elif self.backend_type == "jena":
            self._add_sparql_filter(conditions)
        else:
            self._parameters.update(conditions)
        return self

    def limit(self, count: int):
        """Add LIMIT clause."""
        self._limit = count
        if self.backend_type in ["neo4j", "memory"]:
            self._query_parts.append(f"LIMIT {count}")
        elif self.backend_type == "jena":
            self._query_parts.append(f"LIMIT {count}")
        return self

    def order_by(self, property_name: str, ascending: bool = True):
        """Add ORDER BY clause."""
        direction = "ASC" if ascending else "DESC"
        self._order_by.append((property_name, direction))

        if self.backend_type == "neo4j":
            self._query_parts.append(f"ORDER BY n.{property_name} {direction}")
        elif self.backend_type == "jena":
            self._query_parts.append(f"ORDER BY {direction}(?{property_name})")
        return self

    def build(self) -> GraphQuery:
        """Build the final query."""
        if self.backend_type == "neo4j":
            query = " ".join(self._query_parts)
            query_type = "cypher"
        elif self.backend_type == "jena":
            query = " ".join(self._query_parts)
            query_type = "sparql"
        else:
            query = json.dumps(self._parameters)
            query_type = self._query_type

        return GraphQuery(
            query=query, parameters=self._parameters.copy(), query_type=query_type
        )

    def _add_cypher_node_match(self, labels: List[str], properties: Dict[str, Any]):
        """Add Cypher node match clause."""
        label_str = ":" + ":".join(labels) if labels else ""
        self._query_parts.append(f"MATCH (n{label_str})")

        if properties:
            conditions = []
            for key, value in properties.items():
                param_name = f"prop_{key}"
                conditions.append(f"n.{key} = ${param_name}")
                self._parameters[param_name] = value

            if conditions:
                self._query_parts.append("WHERE " + " AND ".join(conditions))

        self._query_parts.append("RETURN n")

    def _add_sparql_node_match(self, labels: List[str], properties: Dict[str, Any]):
        """Add SPARQL node match clause."""
        self._query_parts.append("SELECT ?node WHERE {")

        if labels:
            for label in labels:
                self._query_parts.append(f"  ?node rdf:type cao:{label} .")

        if properties:
            for key, value in properties.items():
                self._query_parts.append(f'  ?node cao:{key} "{value}" .')

        self._query_parts.append("}")

    def _add_cypher_relationship_match(
        self,
        source_labels: List[str],
        target_labels: List[str],
        relationship_types: List[str],
    ):
        """Add Cypher relationship match clause."""
        source_label = ":" + ":".join(source_labels) if source_labels else ""
        target_label = ":" + ":".join(target_labels) if target_labels else ""
        rel_type = ":" + "|".join(relationship_types) if relationship_types else ""

        self._query_parts.append(
            f"MATCH (a{source_label})-[r{rel_type}]->(b{target_label})"
        )
        self._query_parts.append("RETURN a, r, b")

    def _add_sparql_relationship_match(
        self,
        source_labels: List[str],
        target_labels: List[str],
        relationship_types: List[str],
    ):
        """Add SPARQL relationship match clause."""
        self._query_parts.append("SELECT ?source ?relationship ?target WHERE {")

        if relationship_types:
            rel_pattern = " | ".join([f"cao:{rt}" for rt in relationship_types])
            self._query_parts.append(f"  ?source ({rel_pattern}) ?target .")
        else:
            self._query_parts.append("  ?source ?relationship ?target .")

        if source_labels:
            for label in source_labels:
                self._query_parts.append(f"  ?source rdf:type cao:{label} .")

        if target_labels:
            for label in target_labels:
                self._query_parts.append(f"  ?target rdf:type cao:{label} .")

        self._query_parts.append("}")

    def _add_cypher_where(self, conditions: Dict[str, Any]):
        """Add Cypher WHERE conditions."""
        where_conditions = []
        for key, value in conditions.items():
            param_name = f"where_{key}"
            where_conditions.append(f"n.{key} = ${param_name}")
            self._parameters[param_name] = value

        if where_conditions:
            self._query_parts.append("WHERE " + " AND ".join(where_conditions))

    def _add_sparql_filter(self, conditions: Dict[str, Any]):
        """Add SPARQL FILTER conditions."""
        filters = []
        for key, value in conditions.items():
            if isinstance(value, str):
                filters.append(f'?{key} = "{value}"')
            else:
                filters.append(f"?{key} = {value}")

        if filters:
            self._query_parts.append("FILTER(" + " && ".join(filters) + ")")


class ChemicalQuery:
    """Specialized query builder for chemical entities and relationships."""

    def __init__(self):
        self.entity_types = []
        self.properties = {}
        self.relationships = []
        self.similarity_constraints = []
        self.substructure_constraints = []

    def find_compounds(
        self,
        molecular_formula: str = None,
        smiles: str = None,
        molecular_weight_range: tuple = None,
    ):
        """Find chemical compounds by molecular properties."""
        self.entity_types.append(EntityType.SMALL_MOLECULE)

        if molecular_formula:
            self.properties["molecular_formula"] = molecular_formula
        if smiles:
            self.properties["smiles"] = smiles
        if molecular_weight_range:
            self.properties["molecular_weight_min"] = molecular_weight_range[0]
            self.properties["molecular_weight_max"] = molecular_weight_range[1]

        return self

    def find_reactions(
        self,
        reaction_type: str = None,
        catalyst: str = None,
        yield_threshold: float = None,
    ):
        """Find chemical reactions by type and conditions."""
        self.entity_types.append(EntityType.REACTION)

        if reaction_type:
            self.properties["reaction_type"] = reaction_type
        if catalyst:
            self.properties["catalyst"] = catalyst
        if yield_threshold:
            self.properties["yield_min"] = yield_threshold

        return self

    def with_relationship(
        self, relationship_type: RelationshipType, target_entity_type: EntityType = None
    ):
        """Add relationship constraint."""
        self.relationships.append(
            {"type": relationship_type, "target_type": target_entity_type}
        )
        return self

    def similar_to(self, reference_smiles: str, similarity_threshold: float = 0.7):
        """Add molecular similarity constraint."""
        self.similarity_constraints.append(
            {"reference_smiles": reference_smiles, "threshold": similarity_threshold}
        )
        return self

    def contains_substructure(self, substructure_smiles: str):
        """Add substructure search constraint."""
        self.substructure_constraints.append({"substructure": substructure_smiles})
        return self

    def to_graph_query(self, backend_type) -> GraphQuery:
        """Convert chemical query to backend-specific graph query."""
        if backend_type.value == "neo4j":
            return self._to_cypher_query()
        elif backend_type.value == "jena":
            return self._to_sparql_query()
        else:
            return self._to_generic_query()

    def _to_cypher_query(self) -> GraphQuery:
        """Convert to Cypher query for Neo4j."""
        query_parts = []
        parameters = {}

        # Build node match
        if self.entity_types:
            labels = [et.value for et in self.entity_types]
            label_str = ":" + ":".join(labels)
            query_parts.append(f"MATCH (n{label_str})")
        else:
            query_parts.append("MATCH (n)")

        # Add property constraints
        if self.properties:
            where_conditions = []
            for key, value in self.properties.items():
                if key.endswith("_min"):
                    prop_name = key[:-4]
                    param_name = f"min_{prop_name}"
                    where_conditions.append(f"n.{prop_name} >= ${param_name}")
                    parameters[param_name] = value
                elif key.endswith("_max"):
                    prop_name = key[:-4]
                    param_name = f"max_{prop_name}"
                    where_conditions.append(f"n.{prop_name} <= ${param_name}")
                    parameters[param_name] = value
                else:
                    param_name = f"prop_{key}"
                    where_conditions.append(f"n.{key} = ${param_name}")
                    parameters[param_name] = value

            if where_conditions:
                query_parts.append("WHERE " + " AND ".join(where_conditions))

        # Add relationship constraints
        for rel in self.relationships:
            # This would need more sophisticated relationship handling
            pass

        query_parts.append("RETURN n")
        query = " ".join(query_parts)

        return GraphQuery(query=query, parameters=parameters, query_type="cypher")

    def _to_sparql_query(self) -> GraphQuery:
        """Convert to SPARQL query for Jena."""
        query_parts = ["SELECT ?compound WHERE {"]

        # Add type constraints
        if self.entity_types:
            for et in self.entity_types:
                query_parts.append(f"  ?compound rdf:type cao:{et.value} .")

        # Add property constraints
        for key, value in self.properties.items():
            if isinstance(value, str):
                query_parts.append(f'  ?compound cao:{key} "{value}" .')
            else:
                query_parts.append(f"  ?compound cao:{key} {value} .")

        query_parts.append("}")
        query = "\n".join(query_parts)

        return GraphQuery(query=query, parameters={}, query_type="sparql")

    def _to_generic_query(self) -> GraphQuery:
        """Convert to generic query format."""
        query_data = {
            "entity_types": [et.value for et in self.entity_types],
            "properties": self.properties,
            "relationships": self.relationships,
            "similarity_constraints": self.similarity_constraints,
            "substructure_constraints": self.substructure_constraints,
        }

        return GraphQuery(
            query=json.dumps(query_data),
            parameters=query_data,
            query_type="chemical_search",
        )


class SemanticQuery:
    """Semantic query builder for ontology-aware searches."""

    def __init__(self, ontology_uri: str = "http://chemi-accel.org/ontology/"):
        self.ontology_uri = ontology_uri
        self.concepts = []
        self.semantic_relationships = []
        self.inference_rules = []

    def find_concept(self, concept_uri: str, include_subclasses: bool = True):
        """Find entities of a specific ontological concept."""
        self.concepts.append(
            {"uri": concept_uri, "include_subclasses": include_subclasses}
        )
        return self

    def with_semantic_relationship(
        self, relationship_uri: str, target_concept: str = None
    ):
        """Add semantic relationship constraint."""
        self.semantic_relationships.append(
            {"relationship": relationship_uri, "target_concept": target_concept}
        )
        return self

    def with_inference(self, rule_name: str, parameters: Dict[str, Any] = None):
        """Add inference rule to query."""
        self.inference_rules.append({"rule": rule_name, "parameters": parameters or {}})
        return self

    def to_sparql(self) -> GraphQuery:
        """Convert to SPARQL query with inference."""
        query_parts = ["SELECT DISTINCT ?resource WHERE {"]

        # Add concept matching
        for concept in self.concepts:
            if concept["include_subclasses"]:
                query_parts.append(
                    f"  ?resource rdf:type/rdfs:subClassOf* <{concept['uri']}> ."
                )
            else:
                query_parts.append(f"  ?resource rdf:type <{concept['uri']}> .")

        # Add semantic relationships
        for rel in self.semantic_relationships:
            if rel["target_concept"]:
                query_parts.append(f"  ?resource <{rel['relationship']}> ?target .")
                query_parts.append(f"  ?target rdf:type <{rel['target_concept']}> .")
            else:
                query_parts.append(f"  ?resource <{rel['relationship']}> ?value .")

        query_parts.append("}")
        query = "\n".join(query_parts)

        return GraphQuery(query=query, parameters={}, query_type="sparql")

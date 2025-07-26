"""
Apache Jena RDF/SPARQL Backend

Semantic web integration using RDF triples and SPARQL queries
for the IC-KG with comprehensive ontology support.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

from .graph_database import (
    BulkLoadConfig,
    GraphDatabaseBackend,
    GraphDatabaseType,
    GraphQuery,
    GraphQueryResult,
    IndexConfig,
)

logger = logging.getLogger(__name__)


class JenaBackend(GraphDatabaseBackend):
    """Apache Jena RDF/SPARQL database backend implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.graph = None
        self.namespace_manager = None
        self._setup_rdf_config()

    def _get_backend_type(self) -> GraphDatabaseType:
        return GraphDatabaseType.JENA

    def _setup_rdf_config(self):
        """Setup RDF/SPARQL configuration."""
        self.store_type = self.config.get(
            "store_type", "memory"
        )  # memory, berkeleydb, mysql, postgresql
        self.store_config = self.config.get("store_config", {})
        self.base_uri = self.config.get("base_uri", "http://chemi-accel.org/kg/")
        self.ontology_uri = self.config.get(
            "ontology_uri", "http://chemi-accel.org/ontology/"
        )

        # Predefined namespaces
        self.namespaces = {
            "ca": self.base_uri,
            "cao": self.ontology_uri,
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "dc": "http://purl.org/dc/elements/1.1/",
            "dcterms": "http://purl.org/dc/terms/",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "chem": "http://semanticscience.org/resource/",
            "cheminf": "http://semanticscience.org/ontology/cheminf.owl#",
        }

    def connect(self) -> bool:
        """Establish connection to RDF store."""
        try:
            import rdflib
            from rdflib.namespace import Namespace, NamespaceManager

            # Create graph with specified store
            if self.store_type == "memory":
                self.graph = rdflib.Graph()
            else:
                # For persistent stores
                store = rdflib.plugin.get(self.store_type, rdflib.store.Store)()
                self.graph = rdflib.Graph(store=store)

                # Configure persistent store
                if self.store_config:
                    self.graph.open(self.store_config, create=True)

            # Setup namespace manager
            self.namespace_manager = NamespaceManager(self.graph)
            for prefix, uri in self.namespaces.items():
                self.namespace_manager.bind(prefix, Namespace(uri))

            # Load base ontology if specified
            ontology_file = self.config.get("ontology_file")
            if ontology_file:
                self._load_ontology(ontology_file)

            logger.info(f"Connected to Jena RDF store ({self.store_type})")
            return True

        except ImportError:
            logger.error("RDFLib not available. Install with: pip install rdflib")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Jena store: {e}")
            return False

    def disconnect(self) -> bool:
        """Close connection to RDF store."""
        try:
            if self.graph and hasattr(self.graph, "close"):
                self.graph.close()
            self.graph = None
            self.namespace_manager = None
            logger.info("Disconnected from Jena RDF store")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Jena store: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if RDF store is connected."""
        return self.graph is not None

    def execute_query(self, query: GraphQuery) -> GraphQueryResult:
        """Execute a SPARQL query."""
        if not self.is_connected():
            raise ConnectionError("Not connected to RDF store")

        start_time = time.time()

        try:
            # Execute SPARQL query
            if query.query_type.lower() == "sparql":
                result = self.graph.query(query.query, initNs=self.namespaces)
            else:
                raise ValueError(f"Unsupported query type: {query.query_type}")

            # Convert results to standardized format
            records = []
            if hasattr(result, "bindings"):
                # SELECT query results
                for binding in result.bindings:
                    record = {}
                    for var, value in binding.items():
                        record[str(var)] = self._rdf_value_to_python(value)
                    records.append(record)
            elif hasattr(result, "graph"):
                # CONSTRUCT query results
                records = [{"triples": len(result.graph)}]
            else:
                # ASK query or other boolean results
                records = [{"result": bool(result)}]

            execution_time = time.time() - start_time

            # Create summary
            summary = {
                "query_type": query.query_type,
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
            logger.error(f"SPARQL query execution failed: {e}")

            return GraphQueryResult(
                records=[],
                summary={"error": str(e)},
                execution_time=execution_time,
                query=query,
                metadata={"error": True},
            )

    def create_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
        """Create an RDF resource with types and properties."""
        import rdflib
        from rdflib import RDF, RDFS

        # Generate URI for the resource
        node_id = str(uuid.uuid4())
        resource_uri = rdflib.URIRef(f"{self.base_uri}resource/{node_id}")

        # Add RDF types based on labels
        for label in labels:
            type_uri = rdflib.URIRef(f"{self.ontology_uri}{label}")
            self.graph.add((resource_uri, RDF.type, type_uri))

        # Add properties
        for key, value in properties.items():
            predicate = self._property_to_predicate(key)
            rdf_value = self._python_value_to_rdf(value)
            self.graph.add((resource_uri, predicate, rdf_value))

        # Add metadata
        self.graph.add(
            (
                resource_uri,
                rdflib.URIRef(f"{self.ontology_uri}id"),
                rdflib.Literal(node_id),
            )
        )
        self.graph.add(
            (
                resource_uri,
                rdflib.URIRef(f"{self.ontology_uri}createdAt"),
                rdflib.Literal(
                    datetime.now().isoformat(), datatype=rdflib.XSD.dateTime
                ),
            )
        )

        self.statistics["nodes_created"] += 1
        return node_id

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
    ) -> str:
        """Create an RDF relationship (object property)."""
        import rdflib

        source_uri = rdflib.URIRef(f"{self.base_uri}resource/{source_id}")
        target_uri = rdflib.URIRef(f"{self.base_uri}resource/{target_id}")
        predicate = rdflib.URIRef(f"{self.ontology_uri}{relationship_type}")

        # Create the main relationship
        self.graph.add((source_uri, predicate, target_uri))

        # If properties are provided, create a reified statement
        rel_id = str(uuid.uuid4())
        if properties:
            statement_uri = rdflib.URIRef(f"{self.base_uri}statement/{rel_id}")

            # Reification
            self.graph.add((statement_uri, rdflib.RDF.type, rdflib.RDF.Statement))
            self.graph.add((statement_uri, rdflib.RDF.subject, source_uri))
            self.graph.add((statement_uri, rdflib.RDF.predicate, predicate))
            self.graph.add((statement_uri, rdflib.RDF.object, target_uri))

            # Add properties to the reified statement
            for key, value in properties.items():
                prop_predicate = self._property_to_predicate(key)
                rdf_value = self._python_value_to_rdf(value)
                self.graph.add((statement_uri, prop_predicate, rdf_value))

            # Add metadata
            self.graph.add(
                (
                    statement_uri,
                    rdflib.URIRef(f"{self.ontology_uri}id"),
                    rdflib.Literal(rel_id),
                )
            )

        self.statistics["relationships_created"] += 1
        return rel_id

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get RDF resource by ID."""
        import rdflib

        resource_uri = rdflib.URIRef(f"{self.base_uri}resource/{node_id}")

        # Check if resource exists
        if (resource_uri, None, None) not in self.graph:
            return None

        node_data = {"_id": node_id, "labels": [], "properties": {}}

        # Get all triples where this resource is the subject
        for predicate, obj in self.graph.predicate_objects(resource_uri):
            pred_str = str(predicate)

            if pred_str == str(rdflib.RDF.type):
                # This is a type/label
                type_name = str(obj).replace(self.ontology_uri, "")
                node_data["labels"].append(type_name)
            else:
                # This is a property
                prop_name = self._predicate_to_property(predicate)
                node_data["properties"][prop_name] = self._rdf_value_to_python(obj)

        return node_data

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update RDF resource properties."""
        import rdflib

        resource_uri = rdflib.URIRef(f"{self.base_uri}resource/{node_id}")

        # Check if resource exists
        if (resource_uri, None, None) not in self.graph:
            return False

        # Update properties (replace existing values)
        for key, value in properties.items():
            predicate = self._property_to_predicate(key)

            # Remove existing values for this property
            self.graph.remove((resource_uri, predicate, None))

            # Add new value
            rdf_value = self._python_value_to_rdf(value)
            self.graph.add((resource_uri, predicate, rdf_value))

        # Add updated timestamp
        updated_predicate = rdflib.URIRef(f"{self.ontology_uri}updatedAt")
        self.graph.remove((resource_uri, updated_predicate, None))
        self.graph.add(
            (
                resource_uri,
                updated_predicate,
                rdflib.Literal(
                    datetime.now().isoformat(), datatype=rdflib.XSD.dateTime
                ),
            )
        )

        return True

    def delete_node(self, node_id: str) -> bool:
        """Delete RDF resource by ID."""
        import rdflib

        resource_uri = rdflib.URIRef(f"{self.base_uri}resource/{node_id}")

        # Remove all triples where this resource is subject or object
        triples_to_remove = []

        # As subject
        for predicate, obj in self.graph.predicate_objects(resource_uri):
            triples_to_remove.append((resource_uri, predicate, obj))

        # As object
        for subject, predicate in self.graph.subject_predicates(resource_uri):
            triples_to_remove.append((subject, predicate, resource_uri))

        # Remove triples
        for triple in triples_to_remove:
            self.graph.remove(triple)

        return len(triples_to_remove) > 0

    def find_nodes(
        self,
        labels: List[str] = None,
        properties: Dict[str, Any] = None,
        limit: int = None,
    ) -> List[Dict[str, Any]]:
        """Find RDF resources matching criteria."""
        import rdflib

        # Build SPARQL query
        query_parts = ["SELECT DISTINCT ?resource WHERE {"]

        # Add type constraints
        if labels:
            for i, label in enumerate(labels):
                query_parts.append(f"  ?resource rdf:type cao:{label} .")

        # Add property constraints
        if properties:
            for key, value in properties.items():
                pred_name = key.replace("_", "")  # Simple mapping
                if isinstance(value, str):
                    query_parts.append(f'  ?resource cao:{pred_name} "{value}" .')
                else:
                    query_parts.append(f"  ?resource cao:{pred_name} {value} .")

        query_parts.append("}")

        if limit:
            query_parts.append(f" LIMIT {limit}")

        sparql_query = "\n".join(query_parts)

        # Execute query
        result = self.execute_query(GraphQuery(sparql_query, query_type="sparql"))

        # Get full node data for each result
        nodes = []
        for record in result.records:
            resource_uri = record.get("resource")
            if resource_uri:
                # Extract node ID from URI
                node_id = resource_uri.replace(f"{self.base_uri}resource/", "")
                node_data = self.get_node(node_id)
                if node_data:
                    nodes.append(node_data)

        return nodes

    def create_index(self, config: IndexConfig) -> bool:
        """Create index (limited support in RDF stores)."""
        # RDF stores typically handle indexing automatically
        # Store configuration for reference
        index_name = (
            f"{config.entity_type}_{'_'.join(config.properties)}_{config.index_type}"
        )
        self.indexes[index_name] = config

        logger.info(f"Index configuration stored: {index_name}")
        return True

    def drop_index(self, index_name: str) -> bool:
        """Drop index configuration."""
        if index_name in self.indexes:
            del self.indexes[index_name]
            return True
        return False

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List index configurations."""
        return [
            {
                "name": name,
                "entity_type": config.entity_type,
                "properties": config.properties,
                "index_type": config.index_type,
                "unique": config.unique,
            }
            for name, config in self.indexes.items()
        ]

    def bulk_load_nodes(
        self, nodes: List[Dict[str, Any]], config: BulkLoadConfig = None
    ) -> Dict[str, Any]:
        """Bulk load RDF resources."""
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
                labels = node.get("labels", ["Resource"])
                properties = node.get("properties", {})

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
        """Bulk load RDF relationships."""
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
                relationship_type = rel.get("relationship_type", "relatesTo")
                properties = rel.get("properties", {})

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
        """Get RDF store statistics."""
        try:
            # Count triples
            total_triples = len(self.graph)

            # Count resources (distinct subjects)
            subjects_query = "SELECT DISTINCT ?s WHERE { ?s ?p ?o }"
            result = self.execute_query(GraphQuery(subjects_query, query_type="sparql"))
            total_resources = len(result.records)

            # Get type distribution
            types_query = """
            SELECT ?type (COUNT(?resource) as ?count) WHERE {
                ?resource rdf:type ?type .
                FILTER(STRSTARTS(STR(?type), "{}"))
            } GROUP BY ?type ORDER BY DESC(?count)
            """.format(
                self.ontology_uri
            )

            result = self.execute_query(GraphQuery(types_query, query_type="sparql"))
            type_distribution = {
                r["type"].replace(self.ontology_uri, ""): r["count"]
                for r in result.records
            }

            # Combine with internal statistics
            combined_stats = {
                **self.statistics,
                "database": {
                    "total_triples": total_triples,
                    "total_resources": total_resources,
                    "type_distribution": type_distribution,
                },
                "connection": {
                    "store_type": self.store_type,
                    "base_uri": self.base_uri,
                    "connected": self.is_connected(),
                },
            }

            return combined_stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {**self.statistics, "error": str(e)}

    def clear_database(self) -> bool:
        """Clear all RDF data."""
        try:
            # Remove all triples
            self.graph.remove((None, None, None))

            # Reset statistics
            self.statistics.update({"nodes_created": 0, "relationships_created": 0})

            return True

        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

    def _load_ontology(self, ontology_file: str):
        """Load ontology from file."""
        try:
            self.graph.parse(ontology_file)
            logger.info(f"Loaded ontology from {ontology_file}")
        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")

    def _property_to_predicate(self, property_name: str):
        """Convert property name to RDF predicate URI."""
        import rdflib

        # Simple mapping - could be more sophisticated
        if property_name.startswith("_"):
            return rdflib.URIRef(f"{self.ontology_uri}{property_name[1:]}")
        else:
            return rdflib.URIRef(f"{self.ontology_uri}{property_name}")

    def _predicate_to_property(self, predicate):
        """Convert RDF predicate URI to property name."""
        pred_str = str(predicate)
        if pred_str.startswith(self.ontology_uri):
            return pred_str.replace(self.ontology_uri, "")
        else:
            # Extract local name
            return pred_str.split("/")[-1].split("#")[-1]

    def _python_value_to_rdf(self, value):
        """Convert Python value to RDF value."""
        import rdflib

        if isinstance(value, str):
            return rdflib.Literal(value)
        elif isinstance(value, int):
            return rdflib.Literal(value, datatype=rdflib.XSD.integer)
        elif isinstance(value, float):
            return rdflib.Literal(value, datatype=rdflib.XSD.double)
        elif isinstance(value, bool):
            return rdflib.Literal(value, datatype=rdflib.XSD.boolean)
        elif isinstance(value, datetime):
            return rdflib.Literal(value.isoformat(), datatype=rdflib.XSD.dateTime)
        else:
            return rdflib.Literal(str(value))

    def _rdf_value_to_python(self, rdf_value):
        """Convert RDF value to Python value."""
        import rdflib

        if isinstance(rdf_value, rdflib.Literal):
            if rdf_value.datatype == rdflib.XSD.integer:
                return int(rdf_value)
            elif rdf_value.datatype == rdflib.XSD.double:
                return float(rdf_value)
            elif rdf_value.datatype == rdflib.XSD.boolean:
                return bool(rdf_value)
            elif rdf_value.datatype == rdflib.XSD.dateTime:
                return datetime.fromisoformat(str(rdf_value))
            else:
                return str(rdf_value)
        elif isinstance(rdf_value, rdflib.URIRef):
            return str(rdf_value)
        else:
            return str(rdf_value)

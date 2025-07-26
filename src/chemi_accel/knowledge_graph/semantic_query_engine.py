"""
Semantic Query and Reasoning Engine for IC-KG

Advanced semantic querying capabilities with multi-language support (SPARQL, Cypher-like, Natural Language)
and intelligent reasoning engine for fact inference and relationship discovery.
"""

import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .graph_database import GraphDatabaseBackend, GraphQuery, GraphQueryResult
from .ontology import ChemicalEntity, ChemicalOntology, EntityType, RelationshipType
from .schema import ICKGSchema


class QueryLanguage(Enum):
    """Supported query language types."""

    SPARQL = "sparql"
    CYPHER = "cypher"
    SEMANTIC_CYPHER = "semantic_cypher"
    NATURAL_LANGUAGE = "natural_language"
    CHEMICAL_DSL = "chemical_dsl"


class ReasoningStrategy(Enum):
    """Reasoning strategy types."""

    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    HYBRID = "hybrid"
    PROBABILISTIC = "probabilistic"


@dataclass
class SemanticQueryContext:
    """Context for semantic query execution."""

    query_language: QueryLanguage
    ontology_namespaces: Dict[str, str] = field(default_factory=dict)
    reasoning_enabled: bool = True
    reasoning_depth: int = 3
    confidence_threshold: float = 0.7
    include_inferred_facts: bool = True
    query_timeout: int = 30
    cache_results: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferredFact:
    """Represents an inferred fact from reasoning."""

    subject: str
    predicate: str
    object: str
    confidence: float
    reasoning_path: List[str]
    source_facts: List[str]
    inferred_at: datetime
    rule_applied: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningRule:
    """Represents a reasoning rule."""

    rule_id: str
    name: str
    description: str
    pattern: str
    conclusion_template: str
    confidence_boost: float = 1.0
    conditions: List[str] = field(default_factory=list)
    priority: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticQueryProcessor(ABC):
    """Abstract base class for semantic query processors."""

    @abstractmethod
    def parse_query(self, query: str, context: SemanticQueryContext) -> Dict[str, Any]:
        """Parse a semantic query into structured format."""
        pass

    @abstractmethod
    def validate_query(
        self, parsed_query: Dict[str, Any], context: SemanticQueryContext
    ) -> bool:
        """Validate parsed query structure and semantics."""
        pass

    @abstractmethod
    def optimize_query(
        self, parsed_query: Dict[str, Any], context: SemanticQueryContext
    ) -> Dict[str, Any]:
        """Optimize query for execution."""
        pass

    @abstractmethod
    def translate_to_backend_query(
        self, parsed_query: Dict[str, Any], backend_type: str
    ) -> GraphQuery:
        """Translate semantic query to backend-specific query."""
        pass


class SPARQLProcessor(SemanticQueryProcessor):
    """SPARQL query processor with semantic extensions."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.prefixes = {
            "cao": "http://chemi-accel.org/ontology/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
        }

    def parse_query(self, query: str, context: SemanticQueryContext) -> Dict[str, Any]:
        """Parse SPARQL query with semantic extensions."""
        # Extract prefixes
        prefixes = self._extract_prefixes(query)
        prefixes.update(context.ontology_namespaces)

        # Extract query type (SELECT, ASK, CONSTRUCT, DESCRIBE)
        query_type = self._extract_query_type(query)

        # Extract SELECT variables
        select_vars = (
            self._extract_select_variables(query) if query_type == "SELECT" else []
        )

        # Extract WHERE clause patterns
        where_patterns = self._extract_where_patterns(query)

        # Extract FILTER conditions
        filters = self._extract_filters(query)

        # Extract ORDER BY, LIMIT, OFFSET
        modifiers = self._extract_query_modifiers(query)

        return {
            "query_type": query_type,
            "prefixes": prefixes,
            "select_variables": select_vars,
            "where_patterns": where_patterns,
            "filters": filters,
            "modifiers": modifiers,
            "original_query": query,
        }

    def validate_query(
        self, parsed_query: Dict[str, Any], context: SemanticQueryContext
    ) -> bool:
        """Validate SPARQL query structure."""
        # Check required components
        if not parsed_query.get("query_type"):
            return False

        # Validate variable references
        if parsed_query["query_type"] == "SELECT":
            if not parsed_query.get("select_variables"):
                return False

        # Validate WHERE patterns
        if not parsed_query.get("where_patterns"):
            return False

        # Validate namespace prefixes
        for pattern in parsed_query["where_patterns"]:
            if not self._validate_pattern_namespaces(pattern, parsed_query["prefixes"]):
                return False

        return True

    def optimize_query(
        self, parsed_query: Dict[str, Any], context: SemanticQueryContext
    ) -> Dict[str, Any]:
        """Optimize SPARQL query for execution."""
        optimized = parsed_query.copy()

        # Reorder patterns for optimal execution
        optimized["where_patterns"] = self._optimize_pattern_order(
            parsed_query["where_patterns"]
        )

        # Push filters down to reduce intermediate results
        optimized = self._push_down_filters(optimized)

        # Add implicit reasoning patterns if enabled
        if context.reasoning_enabled:
            optimized = self._add_reasoning_patterns(optimized, context)

        return optimized

    def translate_to_backend_query(
        self, parsed_query: Dict[str, Any], backend_type: str
    ) -> GraphQuery:
        """Translate SPARQL to backend-specific query."""
        if backend_type == "jena":
            # Direct SPARQL execution
            return GraphQuery(
                query=parsed_query["original_query"],
                parameters={"prefixes": parsed_query["prefixes"]},
                query_type="sparql",
            )

        elif backend_type == "neo4j":
            # Convert to Cypher
            cypher_query = self._sparql_to_cypher(parsed_query)
            return GraphQuery(query=cypher_query, parameters={}, query_type="cypher")

        else:
            # Convert to generic query format
            return self._sparql_to_generic(parsed_query)

    def _extract_prefixes(self, query: str) -> Dict[str, str]:
        """Extract PREFIX declarations from SPARQL query."""
        prefixes = self.prefixes.copy()

        prefix_pattern = r"PREFIX\s+(\w+):\s*<([^>]+)>"
        matches = re.findall(prefix_pattern, query, re.IGNORECASE)

        for prefix, uri in matches:
            prefixes[prefix] = uri

        return prefixes

    def _extract_query_type(self, query: str) -> str:
        """Extract query type (SELECT, ASK, etc.)."""
        query_types = ["SELECT", "ASK", "CONSTRUCT", "DESCRIBE"]

        for qtype in query_types:
            if re.search(rf"\b{qtype}\b", query, re.IGNORECASE):
                return qtype

        return "SELECT"  # Default

    def _extract_select_variables(self, query: str) -> List[str]:
        """Extract SELECT variables."""
        select_match = re.search(
            r"SELECT\s+(.+?)\s+WHERE", query, re.IGNORECASE | re.DOTALL
        )
        if not select_match:
            return []

        vars_str = select_match.group(1).strip()

        if vars_str == "*":
            return ["*"]

        # Extract individual variables
        var_pattern = r"\?(\w+)"
        return re.findall(var_pattern, vars_str)

    def _extract_where_patterns(self, query: str) -> List[Dict[str, str]]:
        """Extract WHERE clause triple patterns."""
        where_match = re.search(r"WHERE\s*\{(.+)\}", query, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return []

        where_content = where_match.group(1).strip()

        # Split into triple patterns (simplified)
        patterns = []
        triple_pattern = r"([^\s]+)\s+([^\s]+)\s+([^\s\.]+)"

        for match in re.finditer(triple_pattern, where_content):
            subject, predicate, obj = match.groups()
            patterns.append(
                {
                    "subject": subject.strip(),
                    "predicate": predicate.strip(),
                    "object": obj.strip(),
                }
            )

        return patterns

    def _extract_filters(self, query: str) -> List[str]:
        """Extract FILTER conditions."""
        filter_pattern = r"FILTER\s*\(([^)]+)\)"
        return re.findall(filter_pattern, query, re.IGNORECASE)

    def _extract_query_modifiers(self, query: str) -> Dict[str, Any]:
        """Extract ORDER BY, LIMIT, OFFSET modifiers."""
        modifiers = {}

        # ORDER BY
        order_match = re.search(
            r"ORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s+OFFSET|$)", query, re.IGNORECASE
        )
        if order_match:
            modifiers["order_by"] = order_match.group(1).strip()

        # LIMIT
        limit_match = re.search(r"LIMIT\s+(\d+)", query, re.IGNORECASE)
        if limit_match:
            modifiers["limit"] = int(limit_match.group(1))

        # OFFSET
        offset_match = re.search(r"OFFSET\s+(\d+)", query, re.IGNORECASE)
        if offset_match:
            modifiers["offset"] = int(offset_match.group(1))

        return modifiers

    def _validate_pattern_namespaces(
        self, pattern: Dict[str, str], prefixes: Dict[str, str]
    ) -> bool:
        """Validate that pattern uses valid namespace prefixes."""
        for component in [pattern["subject"], pattern["predicate"], pattern["object"]]:
            if ":" in component and not component.startswith("?"):
                prefix = component.split(":")[0]
                if prefix not in prefixes:
                    return False
        return True

    def _optimize_pattern_order(
        self, patterns: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Reorder patterns for optimal execution."""

        # Simple heuristic: put patterns with fewer variables first
        def pattern_specificity(pattern):
            var_count = sum(1 for v in pattern.values() if v.startswith("?"))
            return var_count

        return sorted(patterns, key=pattern_specificity)

    def _push_down_filters(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Push filter conditions closer to relevant patterns."""
        # Simplified implementation
        return parsed_query

    def _add_reasoning_patterns(
        self, parsed_query: Dict[str, Any], context: SemanticQueryContext
    ) -> Dict[str, Any]:
        """Add implicit reasoning patterns for inference."""
        enhanced = parsed_query.copy()

        # Add rdfs:subClassOf patterns for class hierarchy reasoning
        additional_patterns = []
        for pattern in parsed_query["where_patterns"]:
            if pattern["predicate"] == "rdf:type":
                additional_patterns.append(
                    {
                        "subject": pattern["object"],
                        "predicate": "rdfs:subClassOf*",
                        "object": "?superclass",
                    }
                )

        enhanced["where_patterns"].extend(additional_patterns)
        return enhanced

    def _sparql_to_cypher(self, parsed_query: Dict[str, Any]) -> str:
        """Convert SPARQL query to Cypher."""
        # Simplified SPARQL to Cypher conversion
        patterns = parsed_query["where_patterns"]

        if not patterns:
            return "MATCH (n) RETURN n LIMIT 10"

        # Convert first pattern to Cypher MATCH
        first_pattern = patterns[0]

        if first_pattern["subject"].startswith("?") and first_pattern[
            "object"
        ].startswith("?"):
            # Both subject and object are variables
            return f"""
            MATCH ({first_pattern["subject"][1:]})-[:{first_pattern["predicate"].replace(":", "_")}]->({first_pattern["object"][1:]})
            RETURN {first_pattern["subject"][1:]}, {first_pattern["object"][1:]}
            """

        elif first_pattern["subject"].startswith("?"):
            # Subject is variable
            return f"""
            MATCH ({first_pattern["subject"][1:]})-[:{first_pattern["predicate"].replace(":", "_")}]->()
            WHERE {first_pattern["subject"][1:]}.value = '{first_pattern["object"]}'
            RETURN {first_pattern["subject"][1:]}
            """

        else:
            # Generic conversion
            return f"""
            MATCH (n)
            WHERE n.uri = '{first_pattern["subject"]}'
            RETURN n
            """

    def _sparql_to_generic(self, parsed_query: Dict[str, Any]) -> GraphQuery:
        """Convert SPARQL to generic query format."""
        return GraphQuery(
            query=json.dumps(
                {
                    "operation": "semantic_search",
                    "patterns": parsed_query["where_patterns"],
                    "variables": parsed_query.get("select_variables", []),
                    "filters": parsed_query.get("filters", []),
                }
            ),
            parameters={"prefixes": parsed_query["prefixes"]},
            query_type="semantic_query",
        )


class CypherSemanticProcessor(SemanticQueryProcessor):
    """Semantic Cypher processor with ontology awareness."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.semantic_extensions = {
            "SIMILAR_TO": self._expand_similarity_query,
            "SUBCLASS_OF": self._expand_subclass_query,
            "RELATED_TO": self._expand_relationship_query,
        }

    def parse_query(self, query: str, context: SemanticQueryContext) -> Dict[str, Any]:
        """Parse semantic Cypher query."""
        # Extract MATCH patterns
        match_patterns = self._extract_match_patterns(query)

        # Extract WHERE conditions
        where_conditions = self._extract_where_conditions(query)

        # Extract RETURN clause
        return_clause = self._extract_return_clause(query)

        # Extract semantic extensions
        semantic_ops = self._extract_semantic_operations(query)

        return {
            "query_type": "semantic_cypher",
            "match_patterns": match_patterns,
            "where_conditions": where_conditions,
            "return_clause": return_clause,
            "semantic_operations": semantic_ops,
            "original_query": query,
        }

    def validate_query(
        self, parsed_query: Dict[str, Any], context: SemanticQueryContext
    ) -> bool:
        """Validate semantic Cypher query."""
        # Basic structure validation
        if not parsed_query.get("match_patterns") and not parsed_query.get(
            "semantic_operations"
        ):
            return False

        # Validate semantic operations
        for op in parsed_query.get("semantic_operations", []):
            if op["operation"] not in self.semantic_extensions:
                return False

        return True

    def optimize_query(
        self, parsed_query: Dict[str, Any], context: SemanticQueryContext
    ) -> Dict[str, Any]:
        """Optimize semantic Cypher query."""
        optimized = parsed_query.copy()

        # Expand semantic operations
        for op in parsed_query.get("semantic_operations", []):
            if op["operation"] in self.semantic_extensions:
                expansion = self.semantic_extensions[op["operation"]](op, context)
                optimized.setdefault("expanded_patterns", []).extend(expansion)

        return optimized

    def translate_to_backend_query(
        self, parsed_query: Dict[str, Any], backend_type: str
    ) -> GraphQuery:
        """Translate semantic Cypher to backend query."""
        if backend_type == "neo4j":
            # Generate expanded Cypher
            cypher = self._generate_expanded_cypher(parsed_query)
            return GraphQuery(query=cypher, parameters={}, query_type="cypher")

        else:
            # Convert to generic format
            return GraphQuery(
                query=json.dumps(parsed_query),
                parameters={},
                query_type="semantic_query",
            )

    def _extract_match_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Extract MATCH patterns from query."""
        patterns = []
        match_pattern = r"MATCH\s+(.+?)(?:\s+WHERE|\s+RETURN|$)"

        for match in re.finditer(match_pattern, query, re.IGNORECASE):
            patterns.append({"pattern": match.group(1).strip()})

        return patterns

    def _extract_where_conditions(self, query: str) -> List[str]:
        """Extract WHERE conditions."""
        where_match = re.search(
            r"WHERE\s+(.+?)(?:\s+RETURN|$)", query, re.IGNORECASE | re.DOTALL
        )
        if not where_match:
            return []

        return [where_match.group(1).strip()]

    def _extract_return_clause(self, query: str) -> str:
        """Extract RETURN clause."""
        return_match = re.search(r"RETURN\s+(.+)$", query, re.IGNORECASE)
        return return_match.group(1).strip() if return_match else ""

    def _extract_semantic_operations(self, query: str) -> List[Dict[str, Any]]:
        """Extract semantic operations from query."""
        operations = []

        # Look for semantic operators
        for op_name in self.semantic_extensions.keys():
            if op_name in query.upper():
                operations.append(
                    {
                        "operation": op_name,
                        "context": self._extract_operation_context(query, op_name),
                    }
                )

        return operations

    def _extract_operation_context(self, query: str, operation: str) -> Dict[str, Any]:
        """Extract context for semantic operation."""
        # Simplified context extraction
        return {"query": query, "operation": operation}

    def _expand_similarity_query(
        self, operation: Dict[str, Any], context: SemanticQueryContext
    ) -> List[str]:
        """Expand similarity query using ontology."""
        return ["MATCH (n)-[:SIMILAR_TO]-(m)", "WHERE n.similarity_score >= 0.7"]

    def _expand_subclass_query(
        self, operation: Dict[str, Any], context: SemanticQueryContext
    ) -> List[str]:
        """Expand subclass query using ontology hierarchy."""
        return [
            "MATCH (n)-[:SUBCLASS_OF*]-(parent)",
            "WHERE parent.type IN ['Chemical', 'Drug', 'Compound']",
        ]

    def _expand_relationship_query(
        self, operation: Dict[str, Any], context: SemanticQueryContext
    ) -> List[str]:
        """Expand generic relationship query."""
        return [
            "MATCH (n)-[r]-(m)",
            "WHERE type(r) IN ['TREATS', 'CAUSES', 'INTERACTS_WITH']",
        ]

    def _generate_expanded_cypher(self, parsed_query: Dict[str, Any]) -> str:
        """Generate expanded Cypher query."""
        parts = []

        # Add original MATCH patterns
        for pattern in parsed_query.get("match_patterns", []):
            parts.append(f"MATCH {pattern['pattern']}")

        # Add expanded patterns
        for pattern in parsed_query.get("expanded_patterns", []):
            parts.append(pattern)

        # Add WHERE conditions
        if parsed_query.get("where_conditions"):
            parts.append(f"WHERE {' AND '.join(parsed_query['where_conditions'])}")

        # Add RETURN clause
        if parsed_query.get("return_clause"):
            parts.append(f"RETURN {parsed_query['return_clause']}")

        return "\n".join(parts)


class NaturalLanguageProcessor(SemanticQueryProcessor):
    """Natural language query processor using pattern matching and NLP."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.query_patterns = self._initialize_query_patterns()
        self.entity_recognizer = ChemicalEntityRecognizer(ontology)

    def parse_query(self, query: str, context: SemanticQueryContext) -> Dict[str, Any]:
        """Parse natural language query."""
        # Preprocess query
        processed_query = self._preprocess_query(query)

        # Extract entities
        entities = self.entity_recognizer.extract_entities(processed_query)

        # Detect query intent
        intent = self._detect_query_intent(processed_query)

        # Extract query components
        components = self._extract_query_components(processed_query, entities, intent)

        return {
            "query_type": "natural_language",
            "original_query": query,
            "processed_query": processed_query,
            "entities": entities,
            "intent": intent,
            "components": components,
        }

    def validate_query(
        self, parsed_query: Dict[str, Any], context: SemanticQueryContext
    ) -> bool:
        """Validate natural language query."""
        # Check if intent was detected
        if not parsed_query.get("intent"):
            return False

        # Check if relevant entities were found
        if (
            not parsed_query.get("entities")
            and parsed_query["intent"]["type"] != "general"
        ):
            return False

        return True

    def optimize_query(
        self, parsed_query: Dict[str, Any], context: SemanticQueryContext
    ) -> Dict[str, Any]:
        """Optimize natural language query."""
        optimized = parsed_query.copy()

        # Add context-aware entity disambiguation
        optimized["entities"] = self._disambiguate_entities(
            parsed_query["entities"], context
        )

        # Expand query with synonyms and related terms
        if context.reasoning_enabled:
            optimized = self._expand_with_ontology(optimized, context)

        return optimized

    def translate_to_backend_query(
        self, parsed_query: Dict[str, Any], backend_type: str
    ) -> GraphQuery:
        """Translate natural language query to backend query."""
        intent = parsed_query["intent"]
        entities = parsed_query["entities"]

        if intent["type"] == "find_compounds":
            return self._generate_compound_search_query(entities, backend_type)

        elif intent["type"] == "find_relationships":
            return self._generate_relationship_query(entities, backend_type)

        elif intent["type"] == "compare_entities":
            return self._generate_comparison_query(entities, backend_type)

        else:
            return self._generate_generic_query(parsed_query, backend_type)

    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize natural language query patterns."""
        return {
            "find_compounds": [
                r"find .* compounds?",
                r"show .* molecules?",
                r"list .* chemicals?",
                r"what .* substances?",
            ],
            "find_relationships": [
                r"how .* relate",
                r"what .* interaction",
                r"does .* treat",
                r"can .* cause",
            ],
            "compare_entities": [
                r"compare .* and",
                r"difference between",
                r"similar to",
                r"related to",
            ],
            "properties": [
                r"what .* molecular weight",
                r"what .* formula",
                r"properties of",
            ],
        }

    def _preprocess_query(self, query: str) -> str:
        """Preprocess natural language query."""
        # Convert to lowercase
        processed = query.lower().strip()

        # Remove punctuation
        processed = re.sub(r"[^\w\s]", " ", processed)

        # Normalize whitespace
        processed = re.sub(r"\s+", " ", processed)

        return processed

    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """Detect query intent using pattern matching."""
        for intent_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return {"type": intent_type, "confidence": 0.8, "pattern": pattern}

        return {"type": "general", "confidence": 0.5, "pattern": None}

    def _extract_query_components(
        self, query: str, entities: List[Dict], intent: Dict
    ) -> Dict[str, Any]:
        """Extract structured components from query."""
        components = {
            "target_entities": [],
            "properties": [],
            "constraints": [],
            "modifiers": {},
        }

        # Extract target entities based on intent
        if intent["type"] == "find_compounds":
            components["target_entities"] = ["Chemical", "Compound", "Drug"]

        # Extract properties mentioned in query
        property_keywords = ["weight", "formula", "structure", "properties"]
        for keyword in property_keywords:
            if keyword in query:
                components["properties"].append(keyword)

        # Extract numerical constraints
        numbers = re.findall(r"\d+(?:\.\d+)?", query)
        if numbers:
            components["constraints"].extend(numbers)

        return components

    def _disambiguate_entities(
        self, entities: List[Dict], context: SemanticQueryContext
    ) -> List[Dict]:
        """Disambiguate entities using context and ontology."""
        disambiguated = []

        for entity in entities:
            # Look up in ontology
            ontology_matches = self.ontology.find_entity(entity["text"])
            if ontology_matches:
                entity["ontology_uri"] = ontology_matches[0].uri
                entity["confidence"] = min(entity.get("confidence", 1.0) + 0.2, 1.0)

            disambiguated.append(entity)

        return disambiguated

    def _expand_with_ontology(
        self, parsed_query: Dict[str, Any], context: SemanticQueryContext
    ) -> Dict[str, Any]:
        """Expand query using ontology knowledge."""
        expanded = parsed_query.copy()

        # Add related entities
        expanded_entities = []
        for entity in parsed_query["entities"]:
            expanded_entities.append(entity)

            # Add synonyms and related terms
            if "ontology_uri" in entity:
                related = self.ontology.get_related_entities(entity["ontology_uri"])
                for rel_entity in related[:3]:  # Limit to top 3
                    expanded_entities.append(
                        {
                            "text": rel_entity.name,
                            "type": "related",
                            "confidence": 0.6,
                            "source": entity["text"],
                        }
                    )

        expanded["entities"] = expanded_entities
        return expanded

    def _generate_compound_search_query(
        self, entities: List[Dict], backend_type: str
    ) -> GraphQuery:
        """Generate compound search query."""
        if backend_type == "neo4j":
            entity_names = [e["text"] for e in entities if e.get("type") == "chemical"]

            if entity_names:
                query = f"""
                MATCH (n:Chemical)
                WHERE n.name IN {entity_names}
                RETURN n
                """
            else:
                query = "MATCH (n:Chemical) RETURN n LIMIT 20"

            return GraphQuery(query=query, parameters={}, query_type="cypher")

        else:
            return GraphQuery(
                query=json.dumps(
                    {
                        "operation": "find_nodes",
                        "labels": ["Chemical", "Compound"],
                        "entity_filter": [e["text"] for e in entities],
                    }
                ),
                parameters={},
                query_type="semantic_query",
            )

    def _generate_relationship_query(
        self, entities: List[Dict], backend_type: str
    ) -> GraphQuery:
        """Generate relationship search query."""
        if len(entities) >= 2:
            entity1, entity2 = entities[0]["text"], entities[1]["text"]

            if backend_type == "neo4j":
                query = f"""
                MATCH (a {{name: '{entity1}'}})-[r]-(b {{name: '{entity2}'}})
                RETURN a, r, b
                """
            else:
                query = json.dumps(
                    {
                        "operation": "find_relationships",
                        "source_entity": entity1,
                        "target_entity": entity2,
                    }
                )

            return GraphQuery(
                query=query,
                parameters={},
                query_type="cypher" if backend_type == "neo4j" else "semantic_query",
            )

        return self._generate_generic_query({"entities": entities}, backend_type)

    def _generate_comparison_query(
        self, entities: List[Dict], backend_type: str
    ) -> GraphQuery:
        """Generate entity comparison query."""
        entity_names = [e["text"] for e in entities[:2]]  # Limit to 2 entities

        if backend_type == "neo4j":
            query = f"""
            MATCH (n)
            WHERE n.name IN {entity_names}
            RETURN n.name, n.molecular_weight, n.formula, n.properties
            """
        else:
            query = json.dumps(
                {
                    "operation": "compare_entities",
                    "entities": entity_names,
                    "properties": ["molecular_weight", "formula", "structure"],
                }
            )

        return GraphQuery(
            query=query,
            parameters={},
            query_type="cypher" if backend_type == "neo4j" else "semantic_query",
        )

    def _generate_generic_query(
        self, parsed_query: Dict[str, Any], backend_type: str
    ) -> GraphQuery:
        """Generate generic query for unspecified intents."""
        return GraphQuery(
            query=json.dumps(
                {
                    "operation": "general_search",
                    "entities": parsed_query.get("entities", []),
                    "intent": parsed_query.get("intent", {}),
                }
            ),
            parameters={},
            query_type="semantic_query",
        )


class ChemicalEntityRecognizer:
    """Chemical entity recognition for natural language queries."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.chemical_patterns = self._load_chemical_patterns()

    def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract chemical entities from query text."""
        entities = []

        # Extract using patterns
        entities.extend(self._extract_by_patterns(query))

        # Extract using ontology lookup
        entities.extend(self._extract_by_ontology(query))

        # Deduplicate and score
        return self._deduplicate_entities(entities)

    def _load_chemical_patterns(self) -> Dict[str, str]:
        """Load chemical name and formula patterns."""
        return {
            "molecular_formula": r"\b[A-Z][a-z]?(?:\d+[A-Z][a-z]?\d*)*\b",
            "smiles": r"[CNOPSFClBrI][\w\[\]()=#+\-@/\\]*",
            "drug_names": r"\b(?:aspirin|ibuprofen|morphine|caffeine|ethanol)\b",
            "chemical_prefixes": r"\b(?:methyl|ethyl|propyl|butyl|phenyl)\w*\b",
        }

    def _extract_by_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns."""
        entities = []

        for pattern_type, pattern in self.chemical_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append(
                    {
                        "text": match.group(),
                        "type": pattern_type,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.7,
                    }
                )

        return entities

    def _extract_by_ontology(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities by looking up in ontology."""
        entities = []
        words = query.split()

        # Check individual words and bigrams
        for i in range(len(words)):
            # Single word
            word = words[i]
            if len(word) > 2:  # Skip very short words
                ontology_match = self.ontology.find_entity(word)
                if ontology_match:
                    entities.append(
                        {
                            "text": word,
                            "type": "chemical",
                            "ontology_uri": ontology_match[0].uri,
                            "confidence": 0.8,
                        }
                    )

            # Bigram
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                ontology_match = self.ontology.find_entity(bigram)
                if ontology_match:
                    entities.append(
                        {
                            "text": bigram,
                            "type": "chemical",
                            "ontology_uri": ontology_match[0].uri,
                            "confidence": 0.9,
                        }
                    )

        return entities

    def _deduplicate_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate entities and merge overlapping ones."""
        # Sort by start position
        entities.sort(key=lambda x: x.get("start", 0))

        deduplicated = []
        used_positions = set()

        for entity in entities:
            start = entity.get("start", 0)
            end = entity.get("end", start + len(entity["text"]))

            # Check for overlap with already used positions
            overlap = any(start <= pos <= end for pos in used_positions)

            if not overlap:
                deduplicated.append(entity)
                used_positions.update(range(start, end))

        return deduplicated


class ReasoningEngine:
    """Reasoning engine for inferring new facts and relationships."""

    def __init__(self, ontology: ChemicalOntology, schema: ICKGSchema):
        self.ontology = ontology
        self.schema = schema
        self.rules = []
        self.inferred_facts = []
        self.reasoning_cache = {}
        self.max_reasoning_depth = 5
        self.confidence_threshold = 0.5

        # Load default reasoning rules
        self._load_default_rules()

    def add_rule(self, rule: ReasoningRule) -> None:
        """Add a reasoning rule."""
        self.rules.append(rule)
        logging.info(f"Added reasoning rule: {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a reasoning rule."""
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        return len(self.rules) < initial_count

    def infer_facts(
        self,
        backend: GraphDatabaseBackend,
        strategy: ReasoningStrategy = ReasoningStrategy.FORWARD_CHAINING,
    ) -> List[InferredFact]:
        """Infer new facts using specified reasoning strategy."""
        if strategy == ReasoningStrategy.FORWARD_CHAINING:
            return self._forward_chaining(backend)
        elif strategy == ReasoningStrategy.BACKWARD_CHAINING:
            return self._backward_chaining(backend)
        elif strategy == ReasoningStrategy.HYBRID:
            return self._hybrid_reasoning(backend)
        else:
            return self._forward_chaining(backend)  # Default

    def validate_inference(self, fact: InferredFact) -> bool:
        """Validate an inferred fact against ontology and schema."""
        # Check confidence threshold
        if fact.confidence < self.confidence_threshold:
            return False

        # Validate against schema
        if not self.schema.validate_relationship(
            fact.subject, fact.predicate, fact.object
        ):
            return False

        # Check for contradictions with existing knowledge
        if self._check_contradictions(fact):
            return False

        return True

    def explain_inference(self, fact: InferredFact) -> Dict[str, Any]:
        """Provide explanation for an inferred fact."""
        return {
            "fact": f"{fact.subject} {fact.predicate} {fact.object}",
            "confidence": fact.confidence,
            "reasoning_path": fact.reasoning_path,
            "rule_applied": fact.rule_applied,
            "source_facts": fact.source_facts,
            "inferred_at": fact.inferred_at.isoformat(),
            "validation_status": self.validate_inference(fact),
        }

    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get reasoning engine statistics."""
        total_inferences = len(self.inferred_facts)
        valid_inferences = sum(
            1 for f in self.inferred_facts if self.validate_inference(f)
        )

        confidence_distribution = {}
        for fact in self.inferred_facts:
            conf_bucket = f"{int(fact.confidence * 10) * 10}-{int(fact.confidence * 10) * 10 + 10}%"
            confidence_distribution[conf_bucket] = (
                confidence_distribution.get(conf_bucket, 0) + 1
            )

        rule_usage = {}
        for fact in self.inferred_facts:
            rule_usage[fact.rule_applied] = rule_usage.get(fact.rule_applied, 0) + 1

        return {
            "total_inferences": total_inferences,
            "valid_inferences": valid_inferences,
            "invalid_inferences": total_inferences - valid_inferences,
            "validation_rate": valid_inferences / total_inferences
            if total_inferences > 0
            else 0,
            "confidence_distribution": confidence_distribution,
            "rule_usage": rule_usage,
            "active_rules": len([r for r in self.rules if r.enabled]),
            "cache_size": len(self.reasoning_cache),
        }

    def _load_default_rules(self) -> None:
        """Load default reasoning rules for chemical knowledge."""
        # Transitivity rule for chemical relationships
        self.add_rule(
            ReasoningRule(
                rule_id="transitivity_treats",
                name="Transitivity of Treatment",
                description="If A treats B and B causes C, then A may treat C",
                pattern="(?P<drug>\\w+) TREATS (?P<condition1>\\w+) AND (?P<condition1>\\w+) CAUSES (?P<condition2>\\w+)",
                conclusion_template="{drug} TREATS {condition2}",
                confidence_boost=0.6,
                conditions=["confidence > 0.7"],
                priority=1,
            )
        )

        # Similarity inference rule
        self.add_rule(
            ReasoningRule(
                rule_id="similarity_properties",
                name="Similar Compounds Properties",
                description="Similar compounds likely have similar properties",
                pattern="(?P<comp1>\\w+) SIMILAR_TO (?P<comp2>\\w+) AND (?P<comp1>\\w+) HAS_PROPERTY (?P<prop>\\w+)",
                conclusion_template="{comp2} HAS_PROPERTY {prop}",
                confidence_boost=0.7,
                conditions=["similarity > 0.8"],
                priority=2,
            )
        )

        # Chemical class inheritance
        self.add_rule(
            ReasoningRule(
                rule_id="class_inheritance",
                name="Chemical Class Property Inheritance",
                description="Compounds inherit properties from their chemical classes",
                pattern="(?P<compound>\\w+) IS_A (?P<class>\\w+) AND (?P<class>\\w+) HAS_PROPERTY (?P<prop>\\w+)",
                conclusion_template="{compound} HAS_PROPERTY {prop}",
                confidence_boost=0.8,
                conditions=[],
                priority=3,
            )
        )

        # Drug interaction inference
        self.add_rule(
            ReasoningRule(
                rule_id="drug_interaction",
                name="Drug Interaction Inference",
                description="Drugs affecting the same pathway may interact",
                pattern="(?P<drug1>\\w+) AFFECTS (?P<pathway>\\w+) AND (?P<drug2>\\w+) AFFECTS (?P<pathway>\\w+)",
                conclusion_template="{drug1} INTERACTS_WITH {drug2}",
                confidence_boost=0.5,
                conditions=["drug1 != drug2"],
                priority=1,
            )
        )

    def _forward_chaining(self, backend: GraphDatabaseBackend) -> List[InferredFact]:
        """Forward chaining reasoning from facts to conclusions."""
        new_facts = []

        # Get existing facts from backend
        existing_facts = self._get_existing_facts(backend)

        # Apply rules iteratively
        for depth in range(self.max_reasoning_depth):
            round_facts = []

            for rule in self.rules:
                if not rule.enabled:
                    continue

                # Find facts matching rule pattern
                matching_fact_sets = self._find_matching_facts(existing_facts, rule)

                for fact_set in matching_fact_sets:
                    # Generate inference
                    inferred = self._apply_rule(rule, fact_set, depth)
                    if inferred and self.validate_inference(inferred):
                        round_facts.append(inferred)

            if not round_facts:
                break  # No new facts inferred

            new_facts.extend(round_facts)
            existing_facts.extend(round_facts)  # Add for next iteration

        self.inferred_facts.extend(new_facts)
        return new_facts

    def _backward_chaining(self, backend: GraphDatabaseBackend) -> List[InferredFact]:
        """Backward chaining reasoning from goals to supporting facts."""
        # Simplified backward chaining implementation
        goals = self._identify_reasoning_goals(backend)
        new_facts = []

        for goal in goals:
            supporting_facts = self._find_supporting_facts(goal, backend)
            if supporting_facts:
                inferred = InferredFact(
                    subject=goal["subject"],
                    predicate=goal["predicate"],
                    object=goal["object"],
                    confidence=min([f.confidence for f in supporting_facts]),
                    reasoning_path=[
                        f"backward_chain_{i}" for i in range(len(supporting_facts))
                    ],
                    source_facts=[
                        f"{f.subject} {f.predicate} {f.object}"
                        for f in supporting_facts
                    ],
                    inferred_at=datetime.now(),
                    rule_applied="backward_chaining",
                )

                if self.validate_inference(inferred):
                    new_facts.append(inferred)

        self.inferred_facts.extend(new_facts)
        return new_facts

    def _hybrid_reasoning(self, backend: GraphDatabaseBackend) -> List[InferredFact]:
        """Hybrid reasoning combining forward and backward chaining."""
        forward_facts = self._forward_chaining(backend)
        backward_facts = self._backward_chaining(backend)

        # Combine and deduplicate
        all_facts = forward_facts + backward_facts
        return self._deduplicate_facts(all_facts)

    def _get_existing_facts(self, backend: GraphDatabaseBackend) -> List[InferredFact]:
        """Get existing facts from the knowledge graph."""
        # Query all relationships
        query = GraphQuery(
            query="find_all_relationships",
            parameters={},
            query_type="find_relationships",
        )

        result = backend.execute_query(query)
        facts = []

        for record in result.records:
            fact = InferredFact(
                subject=record.get("source_name", record.get("source_id", "")),
                predicate=record.get("relationship_type", ""),
                object=record.get("target_name", record.get("target_id", "")),
                confidence=1.0,  # Existing facts have full confidence
                reasoning_path=["existing_knowledge"],
                source_facts=["knowledge_graph"],
                inferred_at=datetime.now(),
                rule_applied="existing_knowledge",
            )
            facts.append(fact)

        return facts

    def _find_matching_facts(
        self, facts: List[InferredFact], rule: ReasoningRule
    ) -> List[List[InferredFact]]:
        """Find sets of facts that match a reasoning rule pattern."""
        # Simplified pattern matching
        matching_sets = []

        # For transitivity rules (A->B, B->C patterns)
        if "transitivity" in rule.rule_id.lower():
            for fact1 in facts:
                for fact2 in facts:
                    if (
                        fact1.object == fact2.subject
                        and fact1.predicate == rule.metadata.get("first_relation")
                        and fact2.predicate == rule.metadata.get("second_relation")
                    ):
                        matching_sets.append([fact1, fact2])

        # For similarity rules
        elif "similarity" in rule.rule_id.lower():
            for fact1 in facts:
                for fact2 in facts:
                    if (
                        fact1.predicate == "SIMILAR_TO"
                        and fact2.subject == fact1.subject
                        and fact2.predicate == "HAS_PROPERTY"
                    ):
                        matching_sets.append([fact1, fact2])

        return matching_sets

    def _apply_rule(
        self, rule: ReasoningRule, fact_set: List[InferredFact], depth: int
    ) -> Optional[InferredFact]:
        """Apply a reasoning rule to a set of facts."""
        if not fact_set:
            return None

        # Calculate confidence based on source facts and rule
        base_confidence = min(fact.confidence for fact in fact_set)
        adjusted_confidence = (
            base_confidence * rule.confidence_boost * (0.9**depth)
        )  # Decrease with depth

        if adjusted_confidence < self.confidence_threshold:
            return None

        # Generate conclusion based on rule template
        if rule.rule_id == "transitivity_treats":
            if len(fact_set) >= 2:
                return InferredFact(
                    subject=fact_set[0].subject,
                    predicate="TREATS",
                    object=fact_set[1].object,
                    confidence=adjusted_confidence,
                    reasoning_path=[f"transitivity_depth_{depth}"],
                    source_facts=[
                        f"{f.subject} {f.predicate} {f.object}" for f in fact_set
                    ],
                    inferred_at=datetime.now(),
                    rule_applied=rule.rule_id,
                )

        elif rule.rule_id == "similarity_properties":
            if len(fact_set) >= 2:
                return InferredFact(
                    subject=fact_set[0].object,  # Similar compound
                    predicate=fact_set[1].predicate,  # Same property relation
                    object=fact_set[1].object,  # Property value
                    confidence=adjusted_confidence,
                    reasoning_path=[f"similarity_inference_depth_{depth}"],
                    source_facts=[
                        f"{f.subject} {f.predicate} {f.object}" for f in fact_set
                    ],
                    inferred_at=datetime.now(),
                    rule_applied=rule.rule_id,
                )

        return None

    def _identify_reasoning_goals(
        self, backend: GraphDatabaseBackend
    ) -> List[Dict[str, str]]:
        """Identify potential reasoning goals for backward chaining."""
        # Simplified goal identification
        goals = []

        # Look for incomplete relationships
        incomplete_treatments = [
            {"subject": "aspirin", "predicate": "TREATS", "object": "inflammation"},
            {"subject": "ibuprofen", "predicate": "TREATS", "object": "pain"},
        ]

        goals.extend(incomplete_treatments)
        return goals

    def _find_supporting_facts(
        self, goal: Dict[str, str], backend: GraphDatabaseBackend
    ) -> List[InferredFact]:
        """Find facts that support a reasoning goal."""
        # Simplified implementation
        supporting = []

        # Look for similar compounds with same property
        if goal["predicate"] == "TREATS":
            # Find similar drugs
            query = GraphQuery(
                query="find_nodes",
                parameters={"labels": ["Drug"], "properties": {"type": "nsaid"}},
                query_type="find_nodes",
            )
            result = backend.execute_query(query)

            if result.records:
                supporting.append(
                    InferredFact(
                        subject=result.records[0].get("name", ""),
                        predicate="SIMILAR_TO",
                        object=goal["subject"],
                        confidence=0.8,
                        reasoning_path=["similarity_lookup"],
                        source_facts=["drug_classification"],
                        inferred_at=datetime.now(),
                        rule_applied="similarity_lookup",
                    )
                )

        return supporting

    def _deduplicate_facts(self, facts: List[InferredFact]) -> List[InferredFact]:
        """Remove duplicate inferred facts."""
        seen = set()
        deduplicated = []

        for fact in facts:
            fact_key = (fact.subject, fact.predicate, fact.object)
            if fact_key not in seen:
                seen.add(fact_key)
                deduplicated.append(fact)
            else:
                # Update confidence if higher
                existing_idx = next(
                    i
                    for i, f in enumerate(deduplicated)
                    if (f.subject, f.predicate, f.object) == fact_key
                )
                if fact.confidence > deduplicated[existing_idx].confidence:
                    deduplicated[existing_idx] = fact

        return deduplicated

    def _check_contradictions(self, fact: InferredFact) -> bool:
        """Check if inferred fact contradicts existing knowledge."""
        # Simplified contradiction checking

        # Check for direct contradictions
        for existing_fact in self.inferred_facts:
            if (
                fact.subject == existing_fact.subject
                and fact.predicate == existing_fact.predicate
                and fact.object != existing_fact.object
                and existing_fact.confidence > 0.8
            ):
                return True  # Contradiction found

        return False


class SemanticQueryEngine:
    """Main semantic query and reasoning engine."""

    def __init__(
        self,
        backend: GraphDatabaseBackend,
        ontology: ChemicalOntology,
        schema: ICKGSchema,
    ):
        self.backend = backend
        self.ontology = ontology
        self.schema = schema

        # Initialize processors
        self.processors = {
            QueryLanguage.SPARQL: SPARQLProcessor(ontology),
            QueryLanguage.CYPHER: CypherSemanticProcessor(ontology),
            QueryLanguage.SEMANTIC_CYPHER: CypherSemanticProcessor(ontology),
            QueryLanguage.NATURAL_LANGUAGE: NaturalLanguageProcessor(ontology),
        }

        # Initialize reasoning engine
        self.reasoning_engine = ReasoningEngine(ontology, schema)

        # Query cache and statistics
        self.query_cache = {}
        self.query_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "reasoning_queries": 0,
            "average_execution_time": 0.0,
        }

        self.lock = threading.Lock()

    def execute_semantic_query(
        self,
        query: str,
        language: QueryLanguage,
        context: Optional[SemanticQueryContext] = None,
    ) -> GraphQueryResult:
        """Execute a semantic query with reasoning support."""
        if context is None:
            context = SemanticQueryContext(query_language=language)

        start_time = datetime.now()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, language, context)
            if context.cache_results and cache_key in self.query_cache:
                self.query_stats["cache_hits"] += 1
                cached_result = self.query_cache[cache_key]
                cached_result.metadata["cache_hit"] = True
                return cached_result

            # Get appropriate processor
            processor = self.processors.get(language)
            if not processor:
                raise ValueError(f"Unsupported query language: {language}")

            # Parse and validate query
            parsed_query = processor.parse_query(query, context)
            if not processor.validate_query(parsed_query, context):
                raise ValueError("Invalid query structure")

            # Optimize query
            optimized_query = processor.optimize_query(parsed_query, context)

            # Translate to backend query
            backend_query = processor.translate_to_backend_query(
                optimized_query, self.backend.backend_type.value
            )

            # Execute base query
            result = self.backend.execute_query(backend_query)

            # Apply reasoning if enabled
            if context.reasoning_enabled and context.include_inferred_facts:
                result = self._enhance_with_reasoning(result, context)

            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_statistics(execution_time, context.reasoning_enabled)

            # Cache result
            if context.cache_results:
                self.query_cache[cache_key] = result

            # Add semantic metadata
            result.metadata.update(
                {
                    "semantic_query": True,
                    "language": language.value,
                    "reasoning_applied": context.reasoning_enabled,
                    "execution_time": execution_time,
                    "cache_hit": False,
                }
            )

            return result

        except Exception as e:
            logging.error(f"Semantic query execution failed: {str(e)}")
            return GraphQueryResult(
                records=[],
                summary={"error": str(e), "query": query},
                execution_time=(datetime.now() - start_time).total_seconds(),
                query=GraphQuery(query, {}, language.value),
                metadata={"error": True, "semantic_query": True},
            )

    def ask_natural_language(
        self, question: str, context: Optional[SemanticQueryContext] = None
    ) -> GraphQueryResult:
        """Ask a question in natural language."""
        if context is None:
            context = SemanticQueryContext(
                query_language=QueryLanguage.NATURAL_LANGUAGE,
                reasoning_enabled=True,
                include_inferred_facts=True,
            )

        return self.execute_semantic_query(
            question, QueryLanguage.NATURAL_LANGUAGE, context
        )

    def infer_new_facts(
        self, strategy: ReasoningStrategy = ReasoningStrategy.FORWARD_CHAINING
    ) -> List[InferredFact]:
        """Infer new facts using the reasoning engine."""
        return self.reasoning_engine.infer_facts(self.backend, strategy)

    def add_reasoning_rule(self, rule: ReasoningRule) -> None:
        """Add a new reasoning rule."""
        self.reasoning_engine.add_rule(rule)

    def explain_result(self, result: GraphQueryResult) -> Dict[str, Any]:
        """Provide explanation for query results."""
        explanation = {
            "query_type": "semantic",
            "results_count": len(result.records),
            "execution_time": result.execution_time,
            "reasoning_applied": result.metadata.get("reasoning_applied", False),
            "cache_hit": result.metadata.get("cache_hit", False),
        }

        # Add reasoning explanations if applicable
        if result.metadata.get("inferred_facts"):
            explanation["inferred_facts"] = [
                self.reasoning_engine.explain_inference(fact)
                for fact in result.metadata["inferred_facts"]
            ]

        return explanation

    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query execution statistics."""
        with self.lock:
            stats = self.query_stats.copy()
            stats.update(self.reasoning_engine.get_reasoning_statistics())
            stats["cache_hit_rate"] = (
                stats["cache_hits"] / stats["total_queries"]
                if stats["total_queries"] > 0
                else 0
            )
            return stats

    def clear_cache(self) -> None:
        """Clear query cache."""
        with self.lock:
            self.query_cache.clear()

    def _generate_cache_key(
        self, query: str, language: QueryLanguage, context: SemanticQueryContext
    ) -> str:
        """Generate cache key for query."""
        key_components = [
            query,
            language.value,
            str(context.reasoning_enabled),
            str(context.reasoning_depth),
            str(context.confidence_threshold),
        ]
        return hash(tuple(key_components))

    def _enhance_with_reasoning(
        self, result: GraphQueryResult, context: SemanticQueryContext
    ) -> GraphQueryResult:
        """Enhance query results with inferred facts."""
        # Get relevant inferred facts
        inferred_facts = self.reasoning_engine.infer_facts(self.backend)

        # Filter relevant inferred facts based on query results
        relevant_facts = self._filter_relevant_facts(result, inferred_facts, context)

        if relevant_facts:
            # Add inferred facts to results
            enhanced_records = result.records.copy()

            for fact in relevant_facts:
                enhanced_records.append(
                    {
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        "confidence": fact.confidence,
                        "inferred": True,
                        "rule": fact.rule_applied,
                    }
                )

            # Update result
            result.records = enhanced_records
            result.metadata["inferred_facts"] = relevant_facts
            result.summary["enhanced_with_reasoning"] = True

        return result

    def _filter_relevant_facts(
        self,
        result: GraphQueryResult,
        inferred_facts: List[InferredFact],
        context: SemanticQueryContext,
    ) -> List[InferredFact]:
        """Filter inferred facts relevant to query results."""
        if not result.records:
            return []

        relevant = []
        result_entities = set()

        # Extract entities from query results
        for record in result.records:
            for key, value in record.items():
                if isinstance(value, str):
                    result_entities.add(value)

        # Find facts involving result entities
        for fact in inferred_facts:
            if fact.confidence >= context.confidence_threshold and (
                fact.subject in result_entities or fact.object in result_entities
            ):
                relevant.append(fact)

        return relevant[:10]  # Limit to top 10 relevant facts

    def _update_statistics(
        self, execution_time: float, reasoning_enabled: bool
    ) -> None:
        """Update query execution statistics."""
        with self.lock:
            self.query_stats["total_queries"] += 1
            if reasoning_enabled:
                self.query_stats["reasoning_queries"] += 1

            # Update average execution time
            total = self.query_stats["total_queries"]
            current_avg = self.query_stats["average_execution_time"]
            self.query_stats["average_execution_time"] = (
                current_avg * (total - 1) + execution_time
            ) / total

"""
Semantic Query Language Parsers

Advanced parsers for different semantic query languages including
SPARQL extensions, semantic Cypher, and domain-specific chemical query languages.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .ontology import ChemicalOntology, EntityType, RelationshipType


@dataclass
class QueryClause:
    """Represents a query clause with type and content."""

    clause_type: str
    content: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticPattern:
    """Represents a semantic pattern in queries."""

    subject: str
    predicate: str
    object: str
    optional: bool = False
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChemicalDSLParser:
    """Parser for Chemical Domain Specific Language queries."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.operators = {
            "FIND": self._parse_find_operation,
            "SHOW": self._parse_show_operation,
            "LIST": self._parse_list_operation,
            "COMPARE": self._parse_compare_operation,
            "ANALYZE": self._parse_analyze_operation,
            "RELATE": self._parse_relate_operation,
        }
        self.modifiers = {
            "WHERE": self._parse_where_clause,
            "WITH": self._parse_with_clause,
            "HAVING": self._parse_having_clause,
            "ORDER_BY": self._parse_order_clause,
            "LIMIT": self._parse_limit_clause,
        }

    def parse(self, query: str) -> Dict[str, Any]:
        """Parse Chemical DSL query."""
        # Normalize query
        normalized = self._normalize_query(query)

        # Extract main operation
        operation = self._extract_operation(normalized)

        # Extract clauses
        clauses = self._extract_clauses(normalized)

        # Parse each clause
        parsed_clauses = {}
        for clause_type, content in clauses.items():
            if clause_type in self.modifiers:
                parsed_clauses[clause_type] = self.modifiers[clause_type](content)
            else:
                parsed_clauses[clause_type] = content

        return {
            "query_type": "chemical_dsl",
            "operation": operation,
            "clauses": parsed_clauses,
            "original_query": query,
        }

    def _normalize_query(self, query: str) -> str:
        """Normalize query text."""
        # Convert to uppercase for keywords
        keywords = [
            "FIND",
            "SHOW",
            "LIST",
            "COMPARE",
            "ANALYZE",
            "RELATE",
            "WHERE",
            "WITH",
            "HAVING",
            "ORDER BY",
            "LIMIT",
        ]

        normalized = query
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            normalized = pattern.sub(keyword, normalized)

        return normalized.strip()

    def _extract_operation(self, query: str) -> Dict[str, Any]:
        """Extract main operation from query."""
        for op_name, parser in self.operators.items():
            if query.startswith(op_name):
                # Extract operation content
                content_match = re.match(
                    rf"{op_name}\s+(.+?)(?:\s+WHERE|\s+WITH|\s+ORDER BY|\s+LIMIT|$)",
                    query,
                )
                if content_match:
                    return parser(content_match.group(1).strip())

        return {"type": "unknown", "target": query}

    def _extract_clauses(self, query: str) -> Dict[str, str]:
        """Extract query clauses."""
        clauses = {}

        # Extract WHERE clause
        where_match = re.search(
            r"WHERE\s+(.+?)(?:\s+WITH|\s+HAVING|\s+ORDER BY|\s+LIMIT|$)", query
        )
        if where_match:
            clauses["WHERE"] = where_match.group(1).strip()

        # Extract WITH clause
        with_match = re.search(
            r"WITH\s+(.+?)(?:\s+HAVING|\s+ORDER BY|\s+LIMIT|$)", query
        )
        if with_match:
            clauses["WITH"] = with_match.group(1).strip()

        # Extract HAVING clause
        having_match = re.search(r"HAVING\s+(.+?)(?:\s+ORDER BY|\s+LIMIT|$)", query)
        if having_match:
            clauses["HAVING"] = having_match.group(1).strip()

        # Extract ORDER BY clause
        order_match = re.search(r"ORDER BY\s+(.+?)(?:\s+LIMIT|$)", query)
        if order_match:
            clauses["ORDER_BY"] = order_match.group(1).strip()

        # Extract LIMIT clause
        limit_match = re.search(r"LIMIT\s+(\d+)", query)
        if limit_match:
            clauses["LIMIT"] = limit_match.group(1).strip()

        return clauses

    def _parse_find_operation(self, content: str) -> Dict[str, Any]:
        """Parse FIND operation."""
        # FIND compounds/drugs/molecules/reactions
        entity_types = []

        if "compound" in content.lower():
            entity_types.append(EntityType.SMALL_MOLECULE)
        if "drug" in content.lower():
            entity_types.append(EntityType.DRUG)
        if "molecule" in content.lower():
            entity_types.append(EntityType.SMALL_MOLECULE)
        if "reaction" in content.lower():
            entity_types.append(EntityType.REACTION)
        if "protein" in content.lower():
            entity_types.append(EntityType.PROTEIN)

        return {"type": "find", "entity_types": entity_types, "target": content}

    def _parse_show_operation(self, content: str) -> Dict[str, Any]:
        """Parse SHOW operation."""
        return {"type": "show", "target": content, "display_format": "detailed"}

    def _parse_list_operation(self, content: str) -> Dict[str, Any]:
        """Parse LIST operation."""
        return {"type": "list", "target": content, "display_format": "list"}

    def _parse_compare_operation(self, content: str) -> Dict[str, Any]:
        """Parse COMPARE operation."""
        # COMPARE entity1 AND entity2
        entities = re.split(r"\s+AND\s+", content, flags=re.IGNORECASE)
        return {
            "type": "compare",
            "entities": [e.strip() for e in entities],
            "comparison_type": "full",
        }

    def _parse_analyze_operation(self, content: str) -> Dict[str, Any]:
        """Parse ANALYZE operation."""
        return {"type": "analyze", "target": content, "analysis_type": "comprehensive"}

    def _parse_relate_operation(self, content: str) -> Dict[str, Any]:
        """Parse RELATE operation."""
        # RELATE entity1 TO entity2
        match = re.match(r"(.+?)\s+TO\s+(.+)", content, re.IGNORECASE)
        if match:
            return {
                "type": "relate",
                "source_entity": match.group(1).strip(),
                "target_entity": match.group(2).strip(),
            }

        return {"type": "relate", "entity": content}

    def _parse_where_clause(self, content: str) -> Dict[str, Any]:
        """Parse WHERE clause constraints."""
        constraints = []

        # Split by AND/OR
        parts = re.split(r"\s+(?:AND|OR)\s+", content, flags=re.IGNORECASE)

        for part in parts:
            constraint = self._parse_constraint(part.strip())
            if constraint:
                constraints.append(constraint)

        return {"constraints": constraints}

    def _parse_with_clause(self, content: str) -> Dict[str, Any]:
        """Parse WITH clause for additional data."""
        properties = []

        # Split by comma
        parts = [p.strip() for p in content.split(",")]

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                properties.append({"property": key.strip(), "value": value.strip()})
            else:
                properties.append({"property": part, "value": None})

        return {"properties": properties}

    def _parse_having_clause(self, content: str) -> Dict[str, Any]:
        """Parse HAVING clause for aggregate constraints."""
        return self._parse_where_clause(content)  # Similar parsing logic

    def _parse_order_clause(self, content: str) -> Dict[str, Any]:
        """Parse ORDER BY clause."""
        parts = content.split(",")
        order_specs = []

        for part in parts:
            part = part.strip()
            if part.upper().endswith(" DESC"):
                order_specs.append({"field": part[:-5].strip(), "direction": "DESC"})
            elif part.upper().endswith(" ASC"):
                order_specs.append({"field": part[:-4].strip(), "direction": "ASC"})
            else:
                order_specs.append({"field": part, "direction": "ASC"})

        return {"order_by": order_specs}

    def _parse_limit_clause(self, content: str) -> Dict[str, Any]:
        """Parse LIMIT clause."""
        try:
            limit = int(content)
            return {"limit": limit}
        except ValueError:
            return {"limit": 10}  # Default

    def _parse_constraint(self, constraint: str) -> Optional[Dict[str, Any]]:
        """Parse individual constraint."""
        # Handle different constraint types

        # Equality: property = value
        eq_match = re.match(r"(\w+)\s*=\s*(.+)", constraint)
        if eq_match:
            return {
                "type": "equality",
                "property": eq_match.group(1),
                "value": eq_match.group(2).strip("'\""),
            }

        # Range: property BETWEEN min AND max
        range_match = re.match(
            r"(\w+)\s+BETWEEN\s+(.+?)\s+AND\s+(.+)", constraint, re.IGNORECASE
        )
        if range_match:
            return {
                "type": "range",
                "property": range_match.group(1),
                "min_value": range_match.group(2),
                "max_value": range_match.group(3),
            }

        # Greater than: property > value
        gt_match = re.match(r"(\w+)\s*>\s*(.+)", constraint)
        if gt_match:
            return {
                "type": "greater_than",
                "property": gt_match.group(1),
                "value": gt_match.group(2),
            }

        # Less than: property < value
        lt_match = re.match(r"(\w+)\s*<\s*(.+)", constraint)
        if lt_match:
            return {
                "type": "less_than",
                "property": lt_match.group(1),
                "value": lt_match.group(2),
            }

        # Contains: property CONTAINS value
        contains_match = re.match(r"(\w+)\s+CONTAINS\s+(.+)", constraint, re.IGNORECASE)
        if contains_match:
            return {
                "type": "contains",
                "property": contains_match.group(1),
                "value": contains_match.group(2).strip("'\""),
            }

        return None


class AdvancedSPARQLParser:
    """Advanced SPARQL parser with chemical domain extensions."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.chemical_functions = {
            "SIMILARITY": self._parse_similarity_function,
            "SUBSTRUCTURE": self._parse_substructure_function,
            "MOLECULAR_WEIGHT": self._parse_molecular_weight_function,
            "TANIMOTO": self._parse_tanimoto_function,
        }

    def parse_extended_sparql(self, query: str) -> Dict[str, Any]:
        """Parse SPARQL with chemical extensions."""
        # Basic SPARQL parsing
        basic_parsed = self._parse_basic_sparql(query)

        # Extract chemical functions
        chemical_functions = self._extract_chemical_functions(query)

        # Extract chemical constraints
        chemical_constraints = self._extract_chemical_constraints(query)

        # Enhanced patterns with chemical semantics
        enhanced_patterns = self._enhance_patterns_with_chemistry(
            basic_parsed.get("patterns", [])
        )

        return {
            **basic_parsed,
            "chemical_functions": chemical_functions,
            "chemical_constraints": chemical_constraints,
            "enhanced_patterns": enhanced_patterns,
        }

    def _parse_basic_sparql(self, query: str) -> Dict[str, Any]:
        """Parse basic SPARQL structure."""
        # Extract prefixes
        prefixes = {}
        prefix_pattern = r"PREFIX\s+(\w+):\s*<([^>]+)>"
        for match in re.finditer(prefix_pattern, query, re.IGNORECASE):
            prefixes[match.group(1)] = match.group(2)

        # Extract SELECT variables
        select_match = re.search(
            r"SELECT\s+(.+?)\s+WHERE", query, re.IGNORECASE | re.DOTALL
        )
        select_vars = []
        if select_match:
            vars_str = select_match.group(1).strip()
            if vars_str != "*":
                select_vars = re.findall(r"\?(\w+)", vars_str)

        # Extract WHERE patterns
        where_match = re.search(r"WHERE\s*\{(.+)\}", query, re.IGNORECASE | re.DOTALL)
        patterns = []
        if where_match:
            where_content = where_match.group(1)
            # Simple triple pattern extraction
            triple_pattern = r"([^\s]+)\s+([^\s]+)\s+([^\s\.]+)"
            for match in re.finditer(triple_pattern, where_content):
                patterns.append(
                    {
                        "subject": match.group(1).strip(),
                        "predicate": match.group(2).strip(),
                        "object": match.group(3).strip(),
                    }
                )

        return {
            "query_type": "sparql",
            "prefixes": prefixes,
            "select_variables": select_vars,
            "patterns": patterns,
        }

    def _extract_chemical_functions(self, query: str) -> List[Dict[str, Any]]:
        """Extract chemical-specific functions from SPARQL."""
        functions = []

        for func_name, parser in self.chemical_functions.items():
            pattern = rf"{func_name}\s*\(([^)]+)\)"
            for match in re.finditer(pattern, query, re.IGNORECASE):
                func_result = parser(match.group(1))
                if func_result:
                    functions.append(func_result)

        return functions

    def _extract_chemical_constraints(self, query: str) -> List[Dict[str, Any]]:
        """Extract chemical domain constraints."""
        constraints = []

        # Molecular weight constraints
        mw_pattern = r"molecular_weight\s*([<>=]+)\s*(\d+(?:\.\d+)?)"
        for match in re.finditer(mw_pattern, query, re.IGNORECASE):
            constraints.append(
                {
                    "type": "molecular_weight",
                    "operator": match.group(1),
                    "value": float(match.group(2)),
                }
            )

        # SMILES pattern constraints
        smiles_pattern = r'smiles\s*=\s*["\']([^"\']+)["\']'
        for match in re.finditer(smiles_pattern, query, re.IGNORECASE):
            constraints.append({"type": "smiles", "value": match.group(1)})

        # Chemical class constraints
        class_pattern = r'chemical_class\s*=\s*["\']([^"\']+)["\']'
        for match in re.finditer(class_pattern, query, re.IGNORECASE):
            constraints.append({"type": "chemical_class", "value": match.group(1)})

        return constraints

    def _enhance_patterns_with_chemistry(
        self, patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance SPARQL patterns with chemical semantics."""
        enhanced = []

        for pattern in patterns:
            enhanced_pattern = pattern.copy()

            # Add chemical context
            predicate = pattern["predicate"]

            if "treats" in predicate.lower():
                enhanced_pattern["chemical_context"] = "therapeutic_relationship"
            elif "causes" in predicate.lower():
                enhanced_pattern["chemical_context"] = "causal_relationship"
            elif "similar" in predicate.lower():
                enhanced_pattern["chemical_context"] = "similarity_relationship"
            elif "substructure" in predicate.lower():
                enhanced_pattern["chemical_context"] = "structural_relationship"

            # Add ontology mappings
            if "ontology_uri" not in enhanced_pattern:
                enhanced_pattern["ontology_mappings"] = self._map_to_ontology(pattern)

            enhanced.append(enhanced_pattern)

        return enhanced

    def _parse_similarity_function(self, args: str) -> Dict[str, Any]:
        """Parse chemical similarity function."""
        args_list = [arg.strip() for arg in args.split(",")]

        if len(args_list) >= 2:
            return {
                "function": "similarity",
                "molecule1": args_list[0],
                "molecule2": args_list[1],
                "threshold": float(args_list[2]) if len(args_list) > 2 else 0.7,
            }

        return None

    def _parse_substructure_function(self, args: str) -> Dict[str, Any]:
        """Parse substructure search function."""
        args_list = [arg.strip() for arg in args.split(",")]

        if len(args_list) >= 2:
            return {
                "function": "substructure",
                "molecule": args_list[0],
                "substructure": args_list[1].strip("'\""),
            }

        return None

    def _parse_molecular_weight_function(self, args: str) -> Dict[str, Any]:
        """Parse molecular weight function."""
        return {"function": "molecular_weight", "molecule": args.strip()}

    def _parse_tanimoto_function(self, args: str) -> Dict[str, Any]:
        """Parse Tanimoto similarity function."""
        args_list = [arg.strip() for arg in args.split(",")]

        if len(args_list) >= 2:
            return {
                "function": "tanimoto",
                "molecule1": args_list[0],
                "molecule2": args_list[1],
                "fingerprint_type": args_list[2] if len(args_list) > 2 else "morgan",
            }

        return None

    def _map_to_ontology(self, pattern: Dict[str, Any]) -> Dict[str, str]:
        """Map pattern elements to ontology URIs."""
        mappings = {}

        # Map predicate to ontology
        predicate = pattern["predicate"]
        if predicate.startswith("cao:"):
            mappings[
                "predicate_uri"
            ] = f"http://chemi-accel.org/ontology/{predicate[4:]}"

        # Map subject/object if they are known entities
        for role in ["subject", "object"]:
            element = pattern[role]
            if not element.startswith("?"):  # Not a variable
                # Look up in ontology
                entities = self.ontology.find_entity(element)
                if entities:
                    mappings[f"{role}_uri"] = entities[0].uri

        return mappings


class SemanticCypherExtensions:
    """Extensions for semantic Cypher queries."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.semantic_operators = {
            "SEMANTICALLY_SIMILAR": self._expand_semantic_similarity,
            "CHEMICALLY_RELATED": self._expand_chemical_relationship,
            "STRUCTURALLY_CONTAINS": self._expand_structural_containment,
            "FUNCTIONALLY_EQUIVALENT": self._expand_functional_equivalence,
        }

    def parse_semantic_cypher(self, query: str) -> Dict[str, Any]:
        """Parse Cypher with semantic extensions."""
        # Extract basic Cypher structure
        basic_structure = self._parse_basic_cypher(query)

        # Extract semantic operators
        semantic_ops = self._extract_semantic_operators(query)

        # Generate expanded patterns
        expanded_patterns = []
        for op in semantic_ops:
            if op["operator"] in self.semantic_operators:
                expanded = self.semantic_operators[op["operator"]](op)
                expanded_patterns.extend(expanded)

        return {
            **basic_structure,
            "semantic_operators": semantic_ops,
            "expanded_patterns": expanded_patterns,
            "query_type": "semantic_cypher",
        }

    def _parse_basic_cypher(self, query: str) -> Dict[str, Any]:
        """Parse basic Cypher structure."""
        # Extract MATCH patterns
        match_patterns = []
        match_pattern = r"MATCH\s+(.+?)(?:\s+WHERE|\s+RETURN|$)"
        for match in re.finditer(match_pattern, query, re.IGNORECASE):
            match_patterns.append(match.group(1).strip())

        # Extract WHERE conditions
        where_conditions = []
        where_match = re.search(
            r"WHERE\s+(.+?)(?:\s+RETURN|$)", query, re.IGNORECASE | re.DOTALL
        )
        if where_match:
            where_conditions.append(where_match.group(1).strip())

        # Extract RETURN clause
        return_match = re.search(r"RETURN\s+(.+)$", query, re.IGNORECASE)
        return_clause = return_match.group(1).strip() if return_match else ""

        return {
            "match_patterns": match_patterns,
            "where_conditions": where_conditions,
            "return_clause": return_clause,
        }

    def _extract_semantic_operators(self, query: str) -> List[Dict[str, Any]]:
        """Extract semantic operators from query."""
        operators = []

        for op_name in self.semantic_operators.keys():
            if op_name in query:
                # Extract context around the operator
                pattern = rf"(.+?){op_name}(.+?)(?:\s|$)"
                match = re.search(pattern, query)
                if match:
                    operators.append(
                        {
                            "operator": op_name,
                            "left_context": match.group(1).strip(),
                            "right_context": match.group(2).strip(),
                            "full_match": match.group(0),
                        }
                    )

        return operators

    def _expand_semantic_similarity(self, op: Dict[str, Any]) -> List[str]:
        """Expand semantic similarity operator."""
        return [
            "MATCH (a)-[:SIMILAR_TO]-(b)",
            "WHERE a.similarity_score >= 0.8",
            "OPTIONAL MATCH (a)-[:HAS_FINGERPRINT]->(fp1)",
            "OPTIONAL MATCH (b)-[:HAS_FINGERPRINT]->(fp2)",
            "WHERE gds.similarity.cosine(fp1.vector, fp2.vector) > 0.7",
        ]

    def _expand_chemical_relationship(self, op: Dict[str, Any]) -> List[str]:
        """Expand chemical relationship operator."""
        return [
            "MATCH (a)-[r:TREATS|CAUSES|INTERACTS_WITH|METABOLIZES]-(b)",
            "WHERE r.confidence >= 0.6",
            "OPTIONAL MATCH (a)-[:IS_A]->(class_a)",
            "OPTIONAL MATCH (b)-[:IS_A]->(class_b)",
            "WHERE class_a.type IN ['Drug', 'Chemical'] AND class_b.type IN ['Condition', 'Protein']",
        ]

    def _expand_structural_containment(self, op: Dict[str, Any]) -> List[str]:
        """Expand structural containment operator."""
        return [
            "MATCH (parent)-[:CONTAINS_SUBSTRUCTURE]->(substructure)",
            "WHERE substructure.smarts IS NOT NULL",
            "OPTIONAL MATCH (parent)-[:HAS_STRUCTURE]->(structure)",
            "WHERE structure.contains_pattern(substructure.smarts)",
        ]

    def _expand_functional_equivalence(self, op: Dict[str, Any]) -> List[str]:
        """Expand functional equivalence operator."""
        return [
            "MATCH (a)-[:HAS_FUNCTION]->(func)<-[:HAS_FUNCTION]-(b)",
            "WHERE func.mechanism = func.mechanism",
            "OPTIONAL MATCH (a)-[:TARGETS]->(target)<-[:TARGETS]-(b)",
            "WHERE target.type IN ['Protein', 'Pathway', 'Receptor']",
        ]


class QueryOptimizationEngine:
    """Advanced query optimization for semantic queries."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.optimization_rules = [
            self._optimize_pattern_order,
            self._push_down_filters,
            self._eliminate_redundant_patterns,
            self._add_index_hints,
            self._parallelize_independent_patterns,
        ]

    def optimize_query(
        self, parsed_query: Dict[str, Any], backend_type: str
    ) -> Dict[str, Any]:
        """Apply optimization rules to parsed query."""
        optimized = parsed_query.copy()

        # Apply optimization rules in sequence
        for rule in self.optimization_rules:
            optimized = rule(optimized, backend_type)

        # Add optimization metadata
        optimized["optimization_applied"] = True
        optimized["optimization_rules"] = len(self.optimization_rules)

        return optimized

    def _optimize_pattern_order(
        self, query: Dict[str, Any], backend_type: str
    ) -> Dict[str, Any]:
        """Optimize pattern execution order."""
        if "patterns" not in query:
            return query

        patterns = query["patterns"]

        # Score patterns by selectivity
        scored_patterns = []
        for pattern in patterns:
            selectivity = self._estimate_pattern_selectivity(pattern)
            scored_patterns.append((selectivity, pattern))

        # Sort by selectivity (most selective first)
        scored_patterns.sort(key=lambda x: x[0])

        optimized = query.copy()
        optimized["patterns"] = [pattern for _, pattern in scored_patterns]

        return optimized

    def _push_down_filters(
        self, query: Dict[str, Any], backend_type: str
    ) -> Dict[str, Any]:
        """Push filter conditions closer to relevant patterns."""
        # Simplified implementation
        optimized = query.copy()

        if "where_conditions" in query and "patterns" in query:
            # Analyze filters and associate with patterns
            filter_analysis = self._analyze_filter_conditions(
                query["where_conditions"], query["patterns"]
            )

            optimized["filter_analysis"] = filter_analysis

        return optimized

    def _eliminate_redundant_patterns(
        self, query: Dict[str, Any], backend_type: str
    ) -> Dict[str, Any]:
        """Remove redundant query patterns."""
        if "patterns" not in query:
            return query

        patterns = query["patterns"]
        unique_patterns = []
        seen_patterns = set()

        for pattern in patterns:
            # Create pattern signature
            signature = (
                pattern.get("subject", ""),
                pattern.get("predicate", ""),
                pattern.get("object", ""),
            )

            if signature not in seen_patterns:
                unique_patterns.append(pattern)
                seen_patterns.add(signature)

        optimized = query.copy()
        optimized["patterns"] = unique_patterns

        return optimized

    def _add_index_hints(
        self, query: Dict[str, Any], backend_type: str
    ) -> Dict[str, Any]:
        """Add index usage hints for better performance."""
        optimized = query.copy()

        if backend_type == "neo4j":
            # Add Neo4j index hints
            index_hints = []

            for pattern in query.get("patterns", []):
                if pattern.get("predicate") == "name":
                    index_hints.append("USING INDEX entity:name")
                elif pattern.get("predicate") == "molecular_weight":
                    index_hints.append("USING INDEX compound:molecular_weight")

            if index_hints:
                optimized["index_hints"] = index_hints

        return optimized

    def _parallelize_independent_patterns(
        self, query: Dict[str, Any], backend_type: str
    ) -> Dict[str, Any]:
        """Identify patterns that can be executed in parallel."""
        if "patterns" not in query:
            return query

        # Analyze pattern dependencies
        patterns = query["patterns"]
        dependencies = self._analyze_pattern_dependencies(patterns)

        # Group independent patterns
        parallel_groups = self._group_independent_patterns(patterns, dependencies)

        optimized = query.copy()
        optimized["parallel_groups"] = parallel_groups

        return optimized

    def _estimate_pattern_selectivity(self, pattern: Dict[str, Any]) -> float:
        """Estimate pattern selectivity (lower = more selective)."""
        selectivity = 1.0

        # Specific entity references are more selective
        if pattern.get("subject", "").startswith("?"):
            selectivity *= 0.5
        if pattern.get("object", "").startswith("?"):
            selectivity *= 0.5

        # Certain predicates are more selective
        predicate = pattern.get("predicate", "")
        if "name" in predicate.lower():
            selectivity *= 0.1  # Names are highly selective
        elif "type" in predicate.lower():
            selectivity *= 0.3  # Types are moderately selective
        elif "similar" in predicate.lower():
            selectivity *= 0.7  # Similarity is less selective

        return selectivity

    def _analyze_filter_conditions(
        self, conditions: List[str], patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze which filters can be pushed to which patterns."""
        analysis = {"pushdown_opportunities": []}

        for condition in conditions:
            # Extract variables from condition
            variables = re.findall(r"\?(\w+)", condition)

            # Find patterns that use these variables
            relevant_patterns = []
            for i, pattern in enumerate(patterns):
                pattern_vars = re.findall(r"\?(\w+)", str(pattern))
                if any(var in pattern_vars for var in variables):
                    relevant_patterns.append(i)

            if relevant_patterns:
                analysis["pushdown_opportunities"].append(
                    {
                        "condition": condition,
                        "variables": variables,
                        "relevant_patterns": relevant_patterns,
                    }
                )

        return analysis

    def _analyze_pattern_dependencies(
        self, patterns: List[Dict[str, Any]]
    ) -> Dict[int, List[int]]:
        """Analyze dependencies between patterns."""
        dependencies = defaultdict(list)

        for i, pattern1 in enumerate(patterns):
            vars1 = re.findall(r"\?(\w+)", str(pattern1))

            for j, pattern2 in enumerate(patterns):
                if i != j:
                    vars2 = re.findall(r"\?(\w+)", str(pattern2))

                    # If patterns share variables, there's a dependency
                    if any(var in vars2 for var in vars1):
                        dependencies[i].append(j)

        return dict(dependencies)

    def _group_independent_patterns(
        self, patterns: List[Dict[str, Any]], dependencies: Dict[int, List[int]]
    ) -> List[List[int]]:
        """Group patterns that can be executed independently."""
        groups = []
        processed = set()

        for i in range(len(patterns)):
            if i in processed:
                continue

            # Start a new group
            group = [i]
            processed.add(i)

            # Add patterns that don't depend on this group
            for j in range(i + 1, len(patterns)):
                if j in processed:
                    continue

                # Check if j depends on any pattern in current group
                has_dependency = any(dep in group for dep in dependencies.get(j, []))

                if not has_dependency:
                    group.append(j)
                    processed.add(j)

            groups.append(group)

        return groups

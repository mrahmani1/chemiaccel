"""
Knowledge Graph Integration for Reaction Intelligence

Integrates with the Intelligent Chemical Knowledge Graph (IC-KG) to leverage
known catalyst performance and reaction conditions from literature.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..core.exceptions import ChemiAccelError
from ..knowledge_graph.query_engine import QueryEngine
from .intelligence import CatalystSuggestion, ReactionConditions

logger = logging.getLogger(__name__)


class CatalystKnowledgeRetriever:
    """Retrieve catalyst performance data from knowledge graph."""

    def __init__(self, kg_engine: QueryEngine):
        self.kg_engine = kg_engine

    def query_catalyst_performance(
        self, reaction_type: str, substrate_classes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query catalyst performance for specific reaction types."""
        try:
            # SPARQL query to find catalysts and their performance
            query = f"""
            PREFIX chemi: <http://chemiknowledgegraph.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?catalyst ?catalystName ?yield ?selectivity ?conditions ?reference
            WHERE {{
                ?reaction a chemi:ChemicalReaction ;
                         chemi:hasReactionType "{reaction_type}" ;
                         chemi:usesCatalyst ?catalyst ;
                         chemi:hasYield ?yield ;
                         chemi:hasConditions ?conditions ;
                         chemi:hasReference ?reference .

                ?catalyst rdfs:label ?catalystName .

                OPTIONAL {{ ?reaction chemi:hasSelectivity ?selectivity }}

                {self._build_substrate_filter(substrate_classes) if substrate_classes else ""}
            }}
            ORDER BY DESC(?yield)
            LIMIT 50
            """

            results = self.kg_engine.execute_sparql(query)
            return self._process_catalyst_results(results)

        except Exception as e:
            logger.error(f"Error querying catalyst performance: {e}")
            return []

    def query_reaction_conditions(
        self, reactant_smiles: List[str], product_smiles: List[str]
    ) -> List[Dict[str, Any]]:
        """Query optimal conditions for similar reactions."""
        try:
            # Find similar reactions in the knowledge graph
            query = """
            PREFIX chemi: <http://chemiknowledgegraph.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?reaction ?conditions ?temperature ?solvent ?catalyst ?yield ?reference
            WHERE {
                ?reaction a chemi:ChemicalReaction ;
                         chemi:hasConditions ?conditions ;
                         chemi:hasYield ?yield ;
                         chemi:hasReference ?reference .

                ?conditions chemi:hasTemperature ?temperature ;
                           chemi:hasSolvent ?solvent ;
                           chemi:hasCatalyst ?catalyst .

                # Similarity matching would be implemented here
                # For now, we'll match by reaction type
            }
            ORDER BY DESC(?yield)
            LIMIT 20
            """

            results = self.kg_engine.execute_sparql(query)
            return self._process_condition_results(results)

        except Exception as e:
            logger.error(f"Error querying reaction conditions: {e}")
            return []

    def query_catalyst_literature_precedents(
        self, catalyst_name: str, reaction_type: Optional[str] = None
    ) -> int:
        """Count literature precedents for a catalyst."""
        try:
            reaction_filter = (
                f'chemi:hasReactionType "{reaction_type}" ;' if reaction_type else ""
            )

            query = f"""
            PREFIX chemi: <http://chemiknowledgegraph.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT (COUNT(DISTINCT ?reaction) as ?count)
            WHERE {{
                ?reaction a chemi:ChemicalReaction ;
                         {reaction_filter}
                         chemi:usesCatalyst ?catalyst .

                ?catalyst rdfs:label "{catalyst_name}" .
            }}
            """

            results = self.kg_engine.execute_sparql(query)
            if results and len(results) > 0:
                return int(results[0].get("count", 0))
            return 0

        except Exception as e:
            logger.error(f"Error querying catalyst precedents: {e}")
            return 0

    def query_solvent_compatibility(self, catalyst_name: str) -> List[str]:
        """Query compatible solvents for a catalyst."""
        try:
            query = f"""
            PREFIX chemi: <http://chemiknowledgegraph.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?solvent
            WHERE {{
                ?reaction a chemi:ChemicalReaction ;
                         chemi:usesCatalyst ?catalyst ;
                         chemi:hasConditions ?conditions .

                ?catalyst rdfs:label "{catalyst_name}" .
                ?conditions chemi:hasSolvent ?solvent .

                # Filter for successful reactions (high yield)
                ?reaction chemi:hasYield ?yield .
                FILTER(?yield > 0.7)
            }}
            """

            results = self.kg_engine.execute_sparql(query)
            return [result["solvent"] for result in results if "solvent" in result]

        except Exception as e:
            logger.error(f"Error querying solvent compatibility: {e}")
            return []

    def _build_substrate_filter(self, substrate_classes: List[str]) -> str:
        """Build SPARQL filter for substrate classes."""
        filters = []
        for substrate_class in substrate_classes:
            filters.append(f'?reaction chemi:hasSubstrateClass "{substrate_class}"')

        if filters:
            return "FILTER (" + " || ".join(filters) + ")"
        return ""

    def _process_catalyst_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process raw SPARQL results for catalyst performance."""
        processed = []

        for result in results:
            try:
                processed_result = {
                    "catalyst_name": result.get("catalystName", ""),
                    "catalyst_uri": result.get("catalyst", ""),
                    "yield": float(result.get("yield", 0)),
                    "selectivity": float(result.get("selectivity", 0))
                    if result.get("selectivity")
                    else None,
                    "conditions": result.get("conditions", ""),
                    "reference": result.get("reference", ""),
                }
                processed.append(processed_result)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing catalyst result: {e}")
                continue

        return processed

    def _process_condition_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process raw SPARQL results for reaction conditions."""
        processed = []

        for result in results:
            try:
                processed_result = {
                    "reaction_uri": result.get("reaction", ""),
                    "temperature": float(result.get("temperature", 0))
                    if result.get("temperature")
                    else None,
                    "solvent": result.get("solvent", ""),
                    "catalyst": result.get("catalyst", ""),
                    "yield": float(result.get("yield", 0)),
                    "reference": result.get("reference", ""),
                }
                processed.append(processed_result)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing condition result: {e}")
                continue

        return processed


class LiteratureBasedOptimizer:
    """Optimize reaction conditions based on literature data."""

    def __init__(self, kg_retriever: CatalystKnowledgeRetriever):
        self.kg_retriever = kg_retriever

    def optimize_catalyst_selection(
        self,
        reaction_type: str,
        substrate_classes: Optional[List[str]] = None,
        max_suggestions: int = 5,
    ) -> List[CatalystSuggestion]:
        """Optimize catalyst selection based on literature performance."""
        # Query literature data
        performance_data = self.kg_retriever.query_catalyst_performance(
            reaction_type, substrate_classes
        )

        if not performance_data:
            logger.warning("No literature data found for catalyst optimization")
            return []

        # Group by catalyst and calculate statistics
        catalyst_stats = {}
        for entry in performance_data:
            catalyst_name = entry["catalyst_name"]
            if catalyst_name not in catalyst_stats:
                catalyst_stats[catalyst_name] = {
                    "yields": [],
                    "selectivities": [],
                    "references": set(),
                }

            catalyst_stats[catalyst_name]["yields"].append(entry["yield"])
            if entry["selectivity"] is not None:
                catalyst_stats[catalyst_name]["selectivities"].append(
                    entry["selectivity"]
                )
            catalyst_stats[catalyst_name]["references"].add(entry["reference"])

        # Create suggestions based on statistics
        suggestions = []
        for catalyst_name, stats in catalyst_stats.items():
            if not stats["yields"]:
                continue

            mean_yield = sum(stats["yields"]) / len(stats["yields"])
            mean_selectivity = (
                sum(stats["selectivities"]) / len(stats["selectivities"])
                if stats["selectivities"]
                else None
            )

            # Get additional catalyst information
            precedents = len(stats["references"])
            compatible_solvents = self.kg_retriever.query_solvent_compatibility(
                catalyst_name
            )

            # Calculate confidence score
            confidence = self._calculate_confidence_score(stats, precedents)

            suggestion = CatalystSuggestion(
                catalyst_name=catalyst_name,
                predicted_yield=mean_yield,
                predicted_selectivity=mean_selectivity,
                confidence_score=confidence,
                literature_precedents=precedents,
                compatible_solvents=compatible_solvents,
                notes=f"Based on {precedents} literature precedents",
            )

            suggestions.append(suggestion)

        # Sort by confidence score and return top suggestions
        suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
        return suggestions[:max_suggestions]

    def optimize_reaction_conditions(
        self, reactant_smiles: List[str], product_smiles: List[str]
    ) -> ReactionConditions:
        """Optimize reaction conditions based on similar literature reactions."""
        # Query similar reactions
        similar_reactions = self.kg_retriever.query_reaction_conditions(
            reactant_smiles, product_smiles
        )

        if not similar_reactions:
            logger.warning("No similar reactions found in literature")
            return ReactionConditions()

        # Analyze optimal conditions
        high_yield_reactions = [r for r in similar_reactions if r["yield"] > 0.7]

        if not high_yield_reactions:
            high_yield_reactions = similar_reactions[
                :5
            ]  # Take top 5 if none are high yield

        # Extract common conditions
        temperatures = [
            r["temperature"] for r in high_yield_reactions if r["temperature"]
        ]
        solvents = [r["solvent"] for r in high_yield_reactions if r["solvent"]]
        catalysts = [r["catalyst"] for r in high_yield_reactions if r["catalyst"]]

        # Calculate optimal conditions
        optimal_conditions = ReactionConditions()

        if temperatures:
            optimal_conditions.temperature = sum(temperatures) / len(temperatures)

        if solvents:
            # Most common solvent
            solvent_counts = {}
            for solvent in solvents:
                solvent_counts[solvent] = solvent_counts.get(solvent, 0) + 1
            optimal_conditions.solvent = max(solvent_counts, key=solvent_counts.get)

        if catalysts:
            # Most successful catalyst
            catalyst_yields = {}
            for reaction in high_yield_reactions:
                if reaction["catalyst"]:
                    catalyst = reaction["catalyst"]
                    if catalyst not in catalyst_yields:
                        catalyst_yields[catalyst] = []
                    catalyst_yields[catalyst].append(reaction["yield"])

            if catalyst_yields:
                best_catalyst = max(
                    catalyst_yields,
                    key=lambda c: sum(catalyst_yields[c]) / len(catalyst_yields[c]),
                )
                optimal_conditions.catalyst = best_catalyst

        return optimal_conditions

    def _calculate_confidence_score(
        self, stats: Dict[str, Any], precedents: int
    ) -> float:
        """Calculate confidence score for catalyst suggestion."""
        if not stats["yields"]:
            return 0.0

        # Base score from mean yield
        mean_yield = sum(stats["yields"]) / len(stats["yields"])
        base_score = mean_yield

        # Bonus for consistency (low variance)
        if len(stats["yields"]) > 1:
            variance = sum((y - mean_yield) ** 2 for y in stats["yields"]) / len(
                stats["yields"]
            )
            consistency_bonus = max(0, 0.2 - variance)
            base_score += consistency_bonus

        # Bonus for number of precedents
        precedent_bonus = min(precedents / 100, 0.3)
        base_score += precedent_bonus

        return min(base_score, 1.0)


class KnowledgeGraphEnhancedSuggestionEngine:
    """Enhanced catalyst suggestion engine with knowledge graph integration."""

    def __init__(self, kg_engine: QueryEngine):
        self.kg_retriever = CatalystKnowledgeRetriever(kg_engine)
        self.literature_optimizer = LiteratureBasedOptimizer(self.kg_retriever)

    def suggest_catalysts_with_literature(
        self,
        reaction_type: str,
        substrate_classes: Optional[List[str]] = None,
        max_suggestions: int = 5,
    ) -> List[CatalystSuggestion]:
        """Suggest catalysts enhanced with literature data."""
        return self.literature_optimizer.optimize_catalyst_selection(
            reaction_type, substrate_classes, max_suggestions
        )

    def suggest_conditions_with_literature(
        self, reactant_smiles: List[str], product_smiles: List[str]
    ) -> ReactionConditions:
        """Suggest conditions enhanced with literature data."""
        return self.literature_optimizer.optimize_reaction_conditions(
            reactant_smiles, product_smiles
        )

    def validate_catalyst_choice(
        self, catalyst_name: str, reaction_type: str
    ) -> Dict[str, Any]:
        """Validate catalyst choice against literature data."""
        precedents = self.kg_retriever.query_catalyst_literature_precedents(
            catalyst_name, reaction_type
        )
        compatible_solvents = self.kg_retriever.query_solvent_compatibility(
            catalyst_name
        )

        performance_data = self.kg_retriever.query_catalyst_performance(reaction_type)
        catalyst_performance = [
            entry
            for entry in performance_data
            if entry["catalyst_name"] == catalyst_name
        ]

        validation_result = {
            "literature_precedents": precedents,
            "compatible_solvents": compatible_solvents,
            "performance_data": catalyst_performance,
            "validation_score": self._calculate_validation_score(
                precedents, catalyst_performance
            ),
        }

        return validation_result

    def _calculate_validation_score(
        self, precedents: int, performance_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate validation score for catalyst choice."""
        if precedents == 0:
            return 0.0

        base_score = min(precedents / 50, 0.6)  # Up to 0.6 for precedents

        if performance_data:
            yields = [entry["yield"] for entry in performance_data]
            mean_yield = sum(yields) / len(yields)
            performance_score = mean_yield * 0.4  # Up to 0.4 for performance
            base_score += performance_score

        return min(base_score, 1.0)

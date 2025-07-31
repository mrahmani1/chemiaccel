"""
AI-KG Integration Framework for ChemiAccel

This module provides the foundational architecture for integrating ChemiAccel's AI modules
with the Integrated Chemistry Knowledge Graph (IC-KG) to enable contextual AI predictions,
intelligent suggestions, and knowledge graph updates based on AI outputs.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    from ..knowledge_graph import (
        ChemicalOntology,
        EntityType,
        GraphDatabaseBackend,
        GraphQuery,
        GraphQueryResult,
        ICKGSchema,
        QueryLanguage,
        RelationshipType,
        SemanticQueryEngine,
    )

    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False
    logging.warning(
        "Knowledge Graph modules not available. AI-KG integration will be limited."
    )


class KGContextType(Enum):
    """Types of knowledge graph context."""

    MOLECULAR_SIMILARITY = "molecular_similarity"
    STRUCTURE_ACTIVITY = "structure_activity"
    DRUG_TARGET = "drug_target"
    PATHWAY_INTERACTION = "pathway_interaction"
    SYNTHESIS_ROUTE = "synthesis_route"
    PROPERTY_CORRELATION = "property_correlation"
    THERAPEUTIC_INDICATION = "therapeutic_indication"
    ADVERSE_EFFECT = "adverse_effect"
    CHEMICAL_CLASS = "chemical_class"
    REGULATORY_STATUS = "regulatory_status"


class AITaskType(Enum):
    """Types of AI tasks that can leverage KG context."""

    GENERATIVE_CHEMISTRY = "generative_chemistry"
    INVERSE_DESIGN = "inverse_design"
    ACTIVE_LEARNING = "active_learning"
    PROPERTY_PREDICTION = "property_prediction"
    REACTION_PREDICTION = "reaction_prediction"
    SYNTHESIS_PLANNING = "synthesis_planning"
    DRUG_DISCOVERY = "drug_discovery"


@dataclass
class KGContext:
    """Knowledge graph context for AI predictions."""

    context_type: KGContextType
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    source_queries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIContextualPrediction:
    """AI prediction enriched with knowledge graph context."""

    prediction: Any
    confidence: float
    kg_context: KGContext
    explanation: str
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class KGUpdateSuggestion:
    """Suggestion for updating the knowledge graph based on AI predictions."""

    update_type: (
        str  # "add_entity", "add_relationship", "update_property", "add_evidence"
    )
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    priority: str = "medium"  # "high", "medium", "low"
    validation_required: bool = True


class KGContextProvider:
    """Provides knowledge graph context for AI predictions."""

    def __init__(self, semantic_query_engine: "SemanticQueryEngine"):
        """Initialize KG context provider."""
        if not KG_AVAILABLE:
            raise ImportError("Knowledge Graph modules required for KG integration")

        self.query_engine = semantic_query_engine
        self.context_cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def get_molecular_context(
        self, molecules: List[str], context_types: List[KGContextType]
    ) -> Dict[str, KGContext]:
        """Get knowledge graph context for molecules."""
        contexts = {}

        for molecule in molecules:
            cache_key = f"{molecule}_{hash(tuple(context_types))}"

            if cache_key in self.context_cache:
                cache_entry = self.context_cache[cache_key]
                if (datetime.now() - cache_entry["timestamp"]).seconds < self.cache_ttl:
                    contexts[molecule] = cache_entry["context"]
                    continue

            # Gather context for each type
            molecule_contexts = {}
            for context_type in context_types:
                context = await self._get_context_by_type(molecule, context_type)
                if context:
                    molecule_contexts[context_type] = context

            # Combine contexts
            combined_context = self._combine_contexts(molecule_contexts)
            contexts[molecule] = combined_context

            # Cache result
            self.context_cache[cache_key] = {
                "context": combined_context,
                "timestamp": datetime.now(),
            }

        return contexts

    async def _get_context_by_type(
        self, molecule: str, context_type: KGContextType
    ) -> Optional[KGContext]:
        """Get specific type of context for a molecule."""
        try:
            if context_type == KGContextType.MOLECULAR_SIMILARITY:
                return await self._get_similarity_context(molecule)
            elif context_type == KGContextType.STRUCTURE_ACTIVITY:
                return await self._get_sar_context(molecule)
            elif context_type == KGContextType.DRUG_TARGET:
                return await self._get_drug_target_context(molecule)
            elif context_type == KGContextType.PATHWAY_INTERACTION:
                return await self._get_pathway_context(molecule)
            elif context_type == KGContextType.SYNTHESIS_ROUTE:
                return await self._get_synthesis_context(molecule)
            elif context_type == KGContextType.PROPERTY_CORRELATION:
                return await self._get_property_context(molecule)
            elif context_type == KGContextType.THERAPEUTIC_INDICATION:
                return await self._get_therapeutic_context(molecule)
            elif context_type == KGContextType.ADVERSE_EFFECT:
                return await self._get_adverse_effect_context(molecule)
            elif context_type == KGContextType.CHEMICAL_CLASS:
                return await self._get_chemical_class_context(molecule)
            elif context_type == KGContextType.REGULATORY_STATUS:
                return await self._get_regulatory_context(molecule)
            else:
                return None
        except Exception as e:
            logging.error(f"Error getting context {context_type} for {molecule}: {e}")
            return None

    async def _get_similarity_context(self, molecule: str) -> KGContext:
        """Get molecular similarity context."""
        query = f"""
        FIND compounds WHERE SIMILAR_TO '{molecule}' WITH similarity > 0.7
        ORDER BY similarity DESC LIMIT 20
        """

        result = await self._execute_query(query)

        similar_compounds = []
        relationships = []

        for record in result.records:
            compound_data = {
                "smiles": record.get("smiles", ""),
                "name": record.get("name", ""),
                "similarity_score": record.get("similarity", 0.0),
                "molecular_weight": record.get("molecular_weight", 0.0),
                "logp": record.get("logp", 0.0),
            }
            similar_compounds.append(compound_data)

            relationships.append(
                {
                    "type": "SIMILAR_TO",
                    "source": molecule,
                    "target": compound_data["smiles"],
                    "confidence": compound_data["similarity_score"],
                }
            )

        return KGContext(
            context_type=KGContextType.MOLECULAR_SIMILARITY,
            entities=similar_compounds,
            relationships=relationships,
            confidence_scores={"similarity_query": 0.9},
            source_queries=[query],
        )

    async def _get_sar_context(self, molecule: str) -> KGContext:
        """Get structure-activity relationship context."""
        query = f"""
        MATCH (m:Compound {{smiles: '{molecule}'}})-[:HAS_ACTIVITY]->(activity)
        OPTIONAL MATCH (similar:Compound)-[:SIMILAR_TO]-(m)
        OPTIONAL MATCH (similar)-[:HAS_ACTIVITY]->(similar_activity)
        RETURN m, activity, similar, similar_activity
        LIMIT 50
        """

        result = await self._execute_query(query)

        activities = []
        relationships = []
        sar_patterns = []

        for record in result.records:
            if "activity" in record:
                activity_data = {
                    "activity_type": record["activity"].get("type", ""),
                    "value": record["activity"].get("value", 0.0),
                    "unit": record["activity"].get("unit", ""),
                    "assay": record["activity"].get("assay", ""),
                    "confidence": record["activity"].get("confidence", 0.0),
                }
                activities.append(activity_data)

            if "similar" in record and "similar_activity" in record:
                sar_patterns.append(
                    {
                        "similar_compound": record["similar"].get("smiles", ""),
                        "activity_correlation": record["similar_activity"].get(
                            "value", 0.0
                        ),
                        "structural_difference": record.get(
                            "structural_difference", ""
                        ),
                    }
                )

        return KGContext(
            context_type=KGContextType.STRUCTURE_ACTIVITY,
            entities=activities,
            relationships=relationships,
            properties={"sar_patterns": sar_patterns},
            confidence_scores={"sar_query": 0.85},
            source_queries=[query],
        )

    async def _get_drug_target_context(self, molecule: str) -> KGContext:
        """Get drug-target interaction context."""
        query = f"""
        MATCH (d:Drug {{smiles: '{molecule}'}})-[r:TARGETS]->(target)
        OPTIONAL MATCH (target)-[:PARTICIPATES_IN]->(pathway)
        RETURN d, r, target, pathway
        LIMIT 30
        """

        result = await self._execute_query(query)

        targets = []
        pathways = []
        relationships = []

        for record in result.records:
            if "target" in record:
                target_data = {
                    "protein_id": record["target"].get("id", ""),
                    "name": record["target"].get("name", ""),
                    "type": record["target"].get("type", ""),
                    "function": record["target"].get("function", ""),
                    "binding_affinity": record.get("r", {}).get("affinity", 0.0),
                }
                targets.append(target_data)

                relationships.append(
                    {
                        "type": "TARGETS",
                        "source": molecule,
                        "target": target_data["protein_id"],
                        "confidence": record.get("r", {}).get("confidence", 0.0),
                    }
                )

            if "pathway" in record:
                pathway_data = {
                    "pathway_id": record["pathway"].get("id", ""),
                    "name": record["pathway"].get("name", ""),
                    "description": record["pathway"].get("description", ""),
                }
                pathways.append(pathway_data)

        return KGContext(
            context_type=KGContextType.DRUG_TARGET,
            entities=targets + pathways,
            relationships=relationships,
            confidence_scores={"drug_target_query": 0.9},
            source_queries=[query],
        )

    async def _get_pathway_context(self, molecule: str) -> KGContext:
        """Get pathway interaction context."""
        query = f"""
        MATCH (c:Compound {{smiles: '{molecule}'}})-[:AFFECTS]->(pathway:Pathway)
        OPTIONAL MATCH (pathway)-[:INTERACTS_WITH]->(other_pathway)
        RETURN c, pathway, other_pathway
        LIMIT 20
        """

        result = await self._execute_query(query)

        pathways = []
        interactions = []

        for record in result.records:
            if "pathway" in record:
                pathway_data = {
                    "pathway_id": record["pathway"].get("id", ""),
                    "name": record["pathway"].get("name", ""),
                    "description": record["pathway"].get("description", ""),
                    "effect_type": record.get("effect_type", ""),
                }
                pathways.append(pathway_data)

            if "other_pathway" in record:
                interactions.append(
                    {
                        "type": "PATHWAY_INTERACTION",
                        "source": record["pathway"].get("id", ""),
                        "target": record["other_pathway"].get("id", ""),
                        "interaction_type": record.get("interaction_type", ""),
                    }
                )

        return KGContext(
            context_type=KGContextType.PATHWAY_INTERACTION,
            entities=pathways,
            relationships=interactions,
            confidence_scores={"pathway_query": 0.8},
            source_queries=[query],
        )

    async def _get_synthesis_context(self, molecule: str) -> KGContext:
        """Get synthesis route context."""
        query = f"""
        MATCH (product:Compound {{smiles: '{molecule}'}})
        <-[:PRODUCES]-(reaction:Reaction)
        <-[:PARTICIPATES_IN]-(reactant:Compound)
        RETURN product, reaction, reactant
        LIMIT 15
        """

        result = await self._execute_query(query)

        reactions = []
        reactants = []
        synthesis_routes = []

        for record in result.records:
            if "reaction" in record:
                reaction_data = {
                    "reaction_id": record["reaction"].get("id", ""),
                    "reaction_smiles": record["reaction"].get("smiles", ""),
                    "yield": record["reaction"].get("yield", 0.0),
                    "conditions": record["reaction"].get("conditions", {}),
                    "literature_refs": record["reaction"].get("references", []),
                }
                reactions.append(reaction_data)

            if "reactant" in record:
                reactant_data = {
                    "smiles": record["reactant"].get("smiles", ""),
                    "name": record["reactant"].get("name", ""),
                    "availability": record["reactant"].get("availability", ""),
                    "cost": record["reactant"].get("cost", 0.0),
                }
                reactants.append(reactant_data)

        return KGContext(
            context_type=KGContextType.SYNTHESIS_ROUTE,
            entities=reactions + reactants,
            properties={"synthesis_routes": synthesis_routes},
            confidence_scores={"synthesis_query": 0.85},
            source_queries=[query],
        )

    async def _get_property_context(self, molecule: str) -> KGContext:
        """Get property correlation context."""
        query = f"""
        MATCH (c:Compound {{smiles: '{molecule}'}})
        OPTIONAL MATCH (similar:Compound)-[:SIMILAR_TO]-(c)
        RETURN c, similar,
               c.molecular_weight, c.logp, c.tpsa, c.solubility,
               similar.molecular_weight, similar.logp, similar.tpsa, similar.solubility
        LIMIT 25
        """

        result = await self._execute_query(query)

        properties = []
        correlations = []

        for record in result.records:
            if "c" in record:
                compound_props = {
                    "molecular_weight": record.get("c.molecular_weight", 0.0),
                    "logp": record.get("c.logp", 0.0),
                    "tpsa": record.get("c.tpsa", 0.0),
                    "solubility": record.get("c.solubility", 0.0),
                }
                properties.append(compound_props)

            if "similar" in record:
                similar_props = {
                    "molecular_weight": record.get("similar.molecular_weight", 0.0),
                    "logp": record.get("similar.logp", 0.0),
                    "tpsa": record.get("similar.tpsa", 0.0),
                    "solubility": record.get("similar.solubility", 0.0),
                }
                correlations.append(
                    {
                        "reference_compound": record["similar"].get("smiles", ""),
                        "properties": similar_props,
                    }
                )

        return KGContext(
            context_type=KGContextType.PROPERTY_CORRELATION,
            entities=properties,
            properties={"property_correlations": correlations},
            confidence_scores={"property_query": 0.9},
            source_queries=[query],
        )

    async def _get_therapeutic_context(self, molecule: str) -> KGContext:
        """Get therapeutic indication context."""
        query = f"""
        MATCH (d:Drug {{smiles: '{molecule}'}})-[:TREATS]->(indication)
        OPTIONAL MATCH (d)-[:HAS_MECHANISM]->(mechanism)
        RETURN d, indication, mechanism
        LIMIT 20
        """

        result = await self._execute_query(query)

        indications = []
        mechanisms = []

        for record in result.records:
            if "indication" in record:
                indication_data = {
                    "condition": record["indication"].get("name", ""),
                    "disease_class": record["indication"].get("class", ""),
                    "severity": record["indication"].get("severity", ""),
                    "prevalence": record["indication"].get("prevalence", 0.0),
                    "evidence_level": record["indication"].get("evidence_level", ""),
                }
                indications.append(indication_data)

            if "mechanism" in record:
                mechanism_data = {
                    "mechanism_type": record["mechanism"].get("type", ""),
                    "description": record["mechanism"].get("description", ""),
                    "target_specificity": record["mechanism"].get("specificity", 0.0),
                }
                mechanisms.append(mechanism_data)

        return KGContext(
            context_type=KGContextType.THERAPEUTIC_INDICATION,
            entities=indications + mechanisms,
            confidence_scores={"therapeutic_query": 0.85},
            source_queries=[query],
        )

    async def _get_adverse_effect_context(self, molecule: str) -> KGContext:
        """Get adverse effect context."""
        query = f"""
        MATCH (d:Drug {{smiles: '{molecule}'}})-[:CAUSES]->(adverse_effect)
        RETURN d, adverse_effect
        LIMIT 15
        """

        result = await self._execute_query(query)

        adverse_effects = []

        for record in result.records:
            if "adverse_effect" in record:
                effect_data = {
                    "effect_type": record["adverse_effect"].get("type", ""),
                    "severity": record["adverse_effect"].get("severity", ""),
                    "frequency": record["adverse_effect"].get("frequency", 0.0),
                    "organ_system": record["adverse_effect"].get("organ_system", ""),
                    "reversibility": record["adverse_effect"].get("reversible", True),
                }
                adverse_effects.append(effect_data)

        return KGContext(
            context_type=KGContextType.ADVERSE_EFFECT,
            entities=adverse_effects,
            confidence_scores={"adverse_effect_query": 0.8},
            source_queries=[query],
        )

    async def _get_chemical_class_context(self, molecule: str) -> KGContext:
        """Get chemical class context."""
        query = f"""
        MATCH (c:Compound {{smiles: '{molecule}'}})-[:IS_A]->(class)
        OPTIONAL MATCH (class)-[:SUBCLASS_OF]->(parent_class)
        RETURN c, class, parent_class
        LIMIT 10
        """

        result = await self._execute_query(query)

        classes = []
        hierarchies = []

        for record in result.records:
            if "class" in record:
                class_data = {
                    "class_name": record["class"].get("name", ""),
                    "description": record["class"].get("description", ""),
                    "typical_properties": record["class"].get("typical_properties", {}),
                    "member_count": record["class"].get("member_count", 0),
                }
                classes.append(class_data)

            if "parent_class" in record:
                hierarchies.append(
                    {
                        "child_class": record["class"].get("name", ""),
                        "parent_class": record["parent_class"].get("name", ""),
                        "relationship_type": "SUBCLASS_OF",
                    }
                )

        return KGContext(
            context_type=KGContextType.CHEMICAL_CLASS,
            entities=classes,
            relationships=hierarchies,
            confidence_scores={"class_query": 0.95},
            source_queries=[query],
        )

    async def _get_regulatory_context(self, molecule: str) -> KGContext:
        """Get regulatory status context."""
        query = f"""
        MATCH (d:Drug {{smiles: '{molecule}'}})-[:HAS_STATUS]->(status)
        RETURN d, status
        LIMIT 5
        """

        result = await self._execute_query(query)

        regulatory_info = []

        for record in result.records:
            if "status" in record:
                status_data = {
                    "regulatory_body": record["status"].get("authority", ""),
                    "approval_status": record["status"].get("status", ""),
                    "approval_date": record["status"].get("date", ""),
                    "indication": record["status"].get("indication", ""),
                    "phase": record["status"].get("phase", ""),
                    "restrictions": record["status"].get("restrictions", []),
                }
                regulatory_info.append(status_data)

        return KGContext(
            context_type=KGContextType.REGULATORY_STATUS,
            entities=regulatory_info,
            confidence_scores={"regulatory_query": 0.95},
            source_queries=[query],
        )

    def _combine_contexts(self, contexts: Dict[KGContextType, KGContext]) -> KGContext:
        """Combine multiple context types into a unified context."""
        combined_entities = []
        combined_relationships = []
        combined_properties = {}
        combined_confidence_scores = {}
        combined_queries = []

        for context_type, context in contexts.items():
            combined_entities.extend(context.entities)
            combined_relationships.extend(context.relationships)
            combined_properties.update(context.properties)
            combined_confidence_scores.update(context.confidence_scores)
            combined_queries.extend(context.source_queries)

        # Calculate overall confidence as weighted average
        if combined_confidence_scores:
            overall_confidence = np.mean(list(combined_confidence_scores.values()))
        else:
            overall_confidence = 0.0

        return KGContext(
            context_type=KGContextType.MOLECULAR_SIMILARITY,  # Primary context type
            entities=combined_entities,
            relationships=combined_relationships,
            properties=combined_properties,
            confidence_scores={
                **combined_confidence_scores,
                "overall": overall_confidence,
            },
            source_queries=combined_queries,
            metadata={"context_types": list(contexts.keys())},
        )

    async def _execute_query(self, query: str) -> GraphQueryResult:
        """Execute query against knowledge graph."""
        try:
            graph_query = GraphQuery(
                query=query, language=QueryLanguage.CHEMICAL_DSL, parameters={}
            )

            # Use thread pool for blocking operations
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self.query_engine.execute_query, graph_query
            )

            return result
        except Exception as e:
            logging.error(f"Error executing query: {query}. Error: {e}")
            # Return empty result
            return GraphQueryResult(
                records=[],
                metadata={"error": str(e)},
                execution_time=0.0,
                query=graph_query,
            )


class KGFeedbackManager:
    """Manages feedback from AI predictions to update the knowledge graph."""

    def __init__(self, semantic_query_engine: "SemanticQueryEngine"):
        """Initialize KG feedback manager."""
        if not KG_AVAILABLE:
            raise ImportError("Knowledge Graph modules required for KG integration")

        self.query_engine = semantic_query_engine
        self.pending_updates = []
        self.confidence_threshold = 0.7
        self.validation_required_threshold = 0.5

    def suggest_kg_updates(
        self, ai_predictions: List[AIContextualPrediction], task_type: AITaskType
    ) -> List[KGUpdateSuggestion]:
        """Generate suggestions for updating the knowledge graph based on AI predictions."""
        suggestions = []

        for prediction in ai_predictions:
            # Generate different types of updates based on prediction type and confidence
            if prediction.confidence > self.confidence_threshold:
                suggestions.extend(
                    self._generate_high_confidence_updates(prediction, task_type)
                )
            elif prediction.confidence > self.validation_required_threshold:
                suggestions.extend(
                    self._generate_validation_required_updates(prediction, task_type)
                )

        return suggestions

    def _generate_high_confidence_updates(
        self, prediction: AIContextualPrediction, task_type: AITaskType
    ) -> List[KGUpdateSuggestion]:
        """Generate high-confidence update suggestions."""
        suggestions = []

        if task_type == AITaskType.GENERATIVE_CHEMISTRY:
            suggestions.extend(self._suggest_generated_molecule_updates(prediction))
        elif task_type == AITaskType.INVERSE_DESIGN:
            suggestions.extend(self._suggest_optimized_molecule_updates(prediction))
        elif task_type == AITaskType.ACTIVE_LEARNING:
            suggestions.extend(self._suggest_active_learning_updates(prediction))
        elif task_type == AITaskType.PROPERTY_PREDICTION:
            suggestions.extend(self._suggest_property_updates(prediction))

        return suggestions

    def _generate_validation_required_updates(
        self, prediction: AIContextualPrediction, task_type: AITaskType
    ) -> List[KGUpdateSuggestion]:
        """Generate update suggestions that require validation."""
        suggestions = []

        # Similar to high-confidence updates but with validation_required=True
        base_suggestions = self._generate_high_confidence_updates(prediction, task_type)

        for suggestion in base_suggestions:
            suggestion.validation_required = True
            suggestion.priority = "medium" if suggestion.priority == "high" else "low"
            suggestion.rationale += " (Requires validation due to moderate confidence)"

        suggestions.extend(base_suggestions)
        return suggestions

    def _suggest_generated_molecule_updates(
        self, prediction: AIContextualPrediction
    ) -> List[KGUpdateSuggestion]:
        """Suggest updates for generated molecules."""
        suggestions = []

        if (
            isinstance(prediction.prediction, dict)
            and "molecules" in prediction.prediction
        ):
            molecules = prediction.prediction["molecules"]

            for mol_data in molecules:
                smiles = mol_data.get("smiles", "")
                properties = mol_data.get("properties", {})

                # Suggest adding new molecule entity
                suggestions.append(
                    KGUpdateSuggestion(
                        update_type="add_entity",
                        entities=[
                            {
                                "type": "Compound",
                                "smiles": smiles,
                                "source": "ai_generation",
                                "generation_method": "generative_chemistry",
                                "confidence": prediction.confidence,
                            }
                        ],
                        properties=properties,
                        confidence=prediction.confidence,
                        evidence=[
                            {
                                "type": "ai_generation",
                                "model": "generative_chemistry",
                                "context": prediction.kg_context.metadata,
                            }
                        ],
                        rationale=f"Novel molecule generated with {prediction.confidence:.2f} confidence",
                        priority="high",
                        validation_required=prediction.confidence < 0.9,
                    )
                )

                # Suggest property relationships
                for prop_name, prop_value in properties.items():
                    suggestions.append(
                        KGUpdateSuggestion(
                            update_type="add_relationship",
                            relationships=[
                                {
                                    "type": "HAS_PROPERTY",
                                    "source": smiles,
                                    "target": f"{prop_name}_{prop_value}",
                                    "confidence": prediction.confidence,
                                }
                            ],
                            confidence=prediction.confidence,
                            rationale=f"Predicted property {prop_name}={prop_value}",
                            priority="medium",
                        )
                    )

        return suggestions

    def _suggest_optimized_molecule_updates(
        self, prediction: AIContextualPrediction
    ) -> List[KGUpdateSuggestion]:
        """Suggest updates for optimized molecules."""
        suggestions = []

        if (
            isinstance(prediction.prediction, dict)
            and "optimized_molecules" in prediction.prediction
        ):
            optimized_molecules = prediction.prediction["optimized_molecules"]
            target_properties = prediction.prediction.get("target_properties", {})

            for mol_smiles in optimized_molecules:
                # Suggest optimization relationship
                suggestions.append(
                    KGUpdateSuggestion(
                        update_type="add_relationship",
                        relationships=[
                            {
                                "type": "OPTIMIZED_FOR",
                                "source": mol_smiles,
                                "target": json.dumps(target_properties),
                                "confidence": prediction.confidence,
                                "optimization_method": "inverse_design",
                            }
                        ],
                        confidence=prediction.confidence,
                        rationale=f"Molecule optimized for target properties: {target_properties}",
                        priority="high",
                    )
                )

        return suggestions

    def _suggest_active_learning_updates(
        self, prediction: AIContextualPrediction
    ) -> List[KGUpdateSuggestion]:
        """Suggest updates from active learning results."""
        suggestions = []

        if (
            isinstance(prediction.prediction, dict)
            and "selected_molecules" in prediction.prediction
        ):
            selected_molecules = prediction.prediction["selected_molecules"]
            acquisition_scores = prediction.prediction.get("acquisition_scores", [])

            for i, mol_smiles in enumerate(selected_molecules):
                score = acquisition_scores[i] if i < len(acquisition_scores) else 0.0

                # Suggest information content relationship
                suggestions.append(
                    KGUpdateSuggestion(
                        update_type="add_relationship",
                        relationships=[
                            {
                                "type": "HIGH_INFORMATION_CONTENT",
                                "source": mol_smiles,
                                "target": "active_learning_selection",
                                "acquisition_score": score,
                                "confidence": prediction.confidence,
                            }
                        ],
                        confidence=prediction.confidence,
                        rationale=f"High information content molecule (score: {score:.3f})",
                        priority="medium",
                    )
                )

        return suggestions

    def _suggest_property_updates(
        self, prediction: AIContextualPrediction
    ) -> List[KGUpdateSuggestion]:
        """Suggest updates for property predictions."""
        suggestions = []

        if isinstance(prediction.prediction, dict):
            molecule = prediction.prediction.get("molecule", "")
            properties = prediction.prediction.get("properties", {})

            for prop_name, prop_value in properties.items():
                suggestions.append(
                    KGUpdateSuggestion(
                        update_type="update_property",
                        entities=[
                            {
                                "smiles": molecule,
                                "property": prop_name,
                                "value": prop_value,
                                "prediction_confidence": prediction.confidence,
                            }
                        ],
                        confidence=prediction.confidence,
                        evidence=[
                            {
                                "type": "ai_prediction",
                                "model": "property_predictor",
                                "supporting_evidence": prediction.supporting_evidence,
                            }
                        ],
                        rationale=f"AI-predicted property {prop_name}={prop_value}",
                        priority="medium",
                    )
                )

        return suggestions

    async def validate_and_apply_updates(
        self, suggestions: List[KGUpdateSuggestion]
    ) -> Dict[str, Any]:
        """Validate and apply knowledge graph updates."""
        results = {"applied": [], "rejected": [], "pending_validation": []}

        for suggestion in suggestions:
            if suggestion.validation_required and suggestion.confidence < 0.8:
                results["pending_validation"].append(suggestion)
                continue

            try:
                # Apply update based on type
                if suggestion.update_type == "add_entity":
                    success = await self._add_entities(suggestion)
                elif suggestion.update_type == "add_relationship":
                    success = await self._add_relationships(suggestion)
                elif suggestion.update_type == "update_property":
                    success = await self._update_properties(suggestion)
                else:
                    success = False

                if success:
                    results["applied"].append(suggestion)
                else:
                    results["rejected"].append(suggestion)

            except Exception as e:
                logging.error(f"Error applying update: {e}")
                results["rejected"].append(suggestion)

        return results

    async def _add_entities(self, suggestion: KGUpdateSuggestion) -> bool:
        """Add new entities to the knowledge graph."""
        try:
            for entity in suggestion.entities:
                # Construct appropriate CREATE query
                if entity.get("type") == "Compound":
                    query = f"""
                    CREATE (c:Compound {{
                        smiles: '{entity['smiles']}',
                        source: '{entity.get('source', 'ai')}',
                        generation_method: '{entity.get('generation_method', '')}',
                        confidence: {entity.get('confidence', 0.0)},
                        created_date: '{datetime.now().isoformat()}'
                    }})
                    """

                    result = await self._execute_update_query(query)
                    if not result:
                        return False

            return True
        except Exception as e:
            logging.error(f"Error adding entities: {e}")
            return False

    async def _add_relationships(self, suggestion: KGUpdateSuggestion) -> bool:
        """Add new relationships to the knowledge graph."""
        try:
            for relationship in suggestion.relationships:
                rel_type = relationship["type"]
                source = relationship["source"]
                target = relationship["target"]
                confidence = relationship.get("confidence", 0.0)

                query = f"""
                MATCH (source {{smiles: '{source}'}})
                CREATE (source)-[r:{rel_type} {{
                    confidence: {confidence},
                    created_date: '{datetime.now().isoformat()}',
                    source: 'ai_prediction'
                }}]->(target:Entity {{id: '{target}'}})
                """

                result = await self._execute_update_query(query)
                if not result:
                    return False

            return True
        except Exception as e:
            logging.error(f"Error adding relationships: {e}")
            return False

    async def _update_properties(self, suggestion: KGUpdateSuggestion) -> bool:
        """Update entity properties in the knowledge graph."""
        try:
            for entity in suggestion.entities:
                smiles = entity["smiles"]
                prop_name = entity["property"]
                prop_value = entity["value"]
                confidence = entity.get("prediction_confidence", 0.0)

                query = f"""
                MATCH (c:Compound {{smiles: '{smiles}'}})
                SET c.{prop_name} = {prop_value},
                    c.{prop_name}_confidence = {confidence},
                    c.{prop_name}_updated = '{datetime.now().isoformat()}'
                """

                result = await self._execute_update_query(query)
                if not result:
                    return False

            return True
        except Exception as e:
            logging.error(f"Error updating properties: {e}")
            return False

    async def _execute_update_query(self, query: str) -> bool:
        """Execute update query against knowledge graph."""
        try:
            # This would need to be implemented based on the specific backend
            # For now, we'll simulate success
            logging.info(f"Executing update query: {query}")
            return True
        except Exception as e:
            logging.error(f"Error executing update query: {e}")
            return False


class AIKGIntegrationManager:
    """Main manager for AI-KG integration."""

    def __init__(self, semantic_query_engine: "SemanticQueryEngine"):
        """Initialize AI-KG integration manager."""
        if not KG_AVAILABLE:
            raise ImportError("Knowledge Graph modules required for AI-KG integration")

        self.context_provider = KGContextProvider(semantic_query_engine)
        self.feedback_manager = KGFeedbackManager(semantic_query_engine)

    async def enrich_ai_prediction(
        self,
        prediction: Any,
        molecules: List[str],
        task_type: AITaskType,
        context_types: List[KGContextType] = None,
    ) -> AIContextualPrediction:
        """Enrich AI prediction with knowledge graph context."""
        if context_types is None:
            context_types = [
                KGContextType.MOLECULAR_SIMILARITY,
                KGContextType.STRUCTURE_ACTIVITY,
                KGContextType.PROPERTY_CORRELATION,
            ]

        # Get KG context for molecules
        molecular_contexts = await self.context_provider.get_molecular_context(
            molecules, context_types
        )

        # Combine contexts
        combined_context = self._combine_molecular_contexts(molecular_contexts)

        # Generate explanation based on context
        explanation = self._generate_explanation(
            prediction, combined_context, task_type
        )

        # Extract supporting and contradicting evidence
        supporting_evidence, contradicting_evidence = self._analyze_evidence(
            prediction, combined_context
        )

        # Identify uncertainty factors
        uncertainty_factors = self._identify_uncertainty_factors(
            prediction, combined_context
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            prediction, combined_context, task_type
        )

        # Calculate overall confidence
        confidence = self._calculate_contextual_confidence(prediction, combined_context)

        return AIContextualPrediction(
            prediction=prediction,
            confidence=confidence,
            kg_context=combined_context,
            explanation=explanation,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            uncertainty_factors=uncertainty_factors,
            recommendations=recommendations,
        )

    async def update_kg_from_ai(
        self,
        contextual_predictions: List[AIContextualPrediction],
        task_type: AITaskType,
    ) -> Dict[str, Any]:
        """Update knowledge graph based on AI predictions."""
        # Generate update suggestions
        suggestions = self.feedback_manager.suggest_kg_updates(
            contextual_predictions, task_type
        )

        # Validate and apply updates
        results = await self.feedback_manager.validate_and_apply_updates(suggestions)

        return results

    def _combine_molecular_contexts(
        self, molecular_contexts: Dict[str, KGContext]
    ) -> KGContext:
        """Combine contexts from multiple molecules."""
        all_entities = []
        all_relationships = []
        all_properties = {}
        all_confidence_scores = {}
        all_queries = []

        for molecule, context in molecular_contexts.items():
            all_entities.extend(context.entities)
            all_relationships.extend(context.relationships)
            all_properties.update(context.properties)
            all_confidence_scores.update(context.confidence_scores)
            all_queries.extend(context.source_queries)

        return KGContext(
            context_type=KGContextType.MOLECULAR_SIMILARITY,
            entities=all_entities,
            relationships=all_relationships,
            properties=all_properties,
            confidence_scores=all_confidence_scores,
            source_queries=all_queries,
            metadata={"num_molecules": len(molecular_contexts)},
        )

    def _generate_explanation(
        self, prediction: Any, context: KGContext, task_type: AITaskType
    ) -> str:
        """Generate human-readable explanation for the prediction."""
        explanations = []

        # Base explanation
        if task_type == AITaskType.GENERATIVE_CHEMISTRY:
            explanations.append(
                "Generated novel molecules based on learned chemical patterns."
            )
        elif task_type == AITaskType.INVERSE_DESIGN:
            explanations.append(
                "Optimized molecular structures to achieve target properties."
            )
        elif task_type == AITaskType.ACTIVE_LEARNING:
            explanations.append(
                "Selected molecules with highest information content for experimental validation."
            )

        # Context-based explanations
        if context.entities:
            similar_count = len(
                [e for e in context.entities if "similarity_score" in e]
            )
            if similar_count > 0:
                explanations.append(
                    f"Found {similar_count} similar molecules in the knowledge graph."
                )

        if context.relationships:
            target_rels = len(
                [r for r in context.relationships if r["type"] == "TARGETS"]
            )
            if target_rels > 0:
                explanations.append(
                    f"Identified {target_rels} known target interactions."
                )

        # Confidence explanation
        overall_confidence = context.confidence_scores.get("overall", 0.0)
        if overall_confidence > 0.8:
            explanations.append(
                "High confidence based on extensive knowledge graph evidence."
            )
        elif overall_confidence > 0.6:
            explanations.append("Moderate confidence with some supporting evidence.")
        else:
            explanations.append(
                "Lower confidence due to limited knowledge graph coverage."
            )

        return " ".join(explanations)

    def _analyze_evidence(
        self, prediction: Any, context: KGContext
    ) -> Tuple[List[Dict], List[Dict]]:
        """Analyze supporting and contradicting evidence."""
        supporting_evidence = []
        contradicting_evidence = []

        # Analyze entities for evidence
        for entity in context.entities:
            if "similarity_score" in entity and entity["similarity_score"] > 0.8:
                supporting_evidence.append(
                    {
                        "type": "high_similarity",
                        "evidence": f"Highly similar molecule: {entity.get('name', 'Unknown')}",
                        "confidence": entity["similarity_score"],
                    }
                )

            if "activity_type" in entity:
                supporting_evidence.append(
                    {
                        "type": "known_activity",
                        "evidence": f"Known activity: {entity['activity_type']}",
                        "confidence": entity.get("confidence", 0.0),
                    }
                )

        # Analyze relationships for evidence
        for relationship in context.relationships:
            if (
                relationship["type"] == "TARGETS"
                and relationship.get("confidence", 0) > 0.7
            ):
                supporting_evidence.append(
                    {
                        "type": "target_interaction",
                        "evidence": f"Known target: {relationship['target']}",
                        "confidence": relationship["confidence"],
                    }
                )

        return supporting_evidence, contradicting_evidence

    def _identify_uncertainty_factors(
        self, prediction: Any, context: KGContext
    ) -> List[str]:
        """Identify factors contributing to prediction uncertainty."""
        uncertainty_factors = []

        # Limited knowledge graph coverage
        if len(context.entities) < 5:
            uncertainty_factors.append(
                "Limited knowledge graph coverage for similar molecules"
            )

        # Low confidence scores
        overall_confidence = context.confidence_scores.get("overall", 0.0)
        if overall_confidence < 0.6:
            uncertainty_factors.append("Low confidence in knowledge graph evidence")

        # Contradicting evidence
        if len(context.relationships) > 0:
            conflicting_rels = [
                r for r in context.relationships if r.get("confidence", 1.0) < 0.5
            ]
            if len(conflicting_rels) > 0:
                uncertainty_factors.append("Conflicting evidence in knowledge graph")

        # Novel chemical space
        similarity_scores = [
            e.get("similarity_score", 0)
            for e in context.entities
            if "similarity_score" in e
        ]
        if similarity_scores and max(similarity_scores) < 0.6:
            uncertainty_factors.append(
                "Operating in novel chemical space with limited precedents"
            )

        return uncertainty_factors

    def _generate_recommendations(
        self, prediction: Any, context: KGContext, task_type: AITaskType
    ) -> List[str]:
        """Generate actionable recommendations based on prediction and context."""
        recommendations = []

        # Task-specific recommendations
        if task_type == AITaskType.GENERATIVE_CHEMISTRY:
            recommendations.append(
                "Validate generated molecules through computational or experimental methods"
            )

            if len(context.entities) > 10:
                recommendations.append(
                    "Consider synthesizing the most promising candidates with high similarity to known actives"
                )

        elif task_type == AITaskType.INVERSE_DESIGN:
            recommendations.append("Verify optimized molecules satisfy all constraints")

            # Check for known targets
            target_count = len(
                [r for r in context.relationships if r["type"] == "TARGETS"]
            )
            if target_count > 0:
                recommendations.append(
                    "Evaluate target selectivity for identified interactions"
                )

        elif task_type == AITaskType.ACTIVE_LEARNING:
            recommendations.append(
                "Prioritize selected molecules for experimental validation"
            )

            high_info_molecules = [
                e for e in context.entities if e.get("acquisition_score", 0) > 0.8
            ]
            if high_info_molecules:
                recommendations.append(
                    "Focus on molecules with highest information content scores"
                )

        # General recommendations based on context
        overall_confidence = context.confidence_scores.get("overall", 0.0)
        if overall_confidence < 0.7:
            recommendations.append(
                "Gather additional experimental data to improve prediction confidence"
            )

        if len(context.relationships) > 0:
            recommendations.append(
                "Consider known relationships when interpreting results"
            )

        return recommendations

    def _calculate_contextual_confidence(
        self, prediction: Any, context: KGContext
    ) -> float:
        """Calculate confidence score incorporating knowledge graph context."""
        base_confidence = getattr(prediction, "confidence", 0.5)

        # Adjust based on knowledge graph evidence
        kg_confidence = context.confidence_scores.get("overall", 0.0)

        # Weighted combination
        contextual_confidence = 0.7 * base_confidence + 0.3 * kg_confidence

        # Boost for high-quality evidence
        if len(context.entities) > 10 and kg_confidence > 0.8:
            contextual_confidence = min(1.0, contextual_confidence + 0.1)

        # Penalty for contradicting evidence
        contradicting_count = len(
            [r for r in context.relationships if r.get("confidence", 1.0) < 0.3]
        )
        if contradicting_count > 2:
            contextual_confidence = max(0.0, contextual_confidence - 0.2)

        return contextual_confidence

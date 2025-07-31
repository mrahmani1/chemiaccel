"""
Advanced Reasoning Rules and Inference Mechanisms

Comprehensive rule-based reasoning system for chemical knowledge inference
including molecular similarity, drug interactions, and chemical property prediction.
"""

import json
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .ontology import ChemicalOntology, EntityType, RelationshipType
from .semantic_query_engine import InferredFact, ReasoningRule


class RuleType(Enum):
    """Types of reasoning rules."""

    TRANSITIVE = "transitive"
    SYMMETRIC = "symmetric"
    INVERSE = "inverse"
    SUBSUMPTION = "subsumption"
    EQUIVALENCE = "equivalence"
    IMPLICATION = "implication"
    CAUSAL = "causal"
    STATISTICAL = "statistical"
    CHEMICAL_SIMILARITY = "chemical_similarity"
    STRUCTURAL = "structural"


class ConfidenceModel(Enum):
    """Confidence calculation models."""

    MULTIPLICATIVE = "multiplicative"
    BAYESIAN = "bayesian"
    FUZZY_LOGIC = "fuzzy_logic"
    EVIDENCE_ACCUMULATION = "evidence_accumulation"


@dataclass
class RuleCondition:
    """Represents a condition for rule application."""

    condition_type: str
    expression: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class RuleConsequent:
    """Represents the consequent of a reasoning rule."""

    template: str
    confidence_adjustment: float = 1.0
    metadata_template: Dict[str, str] = field(default_factory=dict)


class AdvancedReasoningRule(ReasoningRule):
    """Extended reasoning rule with advanced features."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        rule_type: RuleType,
        antecedent_patterns: List[str],
        consequent: RuleConsequent,
        conditions: List[RuleCondition] = None,
        confidence_model: ConfidenceModel = ConfidenceModel.MULTIPLICATIVE,
        min_evidence_strength: float = 0.5,
        max_inference_depth: int = 3,
        temporal_constraints: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(
            rule_id,
            name,
            description,
            pattern=" AND ".join(antecedent_patterns),
            conclusion_template=consequent.template,
            **kwargs,
        )

        self.rule_type = rule_type
        self.antecedent_patterns = antecedent_patterns
        self.consequent = consequent
        self.conditions = conditions or []
        self.confidence_model = confidence_model
        self.min_evidence_strength = min_evidence_strength
        self.max_inference_depth = max_inference_depth
        self.temporal_constraints = temporal_constraints or {}

        # Performance tracking
        self.application_count = 0
        self.success_count = 0
        self.average_confidence = 0.0
        self.last_applied = None


class ChemicalReasoningRuleLibrary:
    """Library of domain-specific chemical reasoning rules."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.rules = {}
        self._initialize_chemical_rules()

    def get_all_rules(self) -> List[AdvancedReasoningRule]:
        """Get all available reasoning rules."""
        return list(self.rules.values())

    def get_rules_by_type(self, rule_type: RuleType) -> List[AdvancedReasoningRule]:
        """Get rules of specific type."""
        return [rule for rule in self.rules.values() if rule.rule_type == rule_type]

    def get_rule(self, rule_id: str) -> Optional[AdvancedReasoningRule]:
        """Get specific rule by ID."""
        return self.rules.get(rule_id)

    def add_rule(self, rule: AdvancedReasoningRule) -> None:
        """Add a new reasoning rule."""
        self.rules[rule.rule_id] = rule
        logging.info(f"Added reasoning rule: {rule.name}")

    def _initialize_chemical_rules(self) -> None:
        """Initialize domain-specific chemical reasoning rules."""

        # Drug similarity inference
        self._add_drug_similarity_rules()

        # Chemical property prediction
        self._add_property_prediction_rules()

        # Drug interaction inference
        self._add_drug_interaction_rules()

        # Metabolic pathway reasoning
        self._add_metabolic_pathway_rules()

        # Toxicity prediction rules
        self._add_toxicity_prediction_rules()

        # Structure-activity relationship rules
        self._add_sar_rules()

        # Therapeutic class reasoning
        self._add_therapeutic_class_rules()

    def _add_drug_similarity_rules(self) -> None:
        """Add drug similarity-based reasoning rules."""

        # Similar drugs have similar properties
        similarity_property_rule = AdvancedReasoningRule(
            rule_id="drug_similarity_properties",
            name="Drug Similarity Property Inference",
            description="Similar drugs likely have similar pharmacological properties",
            rule_type=RuleType.CHEMICAL_SIMILARITY,
            antecedent_patterns=[
                "?drug1 SIMILAR_TO ?drug2",
                "?drug1 HAS_PROPERTY ?property",
            ],
            consequent=RuleConsequent(
                template="?drug2 HAS_PROPERTY ?property",
                confidence_adjustment=0.8,
                metadata_template={"inference_type": "similarity_based"},
            ),
            conditions=[
                RuleCondition("similarity_threshold", "similarity_score >= 0.7"),
                RuleCondition(
                    "property_type",
                    "property.type IN ['pharmacokinetic', 'therapeutic']",
                ),
            ],
            confidence_model=ConfidenceModel.BAYESIAN,
            min_evidence_strength=0.6,
            priority=2,
            enabled=True,
        )
        self.add_rule(similarity_property_rule)

        # Similar drugs treat similar conditions
        similarity_indication_rule = AdvancedReasoningRule(
            rule_id="drug_similarity_indications",
            name="Drug Similarity Indication Inference",
            description="Similar drugs may treat similar conditions",
            rule_type=RuleType.CHEMICAL_SIMILARITY,
            antecedent_patterns=[
                "?drug1 SIMILAR_TO ?drug2",
                "?drug1 TREATS ?condition1",
                "?condition1 SIMILAR_TO ?condition2",
            ],
            consequent=RuleConsequent(
                template="?drug2 TREATS ?condition2",
                confidence_adjustment=0.7,
                metadata_template={"inference_type": "cross_similarity"},
            ),
            conditions=[
                RuleCondition("drug_similarity", "drug_similarity >= 0.8"),
                RuleCondition("condition_similarity", "condition_similarity >= 0.6"),
            ],
            confidence_model=ConfidenceModel.MULTIPLICATIVE,
            priority=1,
            enabled=True,
        )
        self.add_rule(similarity_indication_rule)

    def _add_property_prediction_rules(self) -> None:
        """Add chemical property prediction rules."""

        # Molecular weight prediction from structure
        mw_structure_rule = AdvancedReasoningRule(
            rule_id="molecular_weight_from_structure",
            name="Molecular Weight from Structure",
            description="Predict molecular weight from chemical structure",
            rule_type=RuleType.STRUCTURAL,
            antecedent_patterns=[
                "?compound HAS_STRUCTURE ?structure",
                "?structure HAS_MOLECULAR_FORMULA ?formula",
            ],
            consequent=RuleConsequent(
                template="?compound HAS_MOLECULAR_WEIGHT ?calculated_mw",
                confidence_adjustment=0.95,
                metadata_template={"calculation_method": "formula_based"},
            ),
            conditions=[
                RuleCondition("formula_validity", "is_valid_molecular_formula(formula)")
            ],
            confidence_model=ConfidenceModel.EVIDENCE_ACCUMULATION,
            priority=3,
            enabled=True,
        )
        self.add_rule(mw_structure_rule)

        # LogP prediction from structure
        logp_prediction_rule = AdvancedReasoningRule(
            rule_id="logp_from_fragments",
            name="LogP from Molecular Fragments",
            description="Predict LogP from molecular fragment contributions",
            rule_type=RuleType.STRUCTURAL,
            antecedent_patterns=[
                "?compound CONTAINS_FRAGMENT ?fragment",
                "?fragment HAS_LOGP_CONTRIBUTION ?contribution",
            ],
            consequent=RuleConsequent(
                template="?compound HAS_ESTIMATED_LOGP ?predicted_logp",
                confidence_adjustment=0.75,
                metadata_template={"prediction_method": "fragment_based"},
            ),
            conditions=[RuleCondition("fragment_coverage", "fragment_coverage > 0.8")],
            confidence_model=ConfidenceModel.FUZZY_LOGIC,
            priority=2,
            enabled=True,
        )
        self.add_rule(logp_prediction_rule)

    def _add_drug_interaction_rules(self) -> None:
        """Add drug interaction inference rules."""

        # Same pathway interaction
        pathway_interaction_rule = AdvancedReasoningRule(
            rule_id="same_pathway_interaction",
            name="Same Pathway Drug Interaction",
            description="Drugs affecting the same pathway may interact",
            rule_type=RuleType.CAUSAL,
            antecedent_patterns=[
                "?drug1 AFFECTS_PATHWAY ?pathway",
                "?drug2 AFFECTS_PATHWAY ?pathway",
                "?pathway HAS_INTERACTION_POTENTIAL ?potential",
            ],
            consequent=RuleConsequent(
                template="?drug1 POTENTIALLY_INTERACTS_WITH ?drug2",
                confidence_adjustment=0.6,
                metadata_template={"interaction_basis": "pathway_overlap"},
            ),
            conditions=[
                RuleCondition("different_drugs", "drug1 != drug2"),
                RuleCondition("interaction_potential", "potential >= 0.5"),
            ],
            confidence_model=ConfidenceModel.BAYESIAN,
            priority=1,
            enabled=True,
        )
        self.add_rule(pathway_interaction_rule)

        # Enzyme interaction rule
        enzyme_interaction_rule = AdvancedReasoningRule(
            rule_id="enzyme_mediated_interaction",
            name="Enzyme-Mediated Drug Interaction",
            description="Drugs metabolized by the same enzyme may interact",
            rule_type=RuleType.CAUSAL,
            antecedent_patterns=[
                "?drug1 METABOLIZED_BY ?enzyme",
                "?drug2 METABOLIZED_BY ?enzyme",
                "?enzyme HAS_CAPACITY_LIMITATION ?limited",
            ],
            consequent=RuleConsequent(
                template="?drug1 COMPETES_WITH ?drug2",
                confidence_adjustment=0.7,
                metadata_template={"interaction_type": "competitive_inhibition"},
            ),
            conditions=[
                RuleCondition("different_drugs", "drug1 != drug2"),
                RuleCondition("enzyme_limited", "limited == true"),
            ],
            confidence_model=ConfidenceModel.MULTIPLICATIVE,
            priority=2,
            enabled=True,
        )
        self.add_rule(enzyme_interaction_rule)

    def _add_metabolic_pathway_rules(self) -> None:
        """Add metabolic pathway reasoning rules."""

        # Pathway transitivity
        pathway_transitivity_rule = AdvancedReasoningRule(
            rule_id="metabolic_pathway_transitivity",
            name="Metabolic Pathway Transitivity",
            description="If A is converted to B and B to C, then A can be converted to C",
            rule_type=RuleType.TRANSITIVE,
            antecedent_patterns=[
                "?compound_a CONVERTED_TO ?compound_b",
                "?compound_b CONVERTED_TO ?compound_c",
            ],
            consequent=RuleConsequent(
                template="?compound_a ULTIMATELY_CONVERTED_TO ?compound_c",
                confidence_adjustment=0.8,
                metadata_template={
                    "pathway_length": "2",
                    "inference_type": "transitivity",
                },
            ),
            conditions=[
                RuleCondition("pathway_efficiency", "conversion_efficiency > 0.3")
            ],
            confidence_model=ConfidenceModel.MULTIPLICATIVE,
            max_inference_depth=5,
            priority=2,
            enabled=True,
        )
        self.add_rule(pathway_transitivity_rule)

    def _add_toxicity_prediction_rules(self) -> None:
        """Add toxicity prediction rules."""

        # Structural alert rule
        structural_alert_rule = AdvancedReasoningRule(
            rule_id="structural_alert_toxicity",
            name="Structural Alert Toxicity Prediction",
            description="Compounds containing toxic structural alerts may be toxic",
            rule_type=RuleType.STRUCTURAL,
            antecedent_patterns=[
                "?compound CONTAINS_SUBSTRUCTURE ?substructure",
                "?substructure IS_TOXIC_ALERT ?alert_type",
            ],
            consequent=RuleConsequent(
                template="?compound HAS_TOXICITY_ALERT ?alert_type",
                confidence_adjustment=0.85,
                metadata_template={"alert_basis": "structural_pattern"},
            ),
            conditions=[
                RuleCondition("alert_severity", "alert_severity >= 'moderate'")
            ],
            confidence_model=ConfidenceModel.EVIDENCE_ACCUMULATION,
            priority=1,
            enabled=True,
        )
        self.add_rule(structural_alert_rule)

    def _add_sar_rules(self) -> None:
        """Add Structure-Activity Relationship rules."""

        # Similar structure, similar activity
        sar_similarity_rule = AdvancedReasoningRule(
            rule_id="sar_structure_activity",
            name="Structure-Activity Relationship",
            description="Structurally similar compounds have similar biological activities",
            rule_type=RuleType.CHEMICAL_SIMILARITY,
            antecedent_patterns=[
                "?compound1 STRUCTURALLY_SIMILAR_TO ?compound2",
                "?compound1 HAS_ACTIVITY ?activity",
                "?activity HAS_POTENCY ?potency",
            ],
            consequent=RuleConsequent(
                template="?compound2 HAS_PREDICTED_ACTIVITY ?activity",
                confidence_adjustment=0.75,
                metadata_template={"prediction_basis": "sar_similarity"},
            ),
            conditions=[
                RuleCondition("structural_similarity", "structural_similarity >= 0.85"),
                RuleCondition("activity_consistency", "activity_consistency > 0.7"),
            ],
            confidence_model=ConfidenceModel.BAYESIAN,
            priority=2,
            enabled=True,
        )
        self.add_rule(sar_similarity_rule)

    def _add_therapeutic_class_rules(self) -> None:
        """Add therapeutic class reasoning rules."""

        # Class membership inference
        therapeutic_class_rule = AdvancedReasoningRule(
            rule_id="therapeutic_class_membership",
            name="Therapeutic Class Membership",
            description="Drugs with similar mechanisms belong to same therapeutic class",
            rule_type=RuleType.SUBSUMPTION,
            antecedent_patterns=[
                "?drug HAS_MECHANISM ?mechanism",
                "?mechanism BELONGS_TO_CLASS ?therapeutic_class",
            ],
            consequent=RuleConsequent(
                template="?drug IS_A ?therapeutic_class",
                confidence_adjustment=0.9,
                metadata_template={"classification_basis": "mechanism"},
            ),
            conditions=[
                RuleCondition("mechanism_specificity", "mechanism_specificity >= 0.8")
            ],
            confidence_model=ConfidenceModel.EVIDENCE_ACCUMULATION,
            priority=3,
            enabled=True,
        )
        self.add_rule(therapeutic_class_rule)


class InferenceEngine:
    """Advanced inference engine with sophisticated reasoning mechanisms."""

    def __init__(
        self, rule_library: ChemicalReasoningRuleLibrary, ontology: ChemicalOntology
    ):
        self.rule_library = rule_library
        self.ontology = ontology
        self.inference_cache = {}
        self.confidence_calculator = ConfidenceCalculator()
        self.fact_validator = FactValidator(ontology)

        # Performance tracking
        self.inference_stats = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "cache_hits": 0,
            "rule_applications": defaultdict(int),
            "average_confidence": 0.0,
        }

    def infer_facts(
        self, facts: List[InferredFact], max_iterations: int = 5
    ) -> List[InferredFact]:
        """Perform iterative inference to derive new facts."""
        all_facts = facts.copy()
        new_facts = []

        for iteration in range(max_iterations):
            iteration_facts = []

            # Apply all applicable rules
            for rule in self.rule_library.get_all_rules():
                if not rule.enabled:
                    continue

                # Find matching fact patterns
                matching_fact_sets = self._find_matching_fact_sets(all_facts, rule)

                for fact_set in matching_fact_sets:
                    inferred_fact = self._apply_rule(rule, fact_set, iteration)

                    if inferred_fact and self._validate_inference(inferred_fact, rule):
                        iteration_facts.append(inferred_fact)
                        self.inference_stats["rule_applications"][rule.rule_id] += 1

            # Remove duplicates and add to collections
            unique_facts = self._deduplicate_facts(iteration_facts)
            new_facts.extend(unique_facts)
            all_facts.extend(unique_facts)

            # Stop if no new facts were derived
            if not unique_facts:
                logging.info(f"Inference converged after {iteration + 1} iterations")
                break

        # Update statistics
        self.inference_stats["total_inferences"] += len(new_facts)
        self.inference_stats["successful_inferences"] += len(new_facts)

        return new_facts

    def explain_inference_chain(self, fact: InferredFact) -> Dict[str, Any]:
        """Provide detailed explanation of inference chain."""
        explanation = {
            "target_fact": f"{fact.subject} {fact.predicate} {fact.object}",
            "final_confidence": fact.confidence,
            "inference_chain": [],
            "supporting_evidence": [],
            "rule_justification": {},
            "confidence_breakdown": {},
        }

        # Trace back through reasoning path
        for step in fact.reasoning_path:
            if step.startswith("rule_"):
                rule_id = step.split("_", 1)[1]
                rule = self.rule_library.get_rule(rule_id)
                if rule:
                    explanation["inference_chain"].append(
                        {
                            "step": step,
                            "rule_name": rule.name,
                            "rule_type": rule.rule_type.value,
                            "description": rule.description,
                        }
                    )

        # Add source facts as evidence
        for source in fact.source_facts:
            explanation["supporting_evidence"].append(
                {"fact": source, "role": "antecedent"}
            )

        # Add rule justification
        rule = self.rule_library.get_rule(fact.rule_applied)
        if rule:
            explanation["rule_justification"] = {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "confidence_model": rule.confidence_model.value,
                "conditions_met": self._check_rule_conditions(rule, fact),
            }

        return explanation

    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get comprehensive inference statistics."""
        stats = self.inference_stats.copy()

        # Add rule performance statistics
        rule_performance = {}
        for rule in self.rule_library.get_all_rules():
            rule_performance[rule.rule_id] = {
                "applications": rule.application_count,
                "success_rate": rule.success_count / max(rule.application_count, 1),
                "average_confidence": rule.average_confidence,
                "last_applied": (
                    rule.last_applied.isoformat() if rule.last_applied else None
                ),
            }

        stats["rule_performance"] = rule_performance

        # Add cache statistics
        stats["cache_size"] = len(self.inference_cache)
        stats["cache_hit_rate"] = stats["cache_hits"] / max(
            stats["total_inferences"], 1
        )

        return stats

    def _find_matching_fact_sets(
        self, facts: List[InferredFact], rule: AdvancedReasoningRule
    ) -> List[List[InferredFact]]:
        """Find sets of facts that match rule antecedent patterns."""
        matching_sets = []

        if rule.rule_type == RuleType.TRANSITIVE:
            matching_sets.extend(self._find_transitive_matches(facts, rule))
        elif rule.rule_type == RuleType.CHEMICAL_SIMILARITY:
            matching_sets.extend(self._find_similarity_matches(facts, rule))
        elif rule.rule_type == RuleType.STRUCTURAL:
            matching_sets.extend(self._find_structural_matches(facts, rule))
        elif rule.rule_type == RuleType.CAUSAL:
            matching_sets.extend(self._find_causal_matches(facts, rule))
        else:
            matching_sets.extend(self._find_generic_matches(facts, rule))

        return matching_sets

    def _find_transitive_matches(
        self, facts: List[InferredFact], rule: AdvancedReasoningRule
    ) -> List[List[InferredFact]]:
        """Find facts matching transitive rule patterns."""
        matches = []

        # Look for A->B, B->C patterns
        for fact1 in facts:
            for fact2 in facts:
                if (
                    fact1.object == fact2.subject
                    and fact1.predicate == fact2.predicate
                    and fact1.confidence >= rule.min_evidence_strength
                    and fact2.confidence >= rule.min_evidence_strength
                ):
                    matches.append([fact1, fact2])

        return matches

    def _find_similarity_matches(
        self, facts: List[InferredFact], rule: AdvancedReasoningRule
    ) -> List[List[InferredFact]]:
        """Find facts matching similarity rule patterns."""
        matches = []

        # Look for similarity + property patterns
        similarity_facts = [f for f in facts if "SIMILAR" in f.predicate]
        property_facts = [
            f for f in facts if "HAS_PROPERTY" in f.predicate or "TREATS" in f.predicate
        ]

        for sim_fact in similarity_facts:
            for prop_fact in property_facts:
                if (
                    sim_fact.subject == prop_fact.subject
                    and sim_fact.confidence >= 0.7
                    and prop_fact.confidence >= rule.min_evidence_strength
                ):
                    matches.append([sim_fact, prop_fact])

        return matches

    def _find_structural_matches(
        self, facts: List[InferredFact], rule: AdvancedReasoningRule
    ) -> List[List[InferredFact]]:
        """Find facts matching structural rule patterns."""
        matches = []

        # Look for structure-based patterns
        structure_facts = [
            f
            for f in facts
            if any(
                keyword in f.predicate
                for keyword in [
                    "HAS_STRUCTURE",
                    "CONTAINS_FRAGMENT",
                    "CONTAINS_SUBSTRUCTURE",
                ]
            )
        ]

        for struct_fact in structure_facts:
            # Find related property facts
            related_facts = [
                f
                for f in facts
                if f.subject == struct_fact.subject and f != struct_fact
            ]

            if related_facts:
                matches.append(
                    [struct_fact] + related_facts[:2]
                )  # Limit to avoid combinatorial explosion

        return matches

    def _find_causal_matches(
        self, facts: List[InferredFact], rule: AdvancedReasoningRule
    ) -> List[List[InferredFact]]:
        """Find facts matching causal rule patterns."""
        matches = []

        # Look for causal relationship patterns
        causal_predicates = ["AFFECTS_PATHWAY", "METABOLIZED_BY", "CAUSES", "LEADS_TO"]

        for predicate in causal_predicates:
            causal_facts = [f for f in facts if predicate in f.predicate]

            # Group by object (common pathway, enzyme, etc.)
            grouped = defaultdict(list)
            for fact in causal_facts:
                grouped[fact.object].append(fact)

            # Find groups with multiple subjects
            for group in grouped.values():
                if len(group) >= 2:
                    matches.append(group[:3])  # Limit group size

        return matches

    def _find_generic_matches(
        self, facts: List[InferredFact], rule: AdvancedReasoningRule
    ) -> List[List[InferredFact]]:
        """Find facts matching generic rule patterns."""
        matches = []

        # Simple pattern matching based on rule antecedent patterns
        for pattern in rule.antecedent_patterns:
            # Extract key predicates from pattern
            predicates = re.findall(r"(\w+(?:_\w+)*)", pattern)

            # Find facts with matching predicates
            matching_facts = []
            for predicate in predicates:
                facts_with_predicate = [f for f in facts if predicate in f.predicate]
                if facts_with_predicate:
                    matching_facts.extend(
                        facts_with_predicate[:2]
                    )  # Limit per predicate

            if len(matching_facts) >= 2:
                matches.append(matching_facts[:3])  # Limit combination size

        return matches

    def _apply_rule(
        self, rule: AdvancedReasoningRule, fact_set: List[InferredFact], iteration: int
    ) -> Optional[InferredFact]:
        """Apply reasoning rule to derive new fact."""
        if not fact_set:
            return None

        # Update rule statistics
        rule.application_count += 1
        rule.last_applied = datetime.now()

        # Check conditions
        if not self._check_rule_conditions(rule, fact_set):
            return None

        # Calculate confidence
        confidence = self.confidence_calculator.calculate_rule_confidence(
            rule, fact_set, rule.confidence_model
        )

        if confidence < rule.min_evidence_strength:
            return None

        # Generate consequent
        consequent = self._generate_consequent(rule, fact_set)
        if not consequent:
            return None

        # Create inferred fact
        inferred_fact = InferredFact(
            subject=consequent.get("subject", ""),
            predicate=consequent.get("predicate", ""),
            object=consequent.get("object", ""),
            confidence=confidence,
            reasoning_path=[f"rule_{rule.rule_id}_iteration_{iteration}"],
            source_facts=[f"{f.subject} {f.predicate} {f.object}" for f in fact_set],
            inferred_at=datetime.now(),
            rule_applied=rule.rule_id,
            metadata={
                "rule_type": rule.rule_type.value,
                "iteration": iteration,
                "source_fact_count": len(fact_set),
            },
        )

        # Update rule success statistics
        rule.success_count += 1
        rule.average_confidence = (
            rule.average_confidence * (rule.success_count - 1) + confidence
        ) / rule.success_count

        return inferred_fact

    def _check_rule_conditions(
        self, rule: AdvancedReasoningRule, fact_set: List[InferredFact]
    ) -> bool:
        """Check if rule conditions are satisfied."""
        for condition in rule.conditions:
            if not self._evaluate_condition(condition, fact_set):
                return False
        return True

    def _evaluate_condition(
        self, condition: RuleCondition, fact_set: List[InferredFact]
    ) -> bool:
        """Evaluate individual rule condition."""
        try:
            if condition.condition_type == "similarity_threshold":
                # Check similarity scores in facts
                for fact in fact_set:
                    if "SIMILAR" in fact.predicate:
                        similarity_score = fact.metadata.get(
                            "similarity_score", fact.confidence
                        )
                        return similarity_score >= float(
                            condition.expression.split(">=")[1].strip()
                        )

            elif condition.condition_type == "different_drugs":
                # Ensure facts refer to different entities
                subjects = {fact.subject for fact in fact_set}
                return len(subjects) > 1

            elif condition.condition_type == "property_type":
                # Check property types
                property_types = condition.parameters.get("allowed_types", [])
                return any(
                    fact.metadata.get("property_type") in property_types
                    for fact in fact_set
                )

            # Default: return True for unknown conditions
            return True

        except Exception as e:
            logging.warning(
                f"Error evaluating condition {condition.condition_type}: {e}"
            )
            return False

    def _generate_consequent(
        self, rule: AdvancedReasoningRule, fact_set: List[InferredFact]
    ) -> Optional[Dict[str, str]]:
        """Generate consequent from rule template and facts."""
        template = rule.consequent.template

        # Simple template substitution based on rule type
        if rule.rule_type == RuleType.TRANSITIVE and len(fact_set) >= 2:
            return {
                "subject": fact_set[0].subject,
                "predicate": self._derive_transitive_predicate(fact_set[0].predicate),
                "object": fact_set[1].object,
            }

        elif rule.rule_type == RuleType.CHEMICAL_SIMILARITY and len(fact_set) >= 2:
            similarity_fact = next(
                (f for f in fact_set if "SIMILAR" in f.predicate), None
            )
            property_fact = next((f for f in fact_set if f != similarity_fact), None)

            if similarity_fact and property_fact:
                return {
                    "subject": similarity_fact.object,  # Similar entity
                    "predicate": property_fact.predicate,
                    "object": property_fact.object,
                }

        elif rule.rule_type == RuleType.CAUSAL and len(fact_set) >= 2:
            # For causal rules, derive interaction between subjects
            if len(fact_set) >= 2:
                return {
                    "subject": fact_set[0].subject,
                    "predicate": "POTENTIALLY_INTERACTS_WITH",
                    "object": fact_set[1].subject,
                }

        return None

    def _derive_transitive_predicate(self, original_predicate: str) -> str:
        """Derive transitive form of predicate."""
        transitive_mappings = {
            "CONVERTED_TO": "ULTIMATELY_CONVERTED_TO",
            "LEADS_TO": "ULTIMATELY_LEADS_TO",
            "CAUSES": "INDIRECTLY_CAUSES",
        }
        return transitive_mappings.get(
            original_predicate, f"TRANSITIVELY_{original_predicate}"
        )

    def _validate_inference(
        self, fact: InferredFact, rule: AdvancedReasoningRule
    ) -> bool:
        """Validate inferred fact."""
        return self.fact_validator.validate_fact(fact, rule)

    def _deduplicate_facts(self, facts: List[InferredFact]) -> List[InferredFact]:
        """Remove duplicate facts, keeping highest confidence."""
        fact_map = {}

        for fact in facts:
            key = (fact.subject, fact.predicate, fact.object)

            if key not in fact_map or fact.confidence > fact_map[key].confidence:
                fact_map[key] = fact

        return list(fact_map.values())


class ConfidenceCalculator:
    """Advanced confidence calculation for inferred facts."""

    def calculate_rule_confidence(
        self,
        rule: AdvancedReasoningRule,
        fact_set: List[InferredFact],
        model: ConfidenceModel,
    ) -> float:
        """Calculate confidence for rule application."""

        if model == ConfidenceModel.MULTIPLICATIVE:
            return self._multiplicative_confidence(rule, fact_set)
        elif model == ConfidenceModel.BAYESIAN:
            return self._bayesian_confidence(rule, fact_set)
        elif model == ConfidenceModel.FUZZY_LOGIC:
            return self._fuzzy_logic_confidence(rule, fact_set)
        elif model == ConfidenceModel.EVIDENCE_ACCUMULATION:
            return self._evidence_accumulation_confidence(rule, fact_set)
        else:
            return self._multiplicative_confidence(rule, fact_set)

    def _multiplicative_confidence(
        self, rule: AdvancedReasoningRule, fact_set: List[InferredFact]
    ) -> float:
        """Multiplicative confidence model."""
        base_confidence = 1.0
        for fact in fact_set:
            base_confidence *= fact.confidence

        # Apply rule confidence adjustment
        return min(
            base_confidence
            * rule.confidence_boost
            * rule.consequent.confidence_adjustment,
            1.0,
        )

    def _bayesian_confidence(
        self, rule: AdvancedReasoningRule, fact_set: List[InferredFact]
    ) -> float:
        """Bayesian confidence model."""
        # Simplified Bayesian calculation
        prior = rule.confidence_boost
        likelihood = sum(fact.confidence for fact in fact_set) / len(fact_set)

        # Bayesian update
        posterior = (likelihood * prior) / (
            likelihood * prior + (1 - likelihood) * (1 - prior)
        )

        return min(posterior * rule.consequent.confidence_adjustment, 1.0)

    def _fuzzy_logic_confidence(
        self, rule: AdvancedReasoningRule, fact_set: List[InferredFact]
    ) -> float:
        """Fuzzy logic confidence model."""
        # Use minimum for conjunction (AND operation)
        min_confidence = min(fact.confidence for fact in fact_set)

        # Apply fuzzy implication
        implication_strength = rule.confidence_boost
        result = min(min_confidence, implication_strength)

        return min(result * rule.consequent.confidence_adjustment, 1.0)

    def _evidence_accumulation_confidence(
        self, rule: AdvancedReasoningRule, fact_set: List[InferredFact]
    ) -> float:
        """Evidence accumulation confidence model."""
        # Combine evidence using Dempster-Shafer theory approximation
        combined_confidence = 0.0
        combined_uncertainty = 1.0

        for fact in fact_set:
            evidence = fact.confidence
            uncertainty = 1.0 - evidence

            # Combine with existing evidence
            new_combined = combined_confidence + evidence * combined_uncertainty
            new_uncertainty = combined_uncertainty * uncertainty

            combined_confidence = new_combined
            combined_uncertainty = new_uncertainty

        # Normalize and apply rule adjustment
        final_confidence = combined_confidence / (
            combined_confidence + combined_uncertainty
        )

        return min(
            final_confidence
            * rule.confidence_boost
            * rule.consequent.confidence_adjustment,
            1.0,
        )


class FactValidator:
    """Validates inferred facts against ontology and logical constraints."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.contradiction_rules = self._load_contradiction_rules()

    def validate_fact(self, fact: InferredFact, rule: AdvancedReasoningRule) -> bool:
        """Comprehensive fact validation."""

        # Basic structure validation
        if not self._validate_structure(fact):
            return False

        # Ontology consistency validation
        if not self._validate_ontology_consistency(fact):
            return False

        # Contradiction checking
        if self._check_contradictions(fact):
            return False

        # Rule-specific validation
        if not self._validate_rule_specific(fact, rule):
            return False

        return True

    def _validate_structure(self, fact: InferredFact) -> bool:
        """Validate basic fact structure."""
        return (
            fact.subject
            and fact.predicate
            and fact.object
            and 0.0 <= fact.confidence <= 1.0
        )

    def _validate_ontology_consistency(self, fact: InferredFact) -> bool:
        """Validate fact against ontology constraints."""
        # Check if predicate is valid for subject/object types
        try:
            subject_entity = self.ontology.find_entity(fact.subject)
            object_entity = self.ontology.find_entity(fact.object)

            if subject_entity and object_entity:
                # Check if relationship is valid between entity types
                return self.ontology.is_valid_relationship(
                    subject_entity[0].entity_type,
                    fact.predicate,
                    object_entity[0].entity_type,
                )

            return True  # Allow if entities not found in ontology

        except Exception as e:
            logging.warning(f"Ontology validation error: {e}")
            return True  # Default to valid if validation fails

    def _check_contradictions(self, fact: InferredFact) -> bool:
        """Check for logical contradictions."""
        # Check against contradiction rules
        for rule in self.contradiction_rules:
            if self._matches_contradiction_pattern(fact, rule):
                return True  # Contradiction found

        return False

    def _validate_rule_specific(
        self, fact: InferredFact, rule: AdvancedReasoningRule
    ) -> bool:
        """Rule-specific validation."""
        if rule.rule_type == RuleType.CHEMICAL_SIMILARITY:
            # Validate chemical similarity inferences
            return self._validate_chemical_similarity(fact)

        elif rule.rule_type == RuleType.STRUCTURAL:
            # Validate structural inferences
            return self._validate_structural_inference(fact)

        return True

    def _validate_chemical_similarity(self, fact: InferredFact) -> bool:
        """Validate chemical similarity-based inferences."""
        # Check if similarity threshold is reasonable
        if "similarity" in fact.metadata:
            return fact.metadata["similarity"] >= 0.5

        return True

    def _validate_structural_inference(self, fact: InferredFact) -> bool:
        """Validate structure-based inferences."""
        # Ensure structural inferences are chemically reasonable
        if "HAS_MOLECULAR_WEIGHT" in fact.predicate:
            try:
                mw = float(fact.object)
                return 10.0 <= mw <= 10000.0  # Reasonable molecular weight range
            except ValueError:
                return False

        return True

    def _load_contradiction_rules(self) -> List[Dict[str, Any]]:
        """Load logical contradiction rules."""
        return [
            {
                "pattern": "?x TREATS ?y AND ?x CAUSES ?y",
                "description": "Entity cannot both treat and cause the same condition",
            },
            {
                "pattern": "?x HAS_MOLECULAR_WEIGHT ?w1 AND ?x HAS_MOLECULAR_WEIGHT ?w2 WHERE w1 != w2",
                "description": "Entity cannot have multiple different molecular weights",
            },
        ]

    def _matches_contradiction_pattern(
        self, fact: InferredFact, contradiction_rule: Dict[str, Any]
    ) -> bool:
        """Check if fact matches contradiction pattern."""
        # Simplified pattern matching
        pattern = contradiction_rule["pattern"]

        if "TREATS" in fact.predicate and "CAUSES" in pattern:
            # Would need to check against existing facts
            return False  # Simplified for now

        return False

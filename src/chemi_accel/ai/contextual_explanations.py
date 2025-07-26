"""
Contextual Explanation System for ChemiAccel AI-KG Integration

This module provides intelligent explanations for AI predictions by leveraging
contextual information from the Integrated Chemistry Knowledge Graph (IC-KG).
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from .kg_integration import (
        AIContextualPrediction,
        AITaskType,
        KGContext,
        KGContextType,
    )

    KG_INTEGRATION_AVAILABLE = True
except ImportError:
    KG_INTEGRATION_AVAILABLE = False
    logging.warning(
        "KG Integration not available. Explanation features will be limited."
    )


class ExplanationType(Enum):
    """Types of explanations that can be generated."""

    PREDICTION_RATIONALE = "prediction_rationale"
    EVIDENCE_SUMMARY = "evidence_summary"
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"
    RECOMMENDATION_BASIS = "recommendation_basis"
    RISK_ASSESSMENT = "risk_assessment"
    NOVELTY_ANALYSIS = "novelty_analysis"
    SIMILARITY_EXPLANATION = "similarity_explanation"
    PROPERTY_CORRELATION = "property_correlation"


class ExplanationDetail(Enum):
    """Levels of explanation detail."""

    BRIEF = "brief"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    TECHNICAL = "technical"


@dataclass
class ExplanationComponent:
    """Individual component of an explanation."""

    title: str
    content: str
    evidence_strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    source_type: str  # "kg_data", "model_prediction", "literature", etc.
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    visual_aids: List[str] = field(default_factory=list)  # URLs or descriptions


@dataclass
class ContextualExplanation:
    """Complete contextual explanation for an AI prediction."""

    prediction_summary: str
    confidence_level: str  # "high", "medium", "low"
    main_rationale: str
    components: List[ExplanationComponent] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExplanationGenerator(ABC):
    """Abstract base class for explanation generators."""

    @abstractmethod
    def generate_explanation(
        self,
        prediction: AIContextualPrediction,
        explanation_types: List[ExplanationType],
        detail_level: ExplanationDetail,
    ) -> ContextualExplanation:
        """Generate contextual explanation for a prediction."""
        pass


class ChemicalExplanationGenerator(ExplanationGenerator):
    """Chemical domain-specific explanation generator."""

    def __init__(self):
        """Initialize chemical explanation generator."""
        self.molecular_property_descriptions = {
            "molecular_weight": "The molecular weight affects drug absorption, distribution, and clearance.",
            "logp": "LogP represents lipophilicity, influencing membrane permeability and bioavailability.",
            "tpsa": "Topological Polar Surface Area affects blood-brain barrier penetration and oral absorption.",
            "hbd": "Hydrogen bond donors influence molecular interactions and solubility.",
            "hba": "Hydrogen bond acceptors affect binding affinity and pharmacokinetics.",
            "rotatable_bonds": "Rotatable bonds indicate molecular flexibility and binding entropy costs.",
        }

        self.activity_type_descriptions = {
            "ic50": "IC50 represents the concentration required to inhibit 50% of target activity.",
            "ki": "Ki is the inhibition constant, indicating binding affinity to the target.",
            "ec50": "EC50 is the concentration producing 50% of maximal response.",
            "binding_affinity": "Binding affinity measures the strength of molecular target interaction.",
        }

        self.risk_factors = {
            "high_lipophilicity": "Very high LogP (>5) may cause poor aqueous solubility and toxicity.",
            "high_molecular_weight": "High MW (>500 Da) may reduce oral bioavailability (Lipinski's rule).",
            "low_selectivity": "Multiple target interactions may increase side effect risk.",
            "novel_scaffold": "Novel chemical scaffolds may have unpredictable properties.",
            "reactive_groups": "Reactive functional groups may cause off-target effects.",
        }

    def generate_explanation(
        self,
        prediction: AIContextualPrediction,
        explanation_types: List[ExplanationType],
        detail_level: ExplanationDetail,
    ) -> ContextualExplanation:
        """Generate chemical domain explanation."""
        components = []

        # Generate components for each requested explanation type
        for exp_type in explanation_types:
            if exp_type == ExplanationType.PREDICTION_RATIONALE:
                components.append(
                    self._generate_prediction_rationale(prediction, detail_level)
                )
            elif exp_type == ExplanationType.EVIDENCE_SUMMARY:
                components.append(
                    self._generate_evidence_summary(prediction, detail_level)
                )
            elif exp_type == ExplanationType.UNCERTAINTY_ANALYSIS:
                components.append(
                    self._generate_uncertainty_analysis(prediction, detail_level)
                )
            elif exp_type == ExplanationType.RECOMMENDATION_BASIS:
                components.append(
                    self._generate_recommendation_basis(prediction, detail_level)
                )
            elif exp_type == ExplanationType.RISK_ASSESSMENT:
                components.append(
                    self._generate_risk_assessment(prediction, detail_level)
                )
            elif exp_type == ExplanationType.NOVELTY_ANALYSIS:
                components.append(
                    self._generate_novelty_analysis(prediction, detail_level)
                )
            elif exp_type == ExplanationType.SIMILARITY_EXPLANATION:
                components.append(
                    self._generate_similarity_explanation(prediction, detail_level)
                )
            elif exp_type == ExplanationType.PROPERTY_CORRELATION:
                components.append(
                    self._generate_property_correlation(prediction, detail_level)
                )

        # Generate overall summary and insights
        prediction_summary = self._generate_prediction_summary(prediction)
        confidence_level = self._assess_confidence_level(prediction.confidence)
        main_rationale = prediction.explanation
        key_insights = self._extract_key_insights(prediction, components)
        limitations = self._identify_limitations(prediction)
        recommendations = prediction.recommendations

        return ContextualExplanation(
            prediction_summary=prediction_summary,
            confidence_level=confidence_level,
            main_rationale=main_rationale,
            components=components,
            key_insights=key_insights,
            limitations=limitations,
            recommendations=recommendations,
            references=self._generate_references(prediction),
            metadata={
                "generated_at": datetime.now().isoformat(),
                "detail_level": detail_level.value,
                "explanation_types": [t.value for t in explanation_types],
                "kg_context_types": list(
                    prediction.kg_context.metadata.get("context_types", [])
                ),
                "confidence_score": prediction.confidence,
            },
        )

    def _generate_prediction_rationale(
        self, prediction: AIContextualPrediction, detail_level: ExplanationDetail
    ) -> ExplanationComponent:
        """Generate rationale for the prediction."""
        content_parts = []

        # Base AI model rationale
        if hasattr(prediction.prediction, "model_rationale"):
            content_parts.append(
                f"AI Model Analysis: {prediction.prediction['model_rationale']}"
            )

        # Knowledge graph influence
        kg_entities = len(prediction.kg_context.entities)
        if kg_entities > 0:
            content_parts.append(
                f"Knowledge Graph Context: Leveraged information from {kg_entities} related entities "
                f"in the chemical knowledge base to inform this prediction."
            )

        # Supporting evidence count
        evidence_count = len(prediction.supporting_evidence)
        if evidence_count > 0:
            content_parts.append(
                f"Supporting Evidence: Found {evidence_count} pieces of supporting evidence "
                f"from literature and experimental data."
            )

        # Similarity to known compounds
        similar_compounds = [
            e
            for e in prediction.kg_context.entities
            if "similarity_score" in e and e["similarity_score"] > 0.7
        ]
        if similar_compounds:
            max_sim = max(e["similarity_score"] for e in similar_compounds)
            content_parts.append(
                f"Molecular Similarity: Found highly similar compounds (max similarity: {max_sim:.2f}) "
                f"with known properties, increasing prediction reliability."
            )

        if detail_level in [
            ExplanationDetail.DETAILED,
            ExplanationDetail.COMPREHENSIVE,
        ]:
            # Add more detailed analysis
            if prediction.kg_context.relationships:
                rel_types = set(r["type"] for r in prediction.kg_context.relationships)
                content_parts.append(
                    f"Relationship Analysis: Identified {len(rel_types)} types of molecular relationships "
                    f"({', '.join(rel_types)}) that inform this prediction."
                )

        content = (
            " ".join(content_parts)
            if content_parts
            else "Prediction based on trained AI model analysis."
        )

        return ExplanationComponent(
            title="Prediction Rationale",
            content=content,
            evidence_strength=min(1.0, evidence_count / 10.0),
            confidence=prediction.confidence,
            source_type="model_prediction",
        )

    def _generate_evidence_summary(
        self, prediction: AIContextualPrediction, detail_level: ExplanationDetail
    ) -> ExplanationComponent:
        """Generate summary of supporting evidence."""
        content_parts = []
        evidence_strength = 0.0

        # Supporting evidence
        if prediction.supporting_evidence:
            content_parts.append("Supporting Evidence:")
            for i, evidence in enumerate(prediction.supporting_evidence[:5]):  # Top 5
                evidence_type = evidence.get("type", "unknown")
                evidence_text = evidence.get("evidence", "No description")
                confidence = evidence.get("confidence", 0.0)

                if detail_level == ExplanationDetail.BRIEF:
                    content_parts.append(f"• {evidence_text}")
                else:
                    content_parts.append(
                        f"• {evidence_text} (Type: {evidence_type}, Confidence: {confidence:.2f})"
                    )

                evidence_strength += confidence

        # Contradicting evidence
        if prediction.contradicting_evidence:
            content_parts.append("\nContradicting Evidence:")
            for evidence in prediction.contradicting_evidence[:3]:  # Top 3
                evidence_text = evidence.get("evidence", "No description")
                content_parts.append(f"• {evidence_text}")

        # Knowledge graph entities with activities
        active_entities = [
            e
            for e in prediction.kg_context.entities
            if "activity_type" in e and e.get("confidence", 0) > 0.5
        ]
        if active_entities and detail_level != ExplanationDetail.BRIEF:
            content_parts.append(
                f"\nKnown Activities: Found {len(active_entities)} related activities:"
            )
            for entity in active_entities[:3]:
                activity = entity.get("activity_type", "unknown")
                value = entity.get("value", "N/A")
                content_parts.append(f"• {activity}: {value}")

        if prediction.supporting_evidence:
            evidence_strength = evidence_strength / len(prediction.supporting_evidence)

        content = (
            "\n".join(content_parts)
            if content_parts
            else "Limited evidence available in knowledge base."
        )

        return ExplanationComponent(
            title="Evidence Summary",
            content=content,
            evidence_strength=evidence_strength,
            confidence=0.8 if prediction.supporting_evidence else 0.3,
            source_type="kg_data",
        )

    def _generate_uncertainty_analysis(
        self, prediction: AIContextualPrediction, detail_level: ExplanationDetail
    ) -> ExplanationComponent:
        """Generate analysis of prediction uncertainty."""
        content_parts = []

        # Overall confidence assessment
        conf_level = self._assess_confidence_level(prediction.confidence)
        content_parts.append(
            f"Confidence Level: {conf_level.title()} ({prediction.confidence:.2f})"
        )

        # Uncertainty factors
        if prediction.uncertainty_factors:
            content_parts.append("\nUncertainty Factors:")
            for factor in prediction.uncertainty_factors:
                content_parts.append(f"• {factor}")

        # Knowledge gap analysis
        kg_coverage = len(prediction.kg_context.entities)
        if kg_coverage < 5:
            content_parts.append(
                f"\nLimited Knowledge Coverage: Only {kg_coverage} related entities found in knowledge base, "
                f"indicating potential knowledge gaps for this compound class."
            )

        # Prediction consistency
        if hasattr(prediction.prediction, "model_ensemble_std"):
            ensemble_std = prediction.prediction["model_ensemble_std"]
            if ensemble_std > 0.2:
                content_parts.append(
                    f"\nModel Disagreement: High standard deviation ({ensemble_std:.3f}) across "
                    f"ensemble models suggests higher uncertainty."
                )

        # Extrapolation warning
        similar_compounds = [
            e for e in prediction.kg_context.entities if "similarity_score" in e
        ]
        if similar_compounds:
            max_similarity = max(e["similarity_score"] for e in similar_compounds)
            if max_similarity < 0.6:
                content_parts.append(
                    "\nExtrapolation Warning: Low similarity to known compounds may increase prediction uncertainty."
                )

        content = "\n".join(content_parts)

        return ExplanationComponent(
            title="Uncertainty Analysis",
            content=content,
            evidence_strength=1.0 - prediction.confidence,  # Inverse relationship
            confidence=0.9,  # High confidence in uncertainty analysis
            source_type="model_prediction",
        )

    def _generate_recommendation_basis(
        self, prediction: AIContextualPrediction, detail_level: ExplanationDetail
    ) -> ExplanationComponent:
        """Generate basis for recommendations."""
        content_parts = []

        if prediction.recommendations:
            content_parts.append("Recommendation Basis:")

            for i, recommendation in enumerate(prediction.recommendations):
                content_parts.append(f"{i+1}. {recommendation}")

                # Add basis for each recommendation if detailed
                if detail_level in [
                    ExplanationDetail.DETAILED,
                    ExplanationDetail.COMPREHENSIVE,
                ]:
                    basis = self._explain_recommendation_basis(
                        recommendation, prediction
                    )
                    if basis:
                        content_parts.append(f"   Basis: {basis}")
        else:
            content_parts.append(
                "No specific recommendations generated for this prediction."
            )

        content = "\n".join(content_parts)

        return ExplanationComponent(
            title="Recommendation Basis",
            content=content,
            evidence_strength=0.7,
            confidence=0.8,
            source_type="model_prediction",
        )

    def _generate_risk_assessment(
        self, prediction: AIContextualPrediction, detail_level: ExplanationDetail
    ) -> ExplanationComponent:
        """Generate risk assessment based on known information."""
        content_parts = []
        identified_risks = []

        # Check for adverse effects in KG
        adverse_effects = [
            e
            for e in prediction.kg_context.entities
            if "effect_type" in e or "adverse_effect" in str(e).lower()
        ]

        if adverse_effects:
            content_parts.append("Known Risk Factors:")
            for effect in adverse_effects[:3]:
                effect_type = effect.get(
                    "effect_type", effect.get("type", "Unknown effect")
                )
                severity = effect.get("severity", "Unknown severity")
                content_parts.append(f"• {effect_type} (Severity: {severity})")
                identified_risks.append(effect_type)

        # Structural risk assessment
        if (
            isinstance(prediction.prediction, dict)
            and "molecules" in prediction.prediction
        ):
            mol_data = (
                prediction.prediction["molecules"][0]
                if prediction.prediction["molecules"]
                else {}
            )
            properties = mol_data.get("properties", {})

            # Check property-based risks
            structural_risks = []

            if properties.get("logp", 0) > 5:
                structural_risks.append("high_lipophilicity")
            if properties.get("molecular_weight", 0) > 500:
                structural_risks.append("high_molecular_weight")
            if properties.get("tpsa", 0) > 140:
                structural_risks.append("high_polar_surface_area")

            if structural_risks:
                content_parts.append("\nStructural Risk Factors:")
                for risk in structural_risks:
                    description = self.risk_factors.get(risk, f"Potential risk: {risk}")
                    content_parts.append(f"• {description}")
                    identified_risks.extend(structural_risks)

        # Target selectivity risks
        target_rels = [
            r for r in prediction.kg_context.relationships if r["type"] == "TARGETS"
        ]
        if len(target_rels) > 5:
            content_parts.append(
                f"\nSelectivity Concern: {len(target_rels)} target interactions identified, "
                f"which may increase side effect risk."
            )
            identified_risks.append("low_selectivity")

        # Overall risk assessment
        if not identified_risks:
            content_parts.append(
                "No major risk factors identified based on available data."
            )
            risk_level = "Low"
        elif len(identified_risks) <= 2:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        content_parts.insert(0, f"Overall Risk Level: {risk_level}")

        content = "\n".join(content_parts)

        return ExplanationComponent(
            title="Risk Assessment",
            content=content,
            evidence_strength=len(identified_risks) / 10.0,
            confidence=0.7,
            source_type="kg_data",
            supporting_data={
                "identified_risks": identified_risks,
                "risk_level": risk_level,
            },
        )

    def _generate_novelty_analysis(
        self, prediction: AIContextualPrediction, detail_level: ExplanationDetail
    ) -> ExplanationComponent:
        """Generate analysis of molecular novelty."""
        content_parts = []

        # Similarity analysis
        similar_compounds = [
            e for e in prediction.kg_context.entities if "similarity_score" in e
        ]

        if similar_compounds:
            similarities = [e["similarity_score"] for e in similar_compounds]
            max_sim = max(similarities)
            avg_sim = np.mean(similarities)

            if max_sim > 0.9:
                novelty_level = "Low"
                content_parts.append(
                    f"Low Novelty: Highly similar compounds exist (max similarity: {max_sim:.2f})"
                )
            elif max_sim > 0.7:
                novelty_level = "Moderate"
                content_parts.append(
                    f"Moderate Novelty: Similar compounds exist (max similarity: {max_sim:.2f})"
                )
            else:
                novelty_level = "High"
                content_parts.append(
                    f"High Novelty: Low similarity to known compounds (max similarity: {max_sim:.2f})"
                )

            if detail_level != ExplanationDetail.BRIEF:
                content_parts.append(
                    f"Average similarity to {len(similar_compounds)} related compounds: {avg_sim:.2f}"
                )
        else:
            novelty_level = "Very High"
            content_parts.append(
                "Very High Novelty: No similar compounds found in knowledge base"
            )

        # Chemical class analysis
        class_entities = [
            e
            for e in prediction.kg_context.entities
            if "class_name" in e or "chemical_class" in str(e).lower()
        ]

        if class_entities:
            classes = [
                e.get("class_name", e.get("name", "Unknown")) for e in class_entities
            ]
            content_parts.append(f"Chemical Classes: {', '.join(classes[:3])}")
        else:
            content_parts.append(
                "Novel Chemical Scaffold: No established chemical class identified"
            )

        # Literature coverage
        if hasattr(prediction.kg_context, "literature_references"):
            ref_count = len(prediction.kg_context.literature_references)
            if ref_count == 0:
                content_parts.append(
                    "Limited Literature: No literature references found for this compound class"
                )
            else:
                content_parts.append(
                    f"Literature Coverage: {ref_count} relevant publications identified"
                )

        content = "\n".join(content_parts)

        return ExplanationComponent(
            title="Novelty Analysis",
            content=content,
            evidence_strength=1.0 - (max(similarities) if similar_compounds else 0.0),
            confidence=0.8,
            source_type="kg_data",
            supporting_data={"novelty_level": novelty_level},
        )

    def _generate_similarity_explanation(
        self, prediction: AIContextualPrediction, detail_level: ExplanationDetail
    ) -> ExplanationComponent:
        """Generate explanation of molecular similarities."""
        content_parts = []

        similar_compounds = [
            e
            for e in prediction.kg_context.entities
            if "similarity_score" in e and e["similarity_score"] > 0.5
        ]

        if similar_compounds:
            # Sort by similarity
            similar_compounds.sort(key=lambda x: x["similarity_score"], reverse=True)

            content_parts.append(f"Found {len(similar_compounds)} similar compounds:")

            for i, compound in enumerate(similar_compounds[:5]):  # Top 5
                name = compound.get("name", compound.get("smiles", "Unknown"))
                similarity = compound["similarity_score"]

                if detail_level == ExplanationDetail.BRIEF:
                    content_parts.append(f"• {name} (similarity: {similarity:.2f})")
                else:
                    # Add additional context if available
                    activity = compound.get("activity_type", "")
                    mw = compound.get("molecular_weight", "")

                    details = []
                    if activity:
                        details.append(f"Activity: {activity}")
                    if mw:
                        details.append(f"MW: {mw:.1f}")

                    detail_str = f" ({', '.join(details)})" if details else ""
                    content_parts.append(
                        f"• {name} (similarity: {similarity:.2f}){detail_str}"
                    )

            # Similarity distribution analysis
            if detail_level in [
                ExplanationDetail.DETAILED,
                ExplanationDetail.COMPREHENSIVE,
            ]:
                similarities = [c["similarity_score"] for c in similar_compounds]
                avg_sim = np.mean(similarities)
                content_parts.append(f"\nAverage similarity: {avg_sim:.2f}")

                if len(similarities) > 1:
                    std_sim = np.std(similarities)
                    content_parts.append(
                        f"Similarity distribution: {std_sim:.2f} (std dev)"
                    )
        else:
            content_parts.append(
                "No significantly similar compounds found in knowledge base."
            )

        content = "\n".join(content_parts)

        return ExplanationComponent(
            title="Molecular Similarity",
            content=content,
            evidence_strength=len(similar_compounds) / 10.0,
            confidence=0.9,
            source_type="kg_data",
        )

    def _generate_property_correlation(
        self, prediction: AIContextualPrediction, detail_level: ExplanationDetail
    ) -> ExplanationComponent:
        """Generate explanation of property correlations."""
        content_parts = []

        # Extract property correlations from KG context
        correlations = prediction.kg_context.properties.get("property_correlations", [])

        if correlations:
            content_parts.append("Property Correlations with Similar Compounds:")

            # Analyze property patterns
            property_stats = {}
            for corr in correlations:
                props = corr.get("properties", {})
                for prop_name, prop_value in props.items():
                    if prop_name not in property_stats:
                        property_stats[prop_name] = []
                    property_stats[prop_name].append(prop_value)

            for prop_name, values in property_stats.items():
                if len(values) > 1:
                    avg_value = np.mean(values)
                    std_value = np.std(values)

                    description = self.molecular_property_descriptions.get(
                        prop_name, f"{prop_name} affects molecular behavior"
                    )

                    content_parts.append(
                        f"• {prop_name.title()}: avg={avg_value:.2f} (±{std_value:.2f})"
                    )

                    if detail_level in [
                        ExplanationDetail.DETAILED,
                        ExplanationDetail.COMPREHENSIVE,
                    ]:
                        content_parts.append(f"  {description}")

        # Current prediction properties (if available)
        if (
            isinstance(prediction.prediction, dict)
            and "molecules" in prediction.prediction
        ):
            mol_data = (
                prediction.prediction["molecules"][0]
                if prediction.prediction["molecules"]
                else {}
            )
            current_props = mol_data.get("properties", {})

            if current_props and property_stats:
                content_parts.append("\nComparison with Predicted Properties:")

                for prop_name, current_value in current_props.items():
                    if prop_name in property_stats:
                        similar_values = property_stats[prop_name]
                        avg_similar = np.mean(similar_values)
                        diff = current_value - avg_similar

                        if abs(diff) < np.std(similar_values):
                            comparison = "similar to"
                        elif diff > 0:
                            comparison = "higher than"
                        else:
                            comparison = "lower than"

                        content_parts.append(
                            f"• {prop_name.title()}: {current_value:.2f} ({comparison} similar compounds)"
                        )

        if not content_parts:
            content_parts.append("No property correlation data available.")

        content = "\n".join(content_parts)

        return ExplanationComponent(
            title="Property Correlations",
            content=content,
            evidence_strength=len(correlations) / 20.0,
            confidence=0.8,
            source_type="kg_data",
        )

    def _generate_prediction_summary(self, prediction: AIContextualPrediction) -> str:
        """Generate overall prediction summary."""
        if isinstance(prediction.prediction, dict):
            if "molecules" in prediction.prediction:
                return f"Generated {len(prediction.prediction['molecules'])} molecular candidates using AI-guided design."
            elif "optimized_molecules" in prediction.prediction:
                return f"Optimized {len(prediction.prediction['optimized_molecules'])} molecules for target properties."
            elif "selected_molecules" in prediction.prediction:
                return f"Selected {len(prediction.prediction['selected_molecules'])} informative molecules for experimental validation."
            else:
                return (
                    "AI prediction completed with knowledge graph context integration."
                )
        else:
            return f"Prediction result: {prediction.prediction}"

    def _assess_confidence_level(self, confidence: float) -> str:
        """Assess confidence level based on score."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"

    def _extract_key_insights(
        self, prediction: AIContextualPrediction, components: List[ExplanationComponent]
    ) -> List[str]:
        """Extract key insights from prediction and components."""
        insights = []

        # High-confidence insights
        high_conf_components = [c for c in components if c.confidence > 0.8]
        if high_conf_components:
            insights.append(
                f"High confidence findings from {len(high_conf_components)} analysis components"
            )

        # Similarity insights
        similar_count = len(
            [
                e
                for e in prediction.kg_context.entities
                if "similarity_score" in e and e["similarity_score"] > 0.7
            ]
        )
        if similar_count > 0:
            insights.append(
                f"Found {similar_count} highly similar compounds with known properties"
            )

        # Activity insights
        activities = [e for e in prediction.kg_context.entities if "activity_type" in e]
        if activities:
            activity_types = set(e["activity_type"] for e in activities)
            insights.append(
                f"Identified {len(activity_types)} types of biological activities"
            )

        # Risk insights
        risks = len(
            [
                c
                for c in components
                if c.title == "Risk Assessment" and "High" in c.content
            ]
        )
        if risks > 0:
            insights.append(
                "Potential safety concerns identified requiring further evaluation"
            )

        return insights

    def _identify_limitations(self, prediction: AIContextualPrediction) -> List[str]:
        """Identify limitations of the prediction."""
        limitations = []

        # Confidence-based limitations
        if prediction.confidence < 0.6:
            limitations.append(
                "Moderate to low prediction confidence may affect reliability"
            )

        # Knowledge gap limitations
        if len(prediction.kg_context.entities) < 5:
            limitations.append(
                "Limited knowledge graph coverage may miss important molecular relationships"
            )

        # Uncertainty factors
        if prediction.uncertainty_factors:
            limitations.extend(
                [
                    f"Uncertainty factor: {factor}"
                    for factor in prediction.uncertainty_factors[:3]
                ]
            )

        # Novelty limitations
        similar_compounds = [
            e for e in prediction.kg_context.entities if "similarity_score" in e
        ]
        if (
            not similar_compounds
            or max(e["similarity_score"] for e in similar_compounds) < 0.5
        ):
            limitations.append("High molecular novelty may reduce prediction accuracy")

        # Model limitations
        limitations.append(
            "Predictions are based on training data and may not generalize to all scenarios"
        )

        return limitations

    def _explain_recommendation_basis(
        self, recommendation: str, prediction: AIContextualPrediction
    ) -> str:
        """Explain the basis for a specific recommendation."""
        # Simple pattern matching for common recommendations
        if "validation" in recommendation.lower():
            return "Based on prediction uncertainty and limited experimental data"
        elif "synthesis" in recommendation.lower():
            return "Suggested based on molecular similarity to active compounds"
        elif "target" in recommendation.lower():
            return "Inferred from knowledge graph target interaction data"
        elif "property" in recommendation.lower():
            return "Based on molecular property analysis and structure-activity relationships"
        else:
            return "Derived from AI model analysis and knowledge graph context"

    def _generate_references(self, prediction: AIContextualPrediction) -> List[str]:
        """Generate references for the explanation."""
        references = []

        # KG-based references
        if hasattr(prediction.kg_context, "source_queries"):
            references.append(
                "Knowledge Graph: ChemiAccel Integrated Chemistry Knowledge Graph"
            )

        # Literature references (if available in KG context)
        if hasattr(prediction.kg_context, "literature_refs"):
            references.extend(prediction.kg_context.literature_refs[:5])

        # Model references
        references.append("AI Model: ChemiAccel Predictive Models")

        return references


class ExplanationFormatter:
    """Formats explanations for different output formats."""

    @staticmethod
    def to_markdown(explanation: ContextualExplanation) -> str:
        """Format explanation as Markdown."""
        lines = []

        # Header
        lines.append(f"# {explanation.prediction_summary}")
        lines.append(f"**Confidence:** {explanation.confidence_level.title()}")
        lines.append("")

        # Main rationale
        lines.append("## Main Rationale")
        lines.append(explanation.main_rationale)
        lines.append("")

        # Components
        for component in explanation.components:
            lines.append(f"## {component.title}")
            lines.append(component.content)
            lines.append("")

        # Key insights
        if explanation.key_insights:
            lines.append("## Key Insights")
            for insight in explanation.key_insights:
                lines.append(f"- {insight}")
            lines.append("")

        # Limitations
        if explanation.limitations:
            lines.append("## Limitations")
            for limitation in explanation.limitations:
                lines.append(f"- {limitation}")
            lines.append("")

        # Recommendations
        if explanation.recommendations:
            lines.append("## Recommendations")
            for i, rec in enumerate(explanation.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        # References
        if explanation.references:
            lines.append("## References")
            for i, ref in enumerate(explanation.references, 1):
                lines.append(f"{i}. {ref}")

        return "\n".join(lines)

    @staticmethod
    def to_html(explanation: ContextualExplanation) -> str:
        """Format explanation as HTML."""
        html_parts = []

        # Header
        html_parts.append(f"<h1>{explanation.prediction_summary}</h1>")
        html_parts.append(
            f"<p><strong>Confidence:</strong> {explanation.confidence_level.title()}</p>"
        )

        # Main rationale
        html_parts.append("<h2>Main Rationale</h2>")
        html_parts.append(f"<p>{explanation.main_rationale}</p>")

        # Components
        for component in explanation.components:
            html_parts.append(f"<h2>{component.title}</h2>")
            # Convert newlines to <br> tags
            content_html = component.content.replace("\n", "<br>")
            html_parts.append(f"<p>{content_html}</p>")

        # Key insights
        if explanation.key_insights:
            html_parts.append("<h2>Key Insights</h2>")
            html_parts.append("<ul>")
            for insight in explanation.key_insights:
                html_parts.append(f"<li>{insight}</li>")
            html_parts.append("</ul>")

        # Limitations
        if explanation.limitations:
            html_parts.append("<h2>Limitations</h2>")
            html_parts.append("<ul>")
            for limitation in explanation.limitations:
                html_parts.append(f"<li>{limitation}</li>")
            html_parts.append("</ul>")

        # Recommendations
        if explanation.recommendations:
            html_parts.append("<h2>Recommendations</h2>")
            html_parts.append("<ol>")
            for rec in explanation.recommendations:
                html_parts.append(f"<li>{rec}</li>")
            html_parts.append("</ol>")

        # References
        if explanation.references:
            html_parts.append("<h2>References</h2>")
            html_parts.append("<ol>")
            for ref in explanation.references:
                html_parts.append(f"<li>{ref}</li>")
            html_parts.append("</ol>")

        return "\n".join(html_parts)

    @staticmethod
    def to_json(explanation: ContextualExplanation) -> Dict[str, Any]:
        """Format explanation as JSON-serializable dictionary."""
        return {
            "prediction_summary": explanation.prediction_summary,
            "confidence_level": explanation.confidence_level,
            "main_rationale": explanation.main_rationale,
            "components": [
                {
                    "title": comp.title,
                    "content": comp.content,
                    "evidence_strength": comp.evidence_strength,
                    "confidence": comp.confidence,
                    "source_type": comp.source_type,
                    "supporting_data": comp.supporting_data,
                    "visual_aids": comp.visual_aids,
                }
                for comp in explanation.components
            ],
            "key_insights": explanation.key_insights,
            "limitations": explanation.limitations,
            "recommendations": explanation.recommendations,
            "references": explanation.references,
            "metadata": explanation.metadata,
        }


# Utility functions for explanation generation
def generate_comprehensive_explanation(
    prediction: AIContextualPrediction,
    detail_level: ExplanationDetail = ExplanationDetail.DETAILED,
) -> ContextualExplanation:
    """Generate comprehensive explanation for any AI prediction."""
    generator = ChemicalExplanationGenerator()

    explanation_types = [
        ExplanationType.PREDICTION_RATIONALE,
        ExplanationType.EVIDENCE_SUMMARY,
        ExplanationType.UNCERTAINTY_ANALYSIS,
        ExplanationType.SIMILARITY_EXPLANATION,
        ExplanationType.RISK_ASSESSMENT,
        ExplanationType.NOVELTY_ANALYSIS,
    ]

    return generator.generate_explanation(prediction, explanation_types, detail_level)


def generate_brief_explanation(
    prediction: AIContextualPrediction,
) -> ContextualExplanation:
    """Generate brief explanation for quick review."""
    generator = ChemicalExplanationGenerator()

    explanation_types = [
        ExplanationType.PREDICTION_RATIONALE,
        ExplanationType.EVIDENCE_SUMMARY,
    ]

    return generator.generate_explanation(
        prediction, explanation_types, ExplanationDetail.BRIEF
    )


def explain_prediction_with_context(
    prediction: AIContextualPrediction, format_type: str = "markdown"
) -> str:
    """Generate and format explanation for a prediction."""
    explanation = generate_comprehensive_explanation(prediction)

    if format_type.lower() == "markdown":
        return ExplanationFormatter.to_markdown(explanation)
    elif format_type.lower() == "html":
        return ExplanationFormatter.to_html(explanation)
    elif format_type.lower() == "json":
        import json

        return json.dumps(ExplanationFormatter.to_json(explanation), indent=2)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

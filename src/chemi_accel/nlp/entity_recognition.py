"""
Chemical Entity Recognition System

GPU-accelerated chemical entity recognition for automated extraction of chemical
entities and relationships from unstructured text.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .backends import NLPBackend, get_nlp_backend
from .backends.base import EntityExtraction, RelationshipExtraction

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of chemical entities that can be recognized."""

    CHEMICAL = "chemical"
    COMPOUND = "compound"
    MOLECULE = "molecule"
    REACTION = "reaction"
    PROPERTY = "property"
    VALUE = "value"
    UNIT = "unit"
    PROCESS = "process"
    MATERIAL = "material"
    CATALYST = "catalyst"
    SOLVENT = "solvent"
    PRODUCT = "product"
    REAGENT = "reagent"
    CONDITION = "condition"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    TIME = "time"
    CONCENTRATION = "concentration"


@dataclass
class RecognitionConfig:
    """Configuration for chemical entity recognition."""

    backend: str = "auto"  # auto, cuda, jax, rapids, cpu
    model_name: str = "chemical_ner"
    confidence_threshold: float = 0.5
    max_entity_length: int = 200
    include_context: bool = True
    context_window: int = 100
    merge_overlapping: bool = True
    filter_common_words: bool = True
    custom_patterns: Dict[str, str] = field(default_factory=dict)


class ChemicalEntityRecognizer:
    """
    GPU-accelerated chemical entity recognition system.

    Provides comprehensive chemical entity recognition using multiple backends
    and approaches including transformer models, rule-based patterns, and
    chemical knowledge.
    """

    def __init__(self, config: Optional[RecognitionConfig] = None):
        self.config = config or RecognitionConfig()
        self.backend = get_nlp_backend(
            backend=self.config.backend if self.config.backend != "auto" else None
        )

        # Chemical entity patterns
        self.entity_patterns = {
            EntityType.CHEMICAL: [
                r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*\b",  # Molecular formulas
                r"\b\d*[a-z]+(?:-\d+[a-z]+)*(?:yl|ine|ide|ate|oic acid)\b",  # IUPAC names
                r"\b[A-Z][a-z]+(?:mab|zumab|tinib|pril|olol|pine)\b",  # Drug names
            ],
            EntityType.REACTION: [
                r"\b(?:synthesis|reaction|formation|cyclization|oxidation|reduction)\b",
                r"\b\w+\s+reaction\b",
                r"\breaction\s+of\s+\w+\b",
            ],
            EntityType.PROPERTY: [
                r"\b(?:melting point|boiling point|solubility|density|viscosity)\b",
                r"\b(?:molecular weight|molar mass|formula weight)\b",
                r"\b(?:pKa|pH|LogP|TPSA|HLB)\b",
            ],
            EntityType.VALUE: [
                r"\b\d+(?:\.\d+)?\s*(?:°C|K|°F|mg|g|kg|mL|L|mol|mmol|M|mM|µM|nM)\b",
                r"\b\d+(?:\.\d+)?\s*(?:hours?|mins?|minutes?|days?|weeks?)\b",
                r"\b\d+(?:\.\d+)?\s*(?:atm|bar|Pa|kPa|MPa|torr|mmHg)\b",
            ],
            EntityType.CONDITION: [
                r"\bat\s+\d+(?:\.\d+)?\s*(?:°C|K|°F)\b",
                r"\bunder\s+(?:nitrogen|argon|vacuum|pressure)\b",
                r"\bfor\s+\d+(?:\.\d+)?\s*(?:hours?|mins?|days?)\b",
            ],
        }

        # Common words to filter out
        self.common_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "over",
            "after",
            "a",
            "an",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "we",
            "us",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "hers",
        }

        logger.info(
            f"Initialized ChemicalEntityRecognizer with {self.backend.backend_type.value} backend"
        )

    def extract_entities(
        self,
        text: Union[str, List[str]],
        entity_types: Optional[List[EntityType]] = None,
    ) -> List[EntityExtraction]:
        """
        Extract chemical entities from text using multiple approaches.

        Args:
            text: Input text or list of texts
            entity_types: Specific entity types to extract (None for all)

        Returns:
            List of extracted entities
        """
        if isinstance(text, str):
            text = [text]

        all_entities = []

        for single_text in text:
            # Extract using backend ML model
            ml_entities = self.backend.extract_entities(
                single_text, self.config.model_name
            )

            # Extract using rule-based patterns
            pattern_entities = self._extract_pattern_entities(single_text, entity_types)

            # Combine and deduplicate entities
            combined_entities = self._combine_entities(ml_entities, pattern_entities)

            # Filter and validate entities
            filtered_entities = self._filter_entities(combined_entities, single_text)

            all_entities.extend(filtered_entities)

        return all_entities

    def _extract_pattern_entities(
        self, text: str, entity_types: Optional[List[EntityType]] = None
    ) -> List[EntityExtraction]:
        """Extract entities using rule-based patterns."""
        entities = []

        patterns_to_use = self.entity_patterns
        if entity_types:
            patterns_to_use = {
                etype: patterns
                for etype, patterns in self.entity_patterns.items()
                if etype in entity_types
            }

        for entity_type, patterns in patterns_to_use.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()

                    # Skip if too short or common word
                    if len(entity_text) < 2 or entity_text.lower() in self.common_words:
                        continue

                    entity = EntityExtraction(
                        text=entity_text,
                        label=entity_type.value.upper(),
                        start_idx=match.start(),
                        end_idx=match.end(),
                        confidence=0.7,  # Pattern-based confidence
                        properties={
                            "method": "pattern_based",
                            "entity_type": entity_type.value,
                            "pattern": pattern,
                        },
                    )
                    entities.append(entity)

        return entities

    def _combine_entities(
        self,
        ml_entities: List[EntityExtraction],
        pattern_entities: List[EntityExtraction],
    ) -> List[EntityExtraction]:
        """Combine ML and pattern-based entities, handling overlaps."""
        combined = []
        all_entities = ml_entities + pattern_entities

        # Sort by start position
        all_entities.sort(key=lambda x: x.start_idx)

        for entity in all_entities:
            # Check for overlaps with existing entities
            overlapping = False

            if self.config.merge_overlapping:
                for existing in combined:
                    if self._entities_overlap(entity, existing):
                        # Keep the entity with higher confidence
                        if entity.confidence > existing.confidence:
                            combined.remove(existing)
                            combined.append(entity)
                        overlapping = True
                        break

            if not overlapping:
                combined.append(entity)

        return combined

    def _entities_overlap(
        self, entity1: EntityExtraction, entity2: EntityExtraction
    ) -> bool:
        """Check if two entities overlap."""
        return not (
            entity1.end_idx <= entity2.start_idx or entity2.end_idx <= entity1.start_idx
        )

    def _filter_entities(
        self, entities: List[EntityExtraction], text: str
    ) -> List[EntityExtraction]:
        """Filter entities based on configuration criteria."""
        filtered = []

        for entity in entities:
            # Filter by confidence threshold
            if entity.confidence < self.config.confidence_threshold:
                continue

            # Filter by length
            if len(entity.text) > self.config.max_entity_length:
                continue

            # Filter common words if enabled
            if (
                self.config.filter_common_words
                and entity.text.lower() in self.common_words
            ):
                continue

            # Add context if enabled
            if self.config.include_context:
                context_start = max(0, entity.start_idx - self.config.context_window)
                context_end = min(
                    len(text), entity.end_idx + self.config.context_window
                )
                entity.properties["context"] = text[context_start:context_end]

            filtered.append(entity)

        return filtered

    def extract_relationships(
        self,
        text: Union[str, List[str]],
        entities: Optional[List[EntityExtraction]] = None,
    ) -> List[RelationshipExtraction]:
        """
        Extract relationships between chemical entities.

        Args:
            text: Input text or list of texts
            entities: Pre-extracted entities (optional)

        Returns:
            List of extracted relationships
        """
        if isinstance(text, str):
            text = [text]

        if entities is None:
            entities = self.extract_entities(text)

        # Use backend for relationship extraction
        relationships = self.backend.extract_relationships(
            text, entities, "relation_extraction"
        )

        # Add custom relationship patterns
        custom_relationships = self._extract_custom_relationships(text, entities)
        relationships.extend(custom_relationships)

        return self._filter_relationships(relationships)

    def _extract_custom_relationships(
        self, texts: List[str], entities: List[EntityExtraction]
    ) -> List[RelationshipExtraction]:
        """Extract relationships using custom chemical patterns."""
        relationships = []

        # Chemical reaction patterns
        reaction_patterns = [
            (r"(\w+)\s+reacts?\s+with\s+(\w+)", "REACTS_WITH"),
            (r"(\w+)\s+forms?\s+(\w+)", "FORMS"),
            (r"(\w+)\s+catalyzes?\s+(\w+)", "CATALYZES"),
            (r"(\w+)\s+inhibits?\s+(\w+)", "INHIBITS"),
            (r"(\w+)\s+binds?\s+to\s+(\w+)", "BINDS_TO"),
            (r"(\w+)\s+is\s+similar\s+to\s+(\w+)", "SIMILAR_TO"),
        ]

        for text in texts:
            for pattern, relation_type in reaction_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source_text = match.group(1)
                    target_text = match.group(2)

                    # Find corresponding entities
                    source_entity = self._find_entity_by_text(entities, source_text)
                    target_entity = self._find_entity_by_text(entities, target_text)

                    if source_entity and target_entity:
                        relationship = RelationshipExtraction(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            relation_type=relation_type,
                            confidence=0.8,
                            context=match.group(),
                            properties={"method": "custom_pattern", "pattern": pattern},
                        )
                        relationships.append(relationship)

        return relationships

    def _find_entity_by_text(
        self, entities: List[EntityExtraction], text: str
    ) -> Optional[EntityExtraction]:
        """Find entity by matching text."""
        text = text.lower().strip()
        for entity in entities:
            if entity.text.lower().strip() == text:
                return entity
        return None

    def _filter_relationships(
        self, relationships: List[RelationshipExtraction]
    ) -> List[RelationshipExtraction]:
        """Filter relationships based on confidence and validity."""
        filtered = []

        for relationship in relationships:
            if relationship.confidence < self.config.confidence_threshold:
                continue

            # Avoid self-relationships
            if relationship.source_entity.text == relationship.target_entity.text:
                continue

            filtered.append(relationship)

        return filtered

    def recognize_chemical_names(self, text: str) -> List[Dict[str, Any]]:
        """
        Specialized method for recognizing chemical names and identifiers.

        Args:
            text: Input text

        Returns:
            List of recognized chemical names with metadata
        """
        # Chemical name patterns
        patterns = {
            "iupac_name": r"\b\d*[a-z]+(?:-\d+[a-z]+)*(?:yl|ine|ide|ate|oic acid|ol|one|al)\b",
            "molecular_formula": r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*\b",
            "cas_number": r"\b\d{2,7}-\d{2}-\d\b",
            "drug_name": r"\b[A-Z][a-z]+(?:mab|zumab|tinib|pril|olol|pine|mycin|cillin)\b",
            "trade_name": r"\b[A-Z][a-z]*®?\b",
        }

        chemical_names = []

        for name_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group().strip()

                # Validate chemical name
                if len(name) < 3 or name.lower() in self.common_words:
                    continue

                chemical_names.append(
                    {
                        "name": name,
                        "type": name_type,
                        "start_idx": match.start(),
                        "end_idx": match.end(),
                        "confidence": self._calculate_chemical_name_confidence(
                            name, name_type
                        ),
                    }
                )

        return chemical_names

    def _calculate_chemical_name_confidence(self, name: str, name_type: str) -> float:
        """Calculate confidence score for chemical name recognition."""
        base_confidence = 0.7

        # Boost confidence for specific patterns
        if name_type == "cas_number":
            base_confidence = 0.95
        elif name_type == "molecular_formula" and re.match(r"^[A-Z][a-z]?\d*", name):
            base_confidence = 0.9
        elif name_type == "drug_name":
            base_confidence = 0.8

        # Reduce confidence for very short names
        if len(name) < 4:
            base_confidence *= 0.8

        return min(base_confidence, 1.0)

    def extract_numerical_values(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract numerical values with units from chemical text.

        Args:
            text: Input text

        Returns:
            List of numerical values with units and context
        """
        # Patterns for numerical values with units
        value_patterns = [
            r"(\d+(?:\.\d+)?)\s*(°C|K|°F)",  # Temperature
            r"(\d+(?:\.\d+)?)\s*(mg|g|kg|µg|ng)",  # Mass
            r"(\d+(?:\.\d+)?)\s*(mL|L|µL|dL)",  # Volume
            r"(\d+(?:\.\d+)?)\s*(M|mM|µM|nM|mol|mmol)",  # Concentration/amount
            r"(\d+(?:\.\d+)?)\s*(atm|bar|Pa|kPa|MPa|torr|mmHg)",  # Pressure
            r"(\d+(?:\.\d+)?)\s*(hours?|mins?|minutes?|days?|h|min|d)",  # Time
            r"(\d+(?:\.\d+)?)\s*(%|percent)",  # Percentage
            r"(\d+(?:\.\d+)?)\s*(rpm|Hz|kHz|MHz)",  # Frequency/rotation
        ]

        numerical_values = []

        for pattern in value_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = float(match.group(1))
                unit = match.group(2)

                # Get context around the value
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end].strip()

                numerical_values.append(
                    {
                        "value": value,
                        "unit": unit,
                        "text": match.group(),
                        "start_idx": match.start(),
                        "end_idx": match.end(),
                        "context": context,
                        "type": self._classify_value_type(unit),
                    }
                )

        return numerical_values

    def _classify_value_type(self, unit: str) -> str:
        """Classify the type of numerical value based on unit."""
        unit_classifications = {
            "temperature": ["°C", "K", "°F"],
            "mass": ["mg", "g", "kg", "µg", "ng"],
            "volume": ["mL", "L", "µL", "dL"],
            "concentration": ["M", "mM", "µM", "nM"],
            "amount": ["mol", "mmol"],
            "pressure": ["atm", "bar", "Pa", "kPa", "MPa", "torr", "mmHg"],
            "time": ["hours", "hour", "mins", "minutes", "days", "h", "min", "d"],
            "percentage": ["%", "percent"],
            "frequency": ["rpm", "Hz", "kHz", "MHz"],
        }

        for value_type, units in unit_classifications.items():
            if unit in units:
                return value_type

        return "unknown"

    def get_entity_statistics(self, entities: List[EntityExtraction]) -> Dict[str, Any]:
        """Get statistics about extracted entities."""
        if not entities:
            return {"total_entities": 0}

        # Count by label
        label_counts = {}
        for entity in entities:
            label_counts[entity.label] = label_counts.get(entity.label, 0) + 1

        # Count by method
        method_counts = {}
        for entity in entities:
            method = entity.properties.get("method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        # Confidence statistics
        confidences = [entity.confidence for entity in entities]

        return {
            "total_entities": len(entities),
            "unique_entities": len(set(entity.text.lower() for entity in entities)),
            "label_distribution": label_counts,
            "method_distribution": method_counts,
            "confidence_stats": {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
            },
            "average_entity_length": sum(len(entity.text) for entity in entities)
            / len(entities),
        }

    def set_backend(self, backend_name: str):
        """Switch to a different backend."""
        self.backend = get_nlp_backend(backend_name)
        logger.info(f"Switched to {backend_name} backend")

    def add_custom_pattern(self, entity_type: EntityType, pattern: str):
        """Add a custom pattern for entity recognition."""
        if entity_type not in self.entity_patterns:
            self.entity_patterns[entity_type] = []
        self.entity_patterns[entity_type].append(pattern)
        logger.info(f"Added custom pattern for {entity_type.value}: {pattern}")

"""
Advanced Natural Language Processing for Chemical Queries

Sophisticated NLP capabilities for parsing, understanding, and converting
natural language queries into structured semantic queries.
"""

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .ontology import ChemicalOntology, EntityType, RelationshipType


class QueryIntent(Enum):
    """Types of query intents."""

    FIND_ENTITIES = "find_entities"
    COMPARE_ENTITIES = "compare_entities"
    FIND_RELATIONSHIPS = "find_relationships"
    GET_PROPERTIES = "get_properties"
    EXPLAIN_MECHANISM = "explain_mechanism"
    PREDICT_INTERACTION = "predict_interaction"
    FIND_SIMILAR = "find_similar"
    ANALYZE_STRUCTURE = "analyze_structure"
    SEARCH_LITERATURE = "search_literature"
    GENERAL_QUESTION = "general_question"


class EntityMention(Enum):
    """Types of entity mentions."""

    DIRECT_NAME = "direct_name"
    CHEMICAL_FORMULA = "chemical_formula"
    SMILES = "smiles"
    INCHI = "inchi"
    SYNONYM = "synonym"
    DESCRIPTION = "description"
    GENERIC_REFERENCE = "generic_reference"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from natural language."""

    text: str
    entity_type: EntityType
    mention_type: EntityMention
    confidence: float
    start_pos: int
    end_pos: int
    ontology_uri: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    context: str = ""


@dataclass
class QueryContext:
    """Context information for query understanding."""

    domain: str = "chemistry"
    user_expertise: str = "general"  # general, expert, researcher
    query_history: List[str] = field(default_factory=list)
    session_entities: Set[str] = field(default_factory=set)
    temporal_context: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedQuery:
    """Represents a processed natural language query."""

    original_query: str
    cleaned_query: str
    intent: QueryIntent
    confidence: float
    entities: List[ExtractedEntity]
    relationships: List[str]
    constraints: List[Dict[str, Any]]
    query_focus: str
    expected_answer_type: str
    semantic_parse: Dict[str, Any]
    context: QueryContext


class ChemicalEntityExtractor:
    """Advanced chemical entity extraction using multiple techniques."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.patterns = self._initialize_patterns()
        self.stop_words = self._load_stop_words()
        self.chemical_keywords = self._load_chemical_keywords()

    def extract_entities(
        self, text: str, context: QueryContext
    ) -> List[ExtractedEntity]:
        """Extract chemical entities from text."""
        entities = []

        # Rule-based extraction
        entities.extend(self._extract_by_patterns(text))

        # Ontology-based extraction
        entities.extend(self._extract_by_ontology(text))

        # Context-based extraction
        entities.extend(self._extract_by_context(text, context))

        # Chemical formula extraction
        entities.extend(self._extract_chemical_formulas(text))

        # SMILES extraction
        entities.extend(self._extract_smiles(text))

        # Resolve conflicts and rank entities
        return self._resolve_and_rank_entities(entities, text, context)

    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize entity extraction patterns."""
        return {
            "drug_names": [
                r"\b(?:aspirin|ibuprofen|acetaminophen|morphine|codeine|warfarin|insulin)\b",
                r"\b\w*cillin\b",  # Penicillins
                r"\b\w*statin\b",  # Statins
                r"\b\w*pril\b",  # ACE inhibitors
                r"\b\w*sartan\b",  # ARBs
            ],
            "chemical_classes": [
                r"\b(?:antibiotic|analgesic|anti-inflammatory|enzyme|protein|hormone)\b",
                r"\b(?:steroid|alkaloid|glycoside|lipid|carbohydrate)\b",
                r"\b(?:NSAID|ACE inhibitor|beta blocker|calcium channel blocker)\b",
            ],
            "conditions": [
                r"\b(?:pain|fever|inflammation|infection|cancer|diabetes|hypertension)\b",
                r"\b(?:arthritis|asthma|depression|anxiety|migraine|epilepsy)\b",
            ],
            "biological_processes": [
                r"\b(?:metabolism|synthesis|degradation|absorption|distribution)\b",
                r"\b(?:pharmacokinetics|pharmacodynamics|bioavailability)\b",
            ],
            "molecular_properties": [
                r"\b(?:molecular weight|melting point|boiling point|solubility)\b",
                r"\b(?:LogP|pKa|bioavailability|half-life|clearance)\b",
            ],
        }

    def _extract_by_patterns(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        text_lower = text.lower()

        for category, patterns in self.patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))

                for match in matches:
                    entity_type = self._category_to_entity_type(category)
                    if entity_type:
                        entities.append(
                            ExtractedEntity(
                                text=match.group(),
                                entity_type=entity_type,
                                mention_type=EntityMention.DIRECT_NAME,
                                confidence=0.8,
                                start_pos=match.start(),
                                end_pos=match.end(),
                                context=self._extract_context(
                                    text, match.start(), match.end()
                                ),
                            )
                        )

        return entities

    def _extract_by_ontology(self, text: str) -> List[ExtractedEntity]:
        """Extract entities by looking them up in the ontology."""
        entities = []
        words = text.split()

        # Check individual words and n-grams
        for n in range(1, 4):  # 1-gram to 3-gram
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n]).lower()

                # Skip if phrase is too short or contains only stop words
                if len(phrase) < 3 or all(
                    word in self.stop_words for word in phrase.split()
                ):
                    continue

                # Look up in ontology
                ontology_matches = self.ontology.find_entity(phrase)
                if ontology_matches:
                    best_match = ontology_matches[0]

                    # Calculate position in original text
                    start_pos = text.lower().find(phrase)
                    end_pos = start_pos + len(phrase)

                    entities.append(
                        ExtractedEntity(
                            text=phrase,
                            entity_type=best_match.entity_type,
                            mention_type=EntityMention.DIRECT_NAME,
                            confidence=0.9,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            ontology_uri=best_match.uri,
                            synonyms=getattr(best_match, "synonyms", []),
                            context=self._extract_context(text, start_pos, end_pos),
                        )
                    )

        return entities

    def _extract_by_context(
        self, text: str, context: QueryContext
    ) -> List[ExtractedEntity]:
        """Extract entities using contextual information."""
        entities = []

        # Use session entities for disambiguation
        for entity_text in context.session_entities:
            if entity_text.lower() in text.lower():
                pos = text.lower().find(entity_text.lower())
                entities.append(
                    ExtractedEntity(
                        text=entity_text,
                        entity_type=EntityType.SMALL_MOLECULE,  # Default
                        mention_type=EntityMention.GENERIC_REFERENCE,
                        confidence=0.7,
                        start_pos=pos,
                        end_pos=pos + len(entity_text),
                        context="session_reference",
                    )
                )

        # Extract pronouns and resolve them
        pronoun_entities = self._resolve_pronouns(text, context)
        entities.extend(pronoun_entities)

        return entities

    def _extract_chemical_formulas(self, text: str) -> List[ExtractedEntity]:
        """Extract chemical molecular formulas."""
        entities = []

        # Pattern for molecular formulas (e.g., C6H12O6, NaCl)
        formula_pattern = r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)*\b"

        matches = re.finditer(formula_pattern, text)
        for match in matches:
            formula = match.group()

            # Validate that it looks like a chemical formula
            if self._is_valid_chemical_formula(formula):
                entities.append(
                    ExtractedEntity(
                        text=formula,
                        entity_type=EntityType.SMALL_MOLECULE,
                        mention_type=EntityMention.CHEMICAL_FORMULA,
                        confidence=0.85,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        properties={"molecular_formula": formula},
                        context=self._extract_context(text, match.start(), match.end()),
                    )
                )

        return entities

    def _extract_smiles(self, text: str) -> List[ExtractedEntity]:
        """Extract SMILES strings."""
        entities = []

        # Pattern for SMILES (simplified)
        smiles_pattern = r"\b[CNOPSFClBrI][\w\[\]()=#+\-@/\\]*\b"

        matches = re.finditer(smiles_pattern, text)
        for match in matches:
            smiles = match.group()

            # Validate SMILES format
            if self._is_valid_smiles(smiles):
                entities.append(
                    ExtractedEntity(
                        text=smiles,
                        entity_type=EntityType.SMALL_MOLECULE,
                        mention_type=EntityMention.SMILES,
                        confidence=0.9,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        properties={"smiles": smiles},
                        context=self._extract_context(text, match.start(), match.end()),
                    )
                )

        return entities

    def _resolve_and_rank_entities(
        self, entities: List[ExtractedEntity], text: str, context: QueryContext
    ) -> List[ExtractedEntity]:
        """Resolve conflicts and rank entities by confidence."""
        # Remove overlapping entities, keeping highest confidence
        non_overlapping = []
        entities.sort(key=lambda x: (x.start_pos, -x.confidence))

        for entity in entities:
            # Check for overlaps with already selected entities
            overlap = False
            for selected in non_overlapping:
                if (
                    entity.start_pos < selected.end_pos
                    and entity.end_pos > selected.start_pos
                ):
                    overlap = True
                    break

            if not overlap:
                non_overlapping.append(entity)

        # Enhance entities with additional information
        enhanced_entities = []
        for entity in non_overlapping:
            # Add chemical keywords boost
            if any(
                keyword in entity.text.lower() for keyword in self.chemical_keywords
            ):
                entity.confidence = min(entity.confidence + 0.1, 1.0)

            # Add context relevance boost
            if entity.context and any(
                keyword in entity.context.lower() for keyword in self.chemical_keywords
            ):
                entity.confidence = min(entity.confidence + 0.05, 1.0)

            enhanced_entities.append(entity)

        # Sort by confidence
        enhanced_entities.sort(key=lambda x: x.confidence, reverse=True)

        return enhanced_entities

    def _category_to_entity_type(self, category: str) -> Optional[EntityType]:
        """Map pattern category to entity type."""
        mapping = {
            "drug_names": EntityType.DRUG,
            "chemical_classes": EntityType.CHEMICAL_CLASS,
            "conditions": EntityType.CONDITION,
            "biological_processes": EntityType.BIOLOGICAL_PROCESS,
            "molecular_properties": EntityType.MOLECULAR_PROPERTY,
        }
        return mapping.get(category)

    def _extract_context(
        self, text: str, start_pos: int, end_pos: int, window_size: int = 50
    ) -> str:
        """Extract context around entity mention."""
        context_start = max(0, start_pos - window_size)
        context_end = min(len(text), end_pos + window_size)

        return text[context_start:context_end]

    def _is_valid_chemical_formula(self, formula: str) -> bool:
        """Validate chemical formula format."""
        # Check if it contains valid chemical elements
        valid_elements = {
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
        }

        # Extract elements from formula
        elements = re.findall(r"[A-Z][a-z]?", formula)

        return len(elements) > 0 and all(elem in valid_elements for elem in elements)

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Validate SMILES format (simplified)."""
        # Basic SMILES validation
        if len(smiles) < 2:
            return False

        # Check for valid SMILES characters
        valid_chars = set("CNOPSFClBrI[]()=#+-@/\\")
        return all(c.isalnum() or c in valid_chars for c in smiles)

    def _resolve_pronouns(
        self, text: str, context: QueryContext
    ) -> List[ExtractedEntity]:
        """Resolve pronouns to entities from context."""
        entities = []

        pronouns = ["it", "this", "that", "these", "those", "they"]

        for pronoun in pronouns:
            pattern = rf"\b{pronoun}\b"
            matches = list(re.finditer(pattern, text, re.IGNORECASE))

            for match in matches:
                # Try to resolve from query history
                if context.query_history:
                    last_query_entities = self._get_entities_from_history(
                        context.query_history[-1]
                    )
                    if last_query_entities:
                        entities.append(
                            ExtractedEntity(
                                text=match.group(),
                                entity_type=EntityType.SMALL_MOLECULE,
                                mention_type=EntityMention.GENERIC_REFERENCE,
                                confidence=0.6,
                                start_pos=match.start(),
                                end_pos=match.end(),
                                properties={"resolved_to": last_query_entities[0]},
                                context="pronoun_resolution",
                            )
                        )

        return entities

    def _get_entities_from_history(self, query: str) -> List[str]:
        """Extract entity names from previous query."""
        # Simplified implementation
        entities = []
        for keyword in self.chemical_keywords:
            if keyword in query.lower():
                entities.append(keyword)
        return entities

    def _load_stop_words(self) -> Set[str]:
        """Load stop words for entity extraction."""
        return {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
            "would",
            "can",
            "could",
            "should",
            "may",
            "might",
            "what",
            "when",
            "where",
            "why",
            "how",
        }

    def _load_chemical_keywords(self) -> Set[str]:
        """Load chemical domain keywords."""
        return {
            "drug",
            "compound",
            "molecule",
            "chemical",
            "substance",
            "agent",
            "treatment",
            "therapy",
            "medication",
            "pharmaceutical",
            "bioactive",
            "enzyme",
            "protein",
            "receptor",
            "pathway",
            "mechanism",
            "reaction",
            "synthesis",
            "metabolism",
            "pharmacokinetics",
            "pharmacodynamics",
            "toxicity",
            "efficacy",
            "potency",
            "selectivity",
            "specificity",
            "molecular",
            "structural",
            "biological",
            "therapeutic",
            "clinical",
        }


class IntentClassifier:
    """Classifies query intent using multiple techniques."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.intent_patterns = self._initialize_intent_patterns()
        self.question_words = self._initialize_question_words()

    def classify_intent(
        self, query: str, entities: List[ExtractedEntity], context: QueryContext
    ) -> Tuple[QueryIntent, float]:
        """Classify query intent with confidence score."""

        # Rule-based classification
        rule_intent, rule_confidence = self._classify_by_rules(query, entities)

        # Pattern-based classification
        pattern_intent, pattern_confidence = self._classify_by_patterns(query)

        # Entity-based classification
        entity_intent, entity_confidence = self._classify_by_entities(entities)

        # Context-based classification
        context_intent, context_confidence = self._classify_by_context(query, context)

        # Combine classifications
        candidates = [
            (rule_intent, rule_confidence),
            (pattern_intent, pattern_confidence),
            (entity_intent, entity_confidence),
            (context_intent, context_confidence),
        ]

        # Select best candidate
        best_intent, best_confidence = max(candidates, key=lambda x: x[1])

        return best_intent, best_confidence

    def _initialize_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Initialize intent classification patterns."""
        return {
            QueryIntent.FIND_ENTITIES: [
                r"\b(?:find|search|look for|identify|list|show)\b.*\b(?:compound|drug|molecule|chemical)\b",
                r"\bwhat (?:is|are)\b.*\b(?:compound|drug|molecule|chemical)\b",
                r"\b(?:compound|drug|molecule|chemical)s?\s+(?:that|which|with)\b",
            ],
            QueryIntent.COMPARE_ENTITIES: [
                r"\b(?:compare|difference|similar|contrast)\b",
                r"\b(?:better|worse|more|less)\s+(?:effective|potent|toxic)\b",
                r"\bvs\b|\bversus\b",
            ],
            QueryIntent.FIND_RELATIONSHIPS: [
                r"\bhow\s+(?:does|do)\b.*\b(?:interact|affect|influence|relate)\b",
                r"\b(?:relationship|interaction|connection)\s+between\b",
                r"\b(?:treat|cause|lead to|result in)\b",
            ],
            QueryIntent.GET_PROPERTIES: [
                r"\bwhat\s+(?:is|are)\s+the\b.*\b(?:property|properties|characteristic)\b",
                r"\b(?:molecular weight|melting point|boiling point|solubility|LogP)\b",
                r"\b(?:formula|structure|composition)\s+of\b",
            ],
            QueryIntent.EXPLAIN_MECHANISM: [
                r"\bhow\s+(?:does|do)\b.*\bwork\b",
                r"\b(?:mechanism|mode of action|pathway)\b",
                r"\bexplain\b.*\b(?:process|mechanism|action)\b",
            ],
            QueryIntent.PREDICT_INTERACTION: [
                r"\b(?:predict|expect|anticipate)\b.*\b(?:interaction|effect)\b",
                r"\bwhat\s+(?:happens|occurs)\s+(?:if|when)\b",
                r"\b(?:side effect|adverse|reaction)\b",
            ],
            QueryIntent.FIND_SIMILAR: [
                r"\b(?:similar|like|analogous|comparable)\s+(?:to|with)\b",
                r"\b(?:alternative|substitute|replacement)\s+(?:for|to)\b",
                r"\bother\b.*\b(?:compound|drug)s?\s+(?:like|similar)\b",
            ],
            QueryIntent.ANALYZE_STRUCTURE: [
                r"\b(?:structure|conformation|geometry)\s+(?:of|analysis)\b",
                r"\b(?:SMILES|InChI|molecular structure)\b",
                r"\b(?:3D|three-dimensional|structural)\s+(?:analysis|model)\b",
            ],
        }

    def _initialize_question_words(self) -> Dict[str, List[QueryIntent]]:
        """Initialize question word to intent mappings."""
        return {
            "what": [
                QueryIntent.GET_PROPERTIES,
                QueryIntent.FIND_ENTITIES,
                QueryIntent.EXPLAIN_MECHANISM,
            ],
            "how": [QueryIntent.EXPLAIN_MECHANISM, QueryIntent.FIND_RELATIONSHIPS],
            "why": [QueryIntent.EXPLAIN_MECHANISM, QueryIntent.FIND_RELATIONSHIPS],
            "when": [QueryIntent.FIND_RELATIONSHIPS, QueryIntent.EXPLAIN_MECHANISM],
            "where": [QueryIntent.FIND_ENTITIES, QueryIntent.FIND_RELATIONSHIPS],
            "which": [QueryIntent.FIND_ENTITIES, QueryIntent.COMPARE_ENTITIES],
            "who": [QueryIntent.FIND_ENTITIES],
            "can": [QueryIntent.PREDICT_INTERACTION, QueryIntent.FIND_RELATIONSHIPS],
            "does": [QueryIntent.FIND_RELATIONSHIPS, QueryIntent.EXPLAIN_MECHANISM],
            "is": [QueryIntent.GET_PROPERTIES, QueryIntent.COMPARE_ENTITIES],
            "are": [QueryIntent.GET_PROPERTIES, QueryIntent.COMPARE_ENTITIES],
        }

    def _classify_by_rules(
        self, query: str, entities: List[ExtractedEntity]
    ) -> Tuple[QueryIntent, float]:
        """Classify intent using hand-crafted rules."""
        query_lower = query.lower()

        # Check for comparison indicators
        if any(
            word in query_lower
            for word in ["compare", "vs", "versus", "difference", "better", "worse"]
        ):
            return QueryIntent.COMPARE_ENTITIES, 0.9

        # Check for property queries
        property_indicators = [
            "molecular weight",
            "formula",
            "structure",
            "properties",
            "characteristics",
        ]
        if any(indicator in query_lower for indicator in property_indicators):
            return QueryIntent.GET_PROPERTIES, 0.8

        # Check for mechanism queries
        mechanism_indicators = [
            "how does",
            "mechanism",
            "mode of action",
            "pathway",
            "work",
        ]
        if any(indicator in query_lower for indicator in mechanism_indicators):
            return QueryIntent.EXPLAIN_MECHANISM, 0.8

        # Check for similarity queries
        similarity_indicators = [
            "similar",
            "like",
            "analogous",
            "alternative",
            "substitute",
        ]
        if any(indicator in query_lower for indicator in similarity_indicators):
            return QueryIntent.FIND_SIMILAR, 0.7

        # Check for interaction/relationship queries
        relationship_indicators = [
            "interact",
            "affect",
            "influence",
            "relationship",
            "connection",
        ]
        if any(indicator in query_lower for indicator in relationship_indicators):
            return QueryIntent.FIND_RELATIONSHIPS, 0.7

        # Default based on entity count
        if len(entities) > 1:
            return QueryIntent.COMPARE_ENTITIES, 0.5
        elif len(entities) == 1:
            return QueryIntent.GET_PROPERTIES, 0.5
        else:
            return QueryIntent.FIND_ENTITIES, 0.5

    def _classify_by_patterns(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify intent using regex patterns."""
        best_intent = QueryIntent.GENERAL_QUESTION
        best_score = 0.0

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score = 0.7
                    if score > best_score:
                        best_intent = intent
                        best_score = score

        return best_intent, best_score

    def _classify_by_entities(
        self, entities: List[ExtractedEntity]
    ) -> Tuple[QueryIntent, float]:
        """Classify intent based on extracted entities."""
        if not entities:
            return QueryIntent.FIND_ENTITIES, 0.3

        entity_types = [entity.entity_type for entity in entities]

        # Multiple entities suggest comparison
        if len(entities) > 1:
            return QueryIntent.COMPARE_ENTITIES, 0.6

        # Single entity with property type suggests property query
        if EntityType.MOLECULAR_PROPERTY in entity_types:
            return QueryIntent.GET_PROPERTIES, 0.7

        # Drug entities suggest various queries
        if EntityType.DRUG in entity_types:
            return QueryIntent.GET_PROPERTIES, 0.5

        return QueryIntent.FIND_ENTITIES, 0.4

    def _classify_by_context(
        self, query: str, context: QueryContext
    ) -> Tuple[QueryIntent, float]:
        """Classify intent using context information."""
        # Use query history for context
        if context.query_history:
            last_query = context.query_history[-1].lower()

            # Follow-up questions often have similar intent
            if any(
                word in query.lower()
                for word in ["also", "additionally", "furthermore", "moreover"]
            ):
                # Try to infer previous intent (simplified)
                if "compare" in last_query:
                    return QueryIntent.COMPARE_ENTITIES, 0.6
                elif "property" in last_query or "characteristic" in last_query:
                    return QueryIntent.GET_PROPERTIES, 0.6

        # Domain-specific context
        if context.domain == "drug_discovery":
            return QueryIntent.PREDICT_INTERACTION, 0.5
        elif context.domain == "chemical_analysis":
            return QueryIntent.ANALYZE_STRUCTURE, 0.5

        return QueryIntent.GENERAL_QUESTION, 0.3


class QueryRewriter:
    """Rewrites and normalizes natural language queries."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.abbreviation_map = self._load_abbreviation_map()
        self.synonym_map = self._load_synonym_map()

    def rewrite_query(
        self, query: str, entities: List[ExtractedEntity], intent: QueryIntent
    ) -> str:
        """Rewrite query for better processing."""

        # Expand abbreviations
        expanded_query = self._expand_abbreviations(query)

        # Normalize entity mentions
        normalized_query = self._normalize_entity_mentions(expanded_query, entities)

        # Add missing context
        contextualized_query = self._add_missing_context(normalized_query, intent)

        # Resolve coreferences
        resolved_query = self._resolve_coreferences(contextualized_query, entities)

        return resolved_query

    def _expand_abbreviations(self, query: str) -> str:
        """Expand common abbreviations."""
        expanded = query

        for abbrev, expansion in self.abbreviation_map.items():
            # Use word boundaries to avoid partial matches
            pattern = rf"\b{re.escape(abbrev)}\b"
            expanded = re.sub(pattern, expansion, expanded, flags=re.IGNORECASE)

        return expanded

    def _normalize_entity_mentions(
        self, query: str, entities: List[ExtractedEntity]
    ) -> str:
        """Normalize entity mentions to canonical forms."""
        normalized = query

        for entity in entities:
            if entity.ontology_uri:
                # Get canonical name from ontology
                ontology_entity = self.ontology.get_entity_by_uri(entity.ontology_uri)
                if ontology_entity and ontology_entity.name != entity.text:
                    # Replace with canonical name
                    normalized = normalized.replace(entity.text, ontology_entity.name)

        return normalized

    def _add_missing_context(self, query: str, intent: QueryIntent) -> str:
        """Add missing context based on query intent."""

        if intent == QueryIntent.FIND_ENTITIES:
            # Add explicit search terms if missing
            if not any(
                word in query.lower() for word in ["find", "search", "show", "list"]
            ):
                query = f"Find {query}"

        elif intent == QueryIntent.COMPARE_ENTITIES:
            # Add comparison context if missing
            if not any(
                word in query.lower() for word in ["compare", "difference", "vs"]
            ):
                query = f"Compare {query}"

        elif intent == QueryIntent.GET_PROPERTIES:
            # Add property context if missing
            if not any(
                word in query.lower()
                for word in ["property", "properties", "characteristic"]
            ):
                query = f"What are the properties of {query}"

        return query

    def _resolve_coreferences(self, query: str, entities: List[ExtractedEntity]) -> str:
        """Resolve pronouns and other coreferences."""
        resolved = query

        # Simple pronoun resolution
        pronouns = ["it", "this", "that", "they", "these", "those"]

        if entities:
            # Replace pronouns with first entity (simplified)
            first_entity = entities[0].text

            for pronoun in pronouns:
                pattern = rf"\b{pronoun}\b"
                resolved = re.sub(pattern, first_entity, resolved, flags=re.IGNORECASE)

        return resolved

    def _load_abbreviation_map(self) -> Dict[str, str]:
        """Load common chemical abbreviations."""
        return {
            "MW": "molecular weight",
            "BP": "boiling point",
            "MP": "melting point",
            "LogP": "partition coefficient",
            "IC50": "half maximal inhibitory concentration",
            "EC50": "half maximal effective concentration",
            "ADME": "absorption distribution metabolism excretion",
            "PK": "pharmacokinetics",
            "PD": "pharmacodynamics",
            "SAR": "structure activity relationship",
            "QSAR": "quantitative structure activity relationship",
            "MOA": "mode of action",
            "MIC": "minimum inhibitory concentration",
            "LD50": "median lethal dose",
            "ED50": "median effective dose",
        }

    def _load_synonym_map(self) -> Dict[str, str]:
        """Load synonym mappings."""
        return {
            "medication": "drug",
            "pharmaceutical": "drug",
            "medicine": "drug",
            "substance": "compound",
            "agent": "compound",
            "therapeutic": "drug",
            "remedy": "drug",
            "formula": "molecular formula",
            "weight": "molecular weight",
            "mass": "molecular weight",
        }


class AdvancedNLPProcessor:
    """Main NLP processor combining all components."""

    def __init__(self, ontology: ChemicalOntology):
        self.ontology = ontology
        self.entity_extractor = ChemicalEntityExtractor(ontology)
        self.intent_classifier = IntentClassifier(ontology)
        self.query_rewriter = QueryRewriter(ontology)

        # Statistics tracking
        self.processing_stats = {
            "total_queries": 0,
            "successful_processing": 0,
            "entity_extraction_success": 0,
            "intent_classification_success": 0,
            "average_confidence": 0.0,
        }

    def process_query(
        self, query: str, context: Optional[QueryContext] = None
    ) -> ProcessedQuery:
        """Process natural language query comprehensively."""

        if context is None:
            context = QueryContext()

        try:
            self.processing_stats["total_queries"] += 1

            # Clean and preprocess query
            cleaned_query = self._clean_query(query)

            # Extract entities
            entities = self.entity_extractor.extract_entities(cleaned_query, context)

            # Classify intent
            intent, intent_confidence = self.intent_classifier.classify_intent(
                cleaned_query, entities, context
            )

            # Rewrite query for better processing
            rewritten_query = self.query_rewriter.rewrite_query(
                cleaned_query, entities, intent
            )

            # Extract relationships and constraints
            relationships = self._extract_relationships(rewritten_query, entities)
            constraints = self._extract_constraints(rewritten_query, entities)

            # Determine query focus and expected answer type
            query_focus = self._determine_query_focus(rewritten_query, entities, intent)
            answer_type = self._determine_answer_type(intent, entities)

            # Create semantic parse
            semantic_parse = self._create_semantic_parse(
                rewritten_query, entities, relationships, constraints, intent
            )

            # Update statistics
            if entities:
                self.processing_stats["entity_extraction_success"] += 1
            if intent_confidence > 0.5:
                self.processing_stats["intent_classification_success"] += 1

            overall_confidence = self._calculate_overall_confidence(
                entities, intent_confidence, relationships, constraints
            )

            self.processing_stats["successful_processing"] += 1
            self.processing_stats["average_confidence"] = (
                self.processing_stats["average_confidence"]
                * (self.processing_stats["successful_processing"] - 1)
                + overall_confidence
            ) / self.processing_stats["successful_processing"]

            return ProcessedQuery(
                original_query=query,
                cleaned_query=cleaned_query,
                intent=intent,
                confidence=overall_confidence,
                entities=entities,
                relationships=relationships,
                constraints=constraints,
                query_focus=query_focus,
                expected_answer_type=answer_type,
                semantic_parse=semantic_parse,
                context=context,
            )

        except Exception as e:
            logging.error(f"NLP processing failed: {str(e)}")

            # Return minimal processed query
            return ProcessedQuery(
                original_query=query,
                cleaned_query=query,
                intent=QueryIntent.GENERAL_QUESTION,
                confidence=0.1,
                entities=[],
                relationships=[],
                constraints=[],
                query_focus="unknown",
                expected_answer_type="general",
                semantic_parse={},
                context=context or QueryContext(),
            )

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get NLP processing statistics."""
        return self.processing_stats.copy()

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text."""
        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", query.strip())

        # Fix common typos (simplified)
        typo_corrections = {
            r"\bwieght\b": "weight",
            r"\bstucture\b": "structure",
            r"\bcompund\b": "compound",
            r"\binteration\b": "interaction",
        }

        for typo, correction in typo_corrections.items():
            cleaned = re.sub(typo, correction, cleaned, flags=re.IGNORECASE)

        return cleaned

    def _extract_relationships(
        self, query: str, entities: List[ExtractedEntity]
    ) -> List[str]:
        """Extract relationship indicators from query."""
        relationships = []

        relationship_patterns = {
            "treats": r"\b(?:treat|cure|heal|remedy)\b",
            "causes": r"\b(?:cause|lead to|result in|induce)\b",
            "interacts_with": r"\b(?:interact|react|combine)\b",
            "similar_to": r"\b(?:similar|like|analogous|comparable)\b",
            "metabolizes": r"\b(?:metabolize|break down|process)\b",
            "inhibits": r"\b(?:inhibit|block|prevent|suppress)\b",
            "activates": r"\b(?:activate|stimulate|enhance|promote)\b",
        }

        for relation, pattern in relationship_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                relationships.append(relation)

        return relationships

    def _extract_constraints(
        self, query: str, entities: List[ExtractedEntity]
    ) -> List[Dict[str, Any]]:
        """Extract constraints and filters from query."""
        constraints = []

        # Numerical range constraints
        range_pattern = (
            r"(\w+)\s+(?:between|from)\s+(\d+(?:\.\d+)?)\s+(?:to|and)\s+(\d+(?:\.\d+)?)"
        )
        range_matches = re.finditer(range_pattern, query, re.IGNORECASE)

        for match in range_matches:
            constraints.append(
                {
                    "type": "range",
                    "property": match.group(1),
                    "min_value": float(match.group(2)),
                    "max_value": float(match.group(3)),
                }
            )

        # Greater than constraints
        gt_pattern = r"(\w+)\s+(?:greater than|more than|above|>)\s+(\d+(?:\.\d+)?)"
        gt_matches = re.finditer(gt_pattern, query, re.IGNORECASE)

        for match in gt_matches:
            constraints.append(
                {
                    "type": "greater_than",
                    "property": match.group(1),
                    "value": float(match.group(2)),
                }
            )

        # Less than constraints
        lt_pattern = r"(\w+)\s+(?:less than|below|under|<)\s+(\d+(?:\.\d+)?)"
        lt_matches = re.finditer(lt_pattern, query, re.IGNORECASE)

        for match in lt_matches:
            constraints.append(
                {
                    "type": "less_than",
                    "property": match.group(1),
                    "value": float(match.group(2)),
                }
            )

        # Categorical constraints
        if "approved" in query.lower():
            constraints.append(
                {
                    "type": "categorical",
                    "property": "approval_status",
                    "value": "approved",
                }
            )

        if "oral" in query.lower():
            constraints.append(
                {
                    "type": "categorical",
                    "property": "administration_route",
                    "value": "oral",
                }
            )

        return constraints

    def _determine_query_focus(
        self, query: str, entities: List[ExtractedEntity], intent: QueryIntent
    ) -> str:
        """Determine the main focus of the query."""

        if entities:
            # Focus on the most confident entity
            best_entity = max(entities, key=lambda x: x.confidence)
            return best_entity.text

        # Focus based on intent
        focus_map = {
            QueryIntent.FIND_ENTITIES: "entity_search",
            QueryIntent.COMPARE_ENTITIES: "entity_comparison",
            QueryIntent.FIND_RELATIONSHIPS: "relationship_analysis",
            QueryIntent.GET_PROPERTIES: "property_information",
            QueryIntent.EXPLAIN_MECHANISM: "mechanism_explanation",
            QueryIntent.PREDICT_INTERACTION: "interaction_prediction",
            QueryIntent.FIND_SIMILAR: "similarity_search",
            QueryIntent.ANALYZE_STRUCTURE: "structural_analysis",
        }

        return focus_map.get(intent, "general_inquiry")

    def _determine_answer_type(
        self, intent: QueryIntent, entities: List[ExtractedEntity]
    ) -> str:
        """Determine expected answer type."""

        answer_type_map = {
            QueryIntent.FIND_ENTITIES: "entity_list",
            QueryIntent.COMPARE_ENTITIES: "comparison_table",
            QueryIntent.FIND_RELATIONSHIPS: "relationship_graph",
            QueryIntent.GET_PROPERTIES: "property_values",
            QueryIntent.EXPLAIN_MECHANISM: "explanation_text",
            QueryIntent.PREDICT_INTERACTION: "prediction_result",
            QueryIntent.FIND_SIMILAR: "similarity_ranking",
            QueryIntent.ANALYZE_STRUCTURE: "structural_analysis",
            QueryIntent.SEARCH_LITERATURE: "literature_references",
            QueryIntent.GENERAL_QUESTION: "general_response",
        }

        return answer_type_map.get(intent, "general_response")

    def _create_semantic_parse(
        self,
        query: str,
        entities: List[ExtractedEntity],
        relationships: List[str],
        constraints: List[Dict[str, Any]],
        intent: QueryIntent,
    ) -> Dict[str, Any]:
        """Create structured semantic parse of the query."""

        return {
            "intent": intent.value,
            "entities": [
                {
                    "text": entity.text,
                    "type": entity.entity_type.value,
                    "confidence": entity.confidence,
                    "ontology_uri": entity.ontology_uri,
                }
                for entity in entities
            ],
            "relationships": relationships,
            "constraints": constraints,
            "logical_structure": self._analyze_logical_structure(query),
            "temporal_indicators": self._extract_temporal_indicators(query),
            "negations": self._extract_negations(query),
            "quantifiers": self._extract_quantifiers(query),
        }

    def _analyze_logical_structure(self, query: str) -> Dict[str, Any]:
        """Analyze logical structure of query."""
        structure = {
            "conjunction": bool(
                re.search(r"\b(?:and|also|additionally)\b", query, re.IGNORECASE)
            ),
            "disjunction": bool(
                re.search(r"\b(?:or|alternatively)\b", query, re.IGNORECASE)
            ),
            "negation": bool(
                re.search(r"\b(?:not|no|without|except)\b", query, re.IGNORECASE)
            ),
            "conditional": bool(
                re.search(r"\b(?:if|when|unless|provided)\b", query, re.IGNORECASE)
            ),
            "causal": bool(
                re.search(
                    r"\b(?:because|since|due to|caused by)\b", query, re.IGNORECASE
                )
            ),
        }

        return structure

    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from query."""
        temporal_patterns = [
            r"\b(?:before|after|during|while|when)\b",
            r"\b(?:recent|latest|current|new|old)\b",
            r"\b(?:today|yesterday|tomorrow|now|then)\b",
            r"\b\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|later)\b",
        ]

        indicators = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            indicators.extend(matches)

        return indicators

    def _extract_negations(self, query: str) -> List[str]:
        """Extract negation indicators."""
        negation_patterns = [
            r"\b(?:not|no|without|except|excluding|other than)\b",
            r"\b(?:never|rarely|seldom|hardly)\b",
            r"\b(?:non-|un-|in-|dis-)\w+",
        ]

        negations = []
        for pattern in negation_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            negations.extend(matches)

        return negations

    def _extract_quantifiers(self, query: str) -> List[Dict[str, Any]]:
        """Extract quantifiers and their scope."""
        quantifiers = []

        # Numerical quantifiers
        num_pattern = r"\b(\d+(?:\.\d+)?)\s+(\w+)"
        for match in re.finditer(num_pattern, query):
            quantifiers.append(
                {
                    "type": "numerical",
                    "value": float(match.group(1)),
                    "unit": match.group(2),
                }
            )

        # Logical quantifiers
        logical_patterns = {
            "all": r"\b(?:all|every|each)\b",
            "some": r"\b(?:some|several|many)\b",
            "few": r"\b(?:few|little|rare)\b",
            "most": r"\b(?:most|majority|mainly)\b",
            "none": r"\b(?:none|zero|nothing)\b",
        }

        for quantifier_type, pattern in logical_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                quantifiers.append({"type": "logical", "quantifier": quantifier_type})

        return quantifiers

    def _calculate_overall_confidence(
        self,
        entities: List[ExtractedEntity],
        intent_confidence: float,
        relationships: List[str],
        constraints: List[Dict[str, Any]],
    ) -> float:
        """Calculate overall processing confidence."""

        # Entity extraction confidence
        entity_confidence = (
            sum(entity.confidence for entity in entities) / len(entities)
            if entities
            else 0.3
        )

        # Component presence bonuses
        relationship_bonus = 0.1 if relationships else 0.0
        constraint_bonus = 0.1 if constraints else 0.0

        # Weighted combination
        overall = (
            0.4 * entity_confidence
            + 0.4 * intent_confidence
            + 0.1 * relationship_bonus
            + 0.1 * constraint_bonus
        )

        return min(overall, 1.0)

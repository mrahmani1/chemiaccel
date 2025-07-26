"""
Text and Document Processing for Chemical NLP

GPU-accelerated text processing and document analysis for chemical
information extraction from scientific literature.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .backends import NLPBackend, get_nlp_backend
from .entity_recognition import ChemicalEntityRecognizer, EntityExtraction
from .schema_mapper import ICKGSchemaMapper, MappingResult
from .table_extractor import ExperimentalDataExtractor, ExtractedTable, TableExtractor

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of chemical documents that can be processed."""

    RESEARCH_PAPER = "research_paper"
    PATENT = "patent"
    PROTOCOL = "protocol"
    THESIS = "thesis"
    REPORT = "report"
    BOOK_CHAPTER = "book_chapter"
    REVIEW = "review"
    CONFERENCE_PAPER = "conference_paper"
    PREPRINT = "preprint"
    SUPPLEMENTARY = "supplementary"
    UNKNOWN = "unknown"


@dataclass
class ProcessingConfig:
    """Configuration for text processing."""

    backend: str = "auto"
    enable_entity_recognition: bool = True
    enable_table_extraction: bool = True
    enable_relationship_extraction: bool = True
    enable_schema_mapping: bool = True
    normalize_text: bool = True
    remove_references: bool = False
    extract_figures: bool = False
    confidence_threshold: float = 0.5
    max_text_length: int = 1000000  # 1MB limit
    chunk_size: int = 5000  # Characters per chunk
    overlap_size: int = 500  # Overlap between chunks


@dataclass
class ProcessingResult:
    """Result of document processing."""

    document_type: DocumentType
    entities: List[EntityExtraction] = field(default_factory=list)
    relationships: List = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    experimental_records: List = field(default_factory=list)
    mapping_result: Optional[MappingResult] = None
    document_structure: Optional[Dict[str, Any]] = None
    processing_statistics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class TextProcessor:
    """
    GPU-accelerated text processing for chemical documents.

    Provides text normalization, cleaning, and preprocessing optimized
    for chemical and scientific literature.
    """

    def __init__(self, backend: Optional[str] = None):
        self.backend = get_nlp_backend(backend)

        # Chemical-specific cleaning patterns
        self.cleaning_patterns = [
            # Remove excessive whitespace
            (r"\s+", " "),
            # Fix common OCR errors in chemical text
            (r"\bl\b(?=\d)", "1"),  # Replace 'l' with '1' before numbers
            (r"\bO\b(?=\d)", "0"),  # Replace 'O' with '0' before numbers
            # Standardize chemical notation
            (r"(\d+)\s*-\s*(\w)", r"\1-\2"),  # Fix spacing in compound names
            (r"(\w)\s*-\s*(\d+)", r"\1-\2"),
            # Fix degree symbols
            (r"(\d+)\s*degrees?\s*C", r"\1°C"),
            (r"(\d+)\s*deg\s*C", r"\1°C"),
        ]

        # Reference patterns for removal
        self.reference_patterns = [
            r"\[\d+(?:,\s*\d+)*\]",  # [1], [1,2,3]
            r"\(\d+(?:,\s*\d+)*\)",  # (1), (1,2,3)
            r"\b\d+\.\s*[A-Z].*?\(\d{4}\)",  # Citation format: "1. Author et al. (2023)"
        ]

        # Chemical unit standardization
        self.unit_standardizations = {
            "deg C": "°C",
            "degrees C": "°C",
            "degree C": "°C",
            "degC": "°C",
            "mol/L": "M",
            "molar": "M",
            "millimolar": "mM",
            "micromolar": "µM",
            "nanomolar": "nM",
            "gram": "g",
            "milligram": "mg",
            "microgram": "µg",
            "liter": "L",
            "milliliter": "mL",
            "microliter": "µL",
        }

        logger.info(
            f"Initialized TextProcessor with {self.backend.backend_type.value} backend"
        )

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for chemical processing.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            text = re.sub(pattern, replacement, text)

        # Standardize units
        for old_unit, new_unit in self.unit_standardizations.items():
            text = re.sub(
                rf"\b{re.escape(old_unit)}\b", new_unit, text, flags=re.IGNORECASE
            )

        # Clean up whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Multiple newlines to double
        text = text.strip()

        return text

    def remove_references(self, text: str) -> str:
        """Remove reference citations from text."""
        for pattern in self.reference_patterns:
            text = re.sub(pattern, "", text)

        # Clean up any extra whitespace created by removal
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def segment_text(
        self, text: str, max_length: int = 5000, overlap: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Segment long text into manageable chunks with overlap.

        Args:
            text: Input text to segment
            max_length: Maximum length per segment
            overlap: Overlap between segments

        Returns:
            List of text segments with metadata
        """
        if len(text) <= max_length:
            return [
                {
                    "text": text,
                    "start_idx": 0,
                    "end_idx": len(text),
                    "segment_id": 0,
                    "total_segments": 1,
                }
            ]

        segments = []
        start = 0
        segment_id = 0

        while start < len(text):
            end = min(start + max_length, len(text))

            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence ending within the last 200 characters
                sentence_end = text.rfind(".", start + max_length - 200, end)
                if sentence_end != -1 and sentence_end > start + max_length // 2:
                    end = sentence_end + 1

            segment_text = text[start:end]

            segments.append(
                {
                    "text": segment_text,
                    "start_idx": start,
                    "end_idx": end,
                    "segment_id": segment_id,
                    "total_segments": 0,  # Will be updated after loop
                }
            )

            # Move start position with overlap
            start = max(start + max_length - overlap, end)
            segment_id += 1

        # Update total segments count
        for segment in segments:
            segment["total_segments"] = len(segments)

        return segments

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract major sections from a scientific document.

        Args:
            text: Full document text

        Returns:
            Dictionary mapping section names to content
        """
        sections = {}

        # Common section headers in scientific papers
        section_patterns = {
            "abstract": r"(?:^|\n)\s*(?:abstract|summary)\s*(?:\n|$)",
            "introduction": r"(?:^|\n)\s*(?:introduction|background)\s*(?:\n|$)",
            "methods": r"(?:^|\n)\s*(?:methods?|methodology|experimental|materials and methods)\s*(?:\n|$)",
            "results": r"(?:^|\n)\s*(?:results?|findings)\s*(?:\n|$)",
            "discussion": r"(?:^|\n)\s*(?:discussion|analysis)\s*(?:\n|$)",
            "conclusion": r"(?:^|\n)\s*(?:conclusion|summary|concluding remarks)\s*(?:\n|$)",
            "references": r"(?:^|\n)\s*(?:references|bibliography|citations)\s*(?:\n|$)",
            "experimental": r"(?:^|\n)\s*(?:experimental section|experimental procedures)\s*(?:\n|$)",
            "synthesis": r"(?:^|\n)\s*(?:synthesis|synthetic procedures?)\s*(?:\n|$)",
        }

        # Find section boundaries
        section_positions = []
        for section_name, pattern in section_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                section_positions.append(
                    {
                        "name": section_name,
                        "start": match.start(),
                        "end": match.end(),
                        "header": match.group().strip(),
                    }
                )

        # Sort by position
        section_positions.sort(key=lambda x: x["start"])

        # Extract section content
        for i, section in enumerate(section_positions):
            start_pos = section["end"]

            # Find end position (start of next section or end of text)
            if i + 1 < len(section_positions):
                end_pos = section_positions[i + 1]["start"]
            else:
                end_pos = len(text)

            section_content = text[start_pos:end_pos].strip()
            if section_content:
                sections[section["name"]] = section_content

        return sections

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.

        Args:
            text: Input text

        Returns:
            Detected language code (e.g., 'en', 'de', 'fr')
        """
        # Simple language detection based on common words
        # In practice, would use a proper language detection library

        sample = text[:1000].lower()  # Use first 1000 characters

        # English indicators
        english_words = [
            "the",
            "and",
            "of",
            "to",
            "in",
            "is",
            "was",
            "were",
            "with",
            "for",
        ]
        english_count = sum(
            len(re.findall(rf"\b{word}\b", sample)) for word in english_words
        )

        # German indicators
        german_words = [
            "der",
            "die",
            "das",
            "und",
            "von",
            "zu",
            "mit",
            "bei",
            "für",
            "ist",
        ]
        german_count = sum(
            len(re.findall(rf"\b{word}\b", sample)) for word in german_words
        )

        # French indicators
        french_words = [
            "le",
            "la",
            "les",
            "de",
            "du",
            "des",
            "et",
            "avec",
            "pour",
            "dans",
        ]
        french_count = sum(
            len(re.findall(rf"\b{word}\b", sample)) for word in french_words
        )

        # Return language with highest count
        lang_scores = {"en": english_count, "de": german_count, "fr": french_count}
        detected_lang = max(lang_scores.items(), key=lambda x: x[1])

        return detected_lang[0] if detected_lang[1] > 0 else "unknown"

    def clean_chemical_notation(self, text: str) -> str:
        """
        Clean and standardize chemical notation in text.

        Args:
            text: Text containing chemical notation

        Returns:
            Text with cleaned chemical notation
        """
        # Fix subscripts and superscripts
        subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

        text = text.translate(subscript_map)
        text = text.translate(superscript_map)

        # Standardize chemical formulas
        # Fix spacing around chemical bonds
        text = re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", text)
        text = re.sub(r"(\w)\s*=\s*(\w)", r"\1=\2", text)
        text = re.sub(r"(\w)\s*≡\s*(\w)", r"\1≡\2", text)

        # Fix spacing in chemical names
        text = re.sub(
            r"(\d+)\s*,\s*(\d+)", r"\1,\2", text
        )  # 1, 2-dimethyl -> 1,2-dimethyl

        # Standardize stereochemistry notation
        text = re.sub(r"\(([RS])\)", r"(\1)", text)
        text = re.sub(r"\(([±])\)", r"(±)", text)

        return text

    def extract_chemical_formulas(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract chemical formulas from text.

        Args:
            text: Input text

        Returns:
            List of chemical formulas with positions
        """
        formulas = []

        # Pattern for molecular formulas
        formula_pattern = r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*\b"

        matches = re.finditer(formula_pattern, text)
        for match in matches:
            formula = match.group()

            # Validate that it looks like a real chemical formula
            if self._is_valid_chemical_formula(formula):
                formulas.append(
                    {
                        "formula": formula,
                        "start_idx": match.start(),
                        "end_idx": match.end(),
                        "context": text[max(0, match.start() - 50) : match.end() + 50],
                    }
                )

        return formulas

    def _is_valid_chemical_formula(self, formula: str) -> bool:
        """Check if a string looks like a valid chemical formula."""
        # Basic validation rules
        if len(formula) < 2:
            return False

        # Must start with uppercase letter
        if not formula[0].isupper():
            return False

        # Should contain at least one number or multiple elements
        has_number = any(c.isdigit() for c in formula)
        has_multiple_elements = sum(1 for c in formula if c.isupper()) > 1

        if not (has_number or has_multiple_elements):
            return False

        # Exclude common English words that might match the pattern
        common_words = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "And",
            "But",
            "Or",
            "If",
            "When",
            "Where",
        }
        if formula in common_words:
            return False

        return True


class DocumentProcessor:
    """
    Comprehensive document processor for chemical literature.

    Orchestrates entity recognition, table extraction, relationship extraction,
    and schema mapping for complete document processing.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()

        # Initialize components
        self.text_processor = TextProcessor(self.config.backend)

        if self.config.enable_entity_recognition:
            self.entity_recognizer = ChemicalEntityRecognizer()

        if self.config.enable_table_extraction:
            self.table_extractor = TableExtractor(self.config.backend)
            self.experimental_extractor = ExperimentalDataExtractor(self.config.backend)

        if self.config.enable_schema_mapping:
            self.schema_mapper = ICKGSchemaMapper()

        logger.info("Initialized DocumentProcessor")

    def process_document(
        self, text: str, document_type: Optional[DocumentType] = None
    ) -> ProcessingResult:
        """
        Process a complete chemical document.

        Args:
            text: Full document text
            document_type: Type of document (optional, will be detected)

        Returns:
            ProcessingResult containing all extracted information
        """
        result = ProcessingResult(
            document_type=document_type or self._detect_document_type(text)
        )

        try:
            # Text preprocessing
            if self.config.normalize_text:
                text = self.text_processor.normalize_text(text)

            if self.config.remove_references:
                text = self.text_processor.remove_references(text)

            # Check text length
            if len(text) > self.config.max_text_length:
                result.warnings.append(
                    f"Text length ({len(text)}) exceeds maximum ({self.config.max_text_length})"
                )
                text = text[: self.config.max_text_length]

            # Document structure analysis
            result.document_structure = self._analyze_document_structure(text)

            # Process in chunks if necessary
            if len(text) > self.config.chunk_size:
                chunks = self.text_processor.segment_text(
                    text, self.config.chunk_size, self.config.overlap_size
                )
                result = self._process_chunks(chunks, result)
            else:
                result = self._process_single_text(text, result)

            # Schema mapping
            if self.config.enable_schema_mapping and self.schema_mapper:
                result.mapping_result = self.schema_mapper.create_complete_mapping(
                    result.entities,
                    result.relationships,
                    result.tables,
                    result.experimental_records,
                )

            # Generate processing statistics
            result.processing_statistics = self._generate_statistics(result)

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            result.errors.append(f"Processing error: {str(e)}")

        return result

    def _detect_document_type(self, text: str) -> DocumentType:
        """Detect the type of document based on content."""
        text_lower = text.lower()

        # Patent indicators
        if any(
            term in text_lower for term in ["patent", "invention", "claim", "prior art"]
        ):
            return DocumentType.PATENT

        # Research paper indicators
        if any(
            term in text_lower
            for term in ["abstract", "introduction", "methodology", "conclusion"]
        ):
            return DocumentType.RESEARCH_PAPER

        # Protocol indicators
        if any(
            term in text_lower for term in ["protocol", "procedure", "step", "method"]
        ):
            return DocumentType.PROTOCOL

        # Thesis indicators
        if any(term in text_lower for term in ["thesis", "dissertation", "chapter"]):
            return DocumentType.THESIS

        # Review indicators
        if any(term in text_lower for term in ["review", "survey", "overview"]):
            return DocumentType.REVIEW

        return DocumentType.UNKNOWN

    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the overall structure of the document."""
        structure = {
            "sections": self.text_processor.extract_sections(text),
            "language": self.text_processor.detect_language(text),
            "length": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.split("\n")),
            "chemical_formulas": self.text_processor.extract_chemical_formulas(text),
        }

        # Use backend for additional structure analysis
        try:
            backend_structure = self.text_processor.backend.analyze_document_structure(
                text
            )
            structure.update(
                {
                    "detected_sections": backend_structure.sections,
                    "detected_tables": backend_structure.tables,
                    "detected_figures": backend_structure.figures,
                    "detected_references": backend_structure.references,
                }
            )
        except Exception as e:
            logger.debug(f"Backend structure analysis failed: {e}")

        return structure

    def _process_chunks(
        self, chunks: List[Dict[str, Any]], result: ProcessingResult
    ) -> ProcessingResult:
        """Process document chunks and combine results."""
        all_entities = []
        all_relationships = []

        for chunk in chunks:
            chunk_result = self._process_single_text(
                chunk["text"], ProcessingResult(document_type=result.document_type)
            )

            # Adjust entity positions based on chunk offset
            for entity in chunk_result.entities:
                entity.start_idx += chunk["start_idx"]
                entity.end_idx += chunk["start_idx"]

            all_entities.extend(chunk_result.entities)
            all_relationships.extend(chunk_result.relationships)
            result.tables.extend(chunk_result.tables)
            result.experimental_records.extend(chunk_result.experimental_records)

        # Deduplicate entities across chunks
        result.entities = self._deduplicate_entities(all_entities)
        result.relationships = all_relationships

        return result

    def _process_single_text(
        self, text: str, result: ProcessingResult
    ) -> ProcessingResult:
        """Process a single text segment."""
        # Entity recognition
        if self.config.enable_entity_recognition and hasattr(self, "entity_recognizer"):
            try:
                result.entities = self.entity_recognizer.extract_entities(text)
                # Filter by confidence
                result.entities = [
                    e
                    for e in result.entities
                    if e.confidence >= self.config.confidence_threshold
                ]
            except Exception as e:
                logger.error(f"Entity recognition failed: {e}")
                result.errors.append(f"Entity recognition error: {str(e)}")

        # Relationship extraction
        if self.config.enable_relationship_extraction and hasattr(
            self, "entity_recognizer"
        ):
            try:
                result.relationships = self.entity_recognizer.extract_relationships(
                    text, result.entities
                )
            except Exception as e:
                logger.error(f"Relationship extraction failed: {e}")
                result.errors.append(f"Relationship extraction error: {str(e)}")

        # Table extraction
        if self.config.enable_table_extraction and hasattr(self, "table_extractor"):
            try:
                result.tables = self.table_extractor.extract_tables(text)

                # Extract experimental data
                if hasattr(self, "experimental_extractor"):
                    exp_data = self.experimental_extractor.extract_from_document(text)
                    result.experimental_records = exp_data.get(
                        "experimental_records", []
                    )

            except Exception as e:
                logger.error(f"Table extraction failed: {e}")
                result.errors.append(f"Table extraction error: {str(e)}")

        return result

    def _deduplicate_entities(
        self, entities: List[EntityExtraction]
    ) -> List[EntityExtraction]:
        """Remove duplicate entities from chunk processing."""
        unique_entities = []
        seen_entities = set()

        for entity in entities:
            # Create a signature for the entity
            signature = (
                entity.text.lower(),
                entity.label,
                entity.start_idx,
                entity.end_idx,
            )

            if signature not in seen_entities:
                unique_entities.append(entity)
                seen_entities.add(signature)

        return unique_entities

    def _generate_statistics(self, result: ProcessingResult) -> Dict[str, Any]:
        """Generate comprehensive processing statistics."""
        stats = {
            "document_type": result.document_type.value,
            "total_entities": len(result.entities),
            "total_relationships": len(result.relationships),
            "total_tables": len(result.tables),
            "total_experimental_records": len(result.experimental_records),
            "processing_errors": len(result.errors),
            "processing_warnings": len(result.warnings),
        }

        # Entity statistics
        if result.entities:
            entity_types = {}
            confidence_scores = []

            for entity in result.entities:
                entity_types[entity.label] = entity_types.get(entity.label, 0) + 1
                confidence_scores.append(entity.confidence)

            stats["entity_types"] = entity_types
            stats["entity_confidence"] = {
                "mean": sum(confidence_scores) / len(confidence_scores),
                "min": min(confidence_scores),
                "max": max(confidence_scores),
            }

        # Table statistics
        if result.tables:
            table_types = {}
            for table in result.tables:
                table_types[table.table_type.value] = (
                    table_types.get(table.table_type.value, 0) + 1
                )
            stats["table_types"] = table_types

        # Mapping statistics
        if result.mapping_result:
            stats["mapping"] = result.mapping_result.mapping_statistics

        # Document structure statistics
        if result.document_structure:
            stats["document_structure"] = {
                "sections_found": len(result.document_structure.get("sections", {})),
                "language": result.document_structure.get("language", "unknown"),
                "chemical_formulas": len(
                    result.document_structure.get("chemical_formulas", [])
                ),
            }

        return stats

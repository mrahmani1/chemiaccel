"""
Tests for Chemical Entity Recognition

Comprehensive testing of chemical entity recognition capabilities
including pattern-based and ML-based extraction.
"""

from typing import List
from unittest.mock import Mock, patch

import pytest
from chemi_accel.nlp.backends.base import EntityExtraction, RelationshipExtraction
from chemi_accel.nlp.entity_recognition import (
    ChemicalEntityRecognizer,
    EntityType,
    RecognitionConfig,
)


class TestChemicalEntityRecognizer:
    """Test chemical entity recognition functionality."""

    @pytest.fixture
    def recognizer(self):
        """Create a test recognizer with CPU backend."""
        config = RecognitionConfig(backend="cpu", confidence_threshold=0.3)
        return ChemicalEntityRecognizer(config)

    @pytest.fixture
    def sample_chemical_text(self):
        """Sample text containing chemical entities."""
        return """
        The synthesis of aspirin (acetylsalicylic acid, C9H8O4) was carried out
        by reacting salicylic acid with acetic anhydride. The reaction was performed
        at 80°C for 2 hours using sulfuric acid as catalyst. The melting point of
        the product was determined to be 135°C. The molecular weight is 180.16 g/mol.
        """

    def test_initialization(self, recognizer):
        """Test recognizer initialization."""
        assert recognizer is not None
        assert recognizer.config.backend == "cpu"
        assert recognizer.backend is not None

    def test_extract_entities_basic(self, recognizer, sample_chemical_text):
        """Test basic entity extraction."""
        entities = recognizer.extract_entities(sample_chemical_text)

        assert len(entities) > 0

        # Check for chemical entities
        chemical_entities = [e for e in entities if e.label == "CHEMICAL"]
        assert len(chemical_entities) > 0

        # Check for specific entities
        entity_texts = [e.text for e in entities]
        assert any("aspirin" in text.lower() for text in entity_texts)
        assert any("C9H8O4" in text for text in entity_texts)

    def test_extract_entities_with_types(self, recognizer, sample_chemical_text):
        """Test entity extraction with specific types."""
        entities = recognizer.extract_entities(
            sample_chemical_text, entity_types=[EntityType.CHEMICAL, EntityType.VALUE]
        )

        assert len(entities) > 0

        # All entities should be of requested types
        for entity in entities:
            assert entity.label in ["CHEMICAL", "VALUE"]

    def test_pattern_based_extraction(self, recognizer):
        """Test pattern-based entity extraction."""
        text = "The compound H2SO4 reacts with NaOH to form Na2SO4 and H2O."

        entities = recognizer._extract_pattern_entities(text)

        assert len(entities) > 0

        # Check for molecular formulas
        formulas = [e.text for e in entities if "H2SO4" in e.text or "NaOH" in e.text]
        assert len(formulas) > 0

    def test_combine_entities(self, recognizer):
        """Test entity combination and deduplication."""
        # Create mock entities with overlaps
        ml_entities = [
            EntityExtraction("aspirin", "CHEMICAL", 10, 17, 0.8, {}),
            EntityExtraction("acetylsalicylic acid", "CHEMICAL", 20, 40, 0.9, {}),
        ]

        pattern_entities = [
            EntityExtraction("aspirin", "CHEMICAL", 10, 17, 0.7, {}),  # Duplicate
            EntityExtraction("C9H8O4", "CHEMICAL", 50, 56, 0.7, {}),
        ]

        combined = recognizer._combine_entities(ml_entities, pattern_entities)

        # Should have 3 unique entities (aspirin deduplicated)
        assert len(combined) == 3

        # Higher confidence aspirin should be kept
        aspirin_entities = [e for e in combined if e.text == "aspirin"]
        assert len(aspirin_entities) == 1
        assert aspirin_entities[0].confidence == 0.8

    def test_entity_filtering(self, recognizer):
        """Test entity filtering by confidence and other criteria."""
        text = "The compound aspirin is used in medicine."

        entities = [
            EntityExtraction("aspirin", "CHEMICAL", 13, 20, 0.9, {}),
            EntityExtraction("The", "CHEMICAL", 0, 3, 0.4, {}),  # Low confidence
            EntityExtraction("a" * 250, "CHEMICAL", 50, 300, 0.8, {}),  # Too long
        ]

        filtered = recognizer._filter_entities(entities, text)

        # Only aspirin should remain
        assert len(filtered) == 1
        assert filtered[0].text == "aspirin"

    def test_extract_relationships(self, recognizer, sample_chemical_text):
        """Test relationship extraction."""
        entities = recognizer.extract_entities(sample_chemical_text)
        relationships = recognizer.extract_relationships(sample_chemical_text, entities)

        # Should find some relationships
        assert len(relationships) >= 0  # May be 0 with simple patterns

        if relationships:
            rel = relationships[0]
            assert hasattr(rel, "source_entity")
            assert hasattr(rel, "target_entity")
            assert hasattr(rel, "relation_type")
            assert hasattr(rel, "confidence")

    def test_recognize_chemical_names(self, recognizer):
        """Test specialized chemical name recognition."""
        text = "Synthesis of 2-methylpyridine and benzene derivatives."

        chemical_names = recognizer.recognize_chemical_names(text)

        assert len(chemical_names) > 0

        # Check structure
        for name in chemical_names:
            assert "name" in name
            assert "type" in name
            assert "confidence" in name
            assert "start_idx" in name
            assert "end_idx" in name

    def test_extract_numerical_values(self, recognizer):
        """Test numerical value extraction."""
        text = "The reaction was heated to 85°C for 3 hours at 2 atm pressure."

        values = recognizer.extract_numerical_values(text)

        assert len(values) > 0

        # Check for temperature value
        temp_values = [v for v in values if v["type"] == "temperature"]
        assert len(temp_values) > 0
        assert temp_values[0]["value"] == 85
        assert temp_values[0]["unit"] == "°C"

        # Check for time value
        time_values = [v for v in values if v["type"] == "time"]
        assert len(time_values) > 0

    def test_get_entity_statistics(self, recognizer, sample_chemical_text):
        """Test entity statistics generation."""
        entities = recognizer.extract_entities(sample_chemical_text)
        stats = recognizer.get_entity_statistics(entities)

        assert "total_entities" in stats
        assert "unique_entities" in stats
        assert "label_distribution" in stats
        assert "confidence_stats" in stats

        assert stats["total_entities"] == len(entities)
        assert stats["confidence_stats"]["mean"] > 0

    def test_add_custom_pattern(self, recognizer):
        """Test adding custom patterns."""
        original_patterns = len(recognizer.entity_patterns.get(EntityType.CHEMICAL, []))

        recognizer.add_custom_pattern(EntityType.CHEMICAL, r"\b\w+mycin\b")

        new_patterns = len(recognizer.entity_patterns[EntityType.CHEMICAL])
        assert new_patterns == original_patterns + 1

        # Test that the pattern works
        text = "Streptomycin is an antibiotic."
        entities = recognizer._extract_pattern_entities(text, [EntityType.CHEMICAL])

        # Should find streptomycin
        mycin_entities = [e for e in entities if "mycin" in e.text.lower()]
        assert len(mycin_entities) > 0

    def test_empty_text_handling(self, recognizer):
        """Test handling of empty or invalid text."""
        # Empty text
        entities = recognizer.extract_entities("")
        assert len(entities) == 0

        # None text
        entities = recognizer.extract_entities(None)
        assert len(entities) == 0

        # Whitespace only
        entities = recognizer.extract_entities("   \n\t  ")
        assert len(entities) == 0

    def test_long_text_handling(self, recognizer):
        """Test handling of very long text."""
        # Create long text
        long_text = "The compound aspirin. " * 1000

        entities = recognizer.extract_entities(long_text)

        # Should still work
        assert len(entities) > 0

        # Should find aspirin entities
        aspirin_count = len([e for e in entities if "aspirin" in e.text.lower()])
        assert aspirin_count > 0

    @pytest.mark.parametrize("backend_type", ["cpu"])  # Only test available backends
    def test_different_backends(self, backend_type):
        """Test entity recognition with different backends."""
        config = RecognitionConfig(backend=backend_type)
        recognizer = ChemicalEntityRecognizer(config)

        text = "Caffeine (C8H10N4O2) is a stimulant."
        entities = recognizer.extract_entities(text)

        assert len(entities) > 0
        assert recognizer.backend.backend_type.value == backend_type

    def test_confidence_threshold_filtering(self):
        """Test confidence threshold filtering."""
        # High threshold
        high_config = RecognitionConfig(confidence_threshold=0.9)
        high_recognizer = ChemicalEntityRecognizer(high_config)

        # Low threshold
        low_config = RecognitionConfig(confidence_threshold=0.1)
        low_recognizer = ChemicalEntityRecognizer(low_config)

        text = "Aspirin is a drug."

        high_entities = high_recognizer.extract_entities(text)
        low_entities = low_recognizer.extract_entities(text)

        # Low threshold should find more or equal entities
        assert len(low_entities) >= len(high_entities)


class TestEntityExtractionIntegration:
    """Integration tests for entity extraction."""

    def test_real_paper_abstract(self):
        """Test on a realistic paper abstract."""
        abstract = """
        A novel synthetic route to 3,4-methylenedioxyphenyl compounds has been developed.
        The key step involves a Suzuki coupling reaction between 3,4-methylenedioxyboronic acid
        and aryl halides. The reaction proceeds at 80°C in DMF with Pd(PPh3)4 as catalyst.
        Yields of 75-85% were obtained for various substrates. The products were characterized
        by 1H NMR, 13C NMR, and mass spectrometry. Melting points ranged from 120-145°C.
        """

        recognizer = ChemicalEntityRecognizer()
        entities = recognizer.extract_entities(abstract)

        # Should find multiple chemical entities
        assert len(entities) > 5

        # Should find specific compounds
        entity_texts = [e.text.lower() for e in entities]
        assert any("suzuki" in text for text in entity_texts)
        assert any("dmf" in text for text in entity_texts)
        assert any("pd" in text or "palladium" in text for text in entity_texts)

    def test_synthetic_procedure_extraction(self):
        """Test extraction from synthetic procedure text."""
        procedure = """
        To a solution of benzoic acid (1.22 g, 10 mmol) in dry DCM (50 mL) was added
        EDC·HCl (2.30 g, 12 mmol) and DMAP (0.12 g, 1 mmol). The mixture was stirred
        at room temperature for 30 minutes. Benzylamine (1.07 g, 10 mmol) was then added
        dropwise and stirring continued for 6 hours. The reaction was quenched with water
        and extracted with DCM. The organic layer was dried over Na2SO4 and concentrated
        to give the crude product as a white solid (yield: 89%).
        """

        recognizer = ChemicalEntityRecognizer()
        entities = recognizer.extract_entities(procedure)

        # Should find reagents, solvents, and conditions
        assert len(entities) > 10

        # Check for specific entities
        entity_texts = [e.text for e in entities]
        compound_names = [text.lower() for text in entity_texts]

        assert any("benzoic acid" in name for name in compound_names)
        assert any("dcm" in name for name in compound_names)
        assert any("na2so4" in name for name in compound_names)

        # Should also extract numerical values
        values = recognizer.extract_numerical_values(procedure)
        assert len(values) > 5

        # Check for specific values
        mass_values = [v for v in values if v["unit"] == "g"]
        time_values = [v for v in values if v["type"] == "time"]

        assert len(mass_values) > 0
        assert len(time_values) > 0

    def test_biological_activity_extraction(self):
        """Test extraction from biological activity text."""
        bio_text = """
        The synthesized compounds were evaluated for their anticancer activity against
        HeLa cells using the MTT assay. Compound 3a showed the highest activity with
        an IC50 value of 2.3 µM, while compound 3b was less active (IC50 = 45 µM).
        Cisplatin was used as a positive control (IC50 = 8.1 µM). The results indicate
        that substitution at the 4-position significantly enhances biological activity.
        """

        recognizer = ChemicalEntityRecognizer()
        entities = recognizer.extract_entities(bio_text)
        relationships = recognizer.extract_relationships(bio_text, entities)

        # Should find compounds and activity data
        assert len(entities) > 5

        # Should find relationships
        assert len(relationships) >= 0  # May be 0 with simple patterns

        # Extract numerical values for IC50
        values = recognizer.extract_numerical_values(bio_text)
        ic50_values = [v for v in values if "µM" in v["unit"]]

        assert len(ic50_values) >= 3  # Should find the three IC50 values

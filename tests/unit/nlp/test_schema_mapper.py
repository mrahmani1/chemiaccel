"""
Tests for IC-KG Schema Mapping

Testing the mapping of NLP-extracted entities and relationships to the
Integrated Chemistry Knowledge Graph (IC-KG) schema.
"""

import uuid
from unittest.mock import Mock, patch

import pytest
from chemi_accel.knowledge_graph.ontology import ChemicalEntity, ChemicalRelationship
from chemi_accel.knowledge_graph.schema import EntityType as KGEntityType
from chemi_accel.knowledge_graph.schema import PropertyType, RelationshipType
from chemi_accel.nlp.backends.base import EntityExtraction, RelationshipExtraction
from chemi_accel.nlp.schema_mapper import ICKGSchemaMapper, MappingConfig, MappingResult
from chemi_accel.nlp.table_extractor import (
    ExperimentalRecord,
    ExtractedTable,
    TableCell,
    TableType,
)


class TestICKGSchemaMapper:
    """Test IC-KG schema mapping functionality."""

    @pytest.fixture
    def mapper(self):
        """Create a test schema mapper."""
        config = MappingConfig(confidence_threshold=0.5)
        return ICKGSchemaMapper(config)

    @pytest.fixture
    def sample_entities(self):
        """Sample extracted entities for testing."""
        return [
            EntityExtraction(
                text="aspirin",
                label="CHEMICAL",
                start_idx=10,
                end_idx=17,
                confidence=0.9,
                properties={"method": "pattern_based", "context": "aspirin is a drug"},
            ),
            EntityExtraction(
                text="C9H8O4",
                label="CHEMICAL",
                start_idx=50,
                end_idx=56,
                confidence=0.8,
                properties={"method": "ml_model"},
            ),
            EntityExtraction(
                text="synthesis",
                label="REACTION",
                start_idx=100,
                end_idx=109,
                confidence=0.7,
                properties={"method": "pattern_based"},
            ),
            EntityExtraction(
                text="low_confidence",
                label="CHEMICAL",
                start_idx=200,
                end_idx=214,
                confidence=0.3,  # Below threshold
                properties={"method": "ml_model"},
            ),
        ]

    @pytest.fixture
    def sample_relationships(self):
        """Sample extracted relationships for testing."""
        entity1 = EntityExtraction("aspirin", "CHEMICAL", 10, 17, 0.9, {})
        entity2 = EntityExtraction("pain", "PROPERTY", 20, 24, 0.8, {})

        return [
            RelationshipExtraction(
                source_entity=entity1,
                target_entity=entity2,
                relation_type="INHIBITS",
                confidence=0.8,
                context="aspirin inhibits pain",
                properties={"method": "pattern_based"},
            )
        ]

    @pytest.fixture
    def sample_table(self):
        """Sample extracted table for testing."""
        headers = ["Compound", "Molecular Weight", "Melting Point"]
        cells = [
            [
                TableCell("aspirin", 0, 0, "text"),
                TableCell("180.16", 0, 1, "number"),
                TableCell("135°C", 0, 2, "number_with_unit"),
            ],
            [
                TableCell("caffeine", 1, 0, "text"),
                TableCell("194.19", 1, 1, "number"),
                TableCell("235°C", 1, 2, "number_with_unit"),
            ],
        ]

        return ExtractedTable(
            headers=headers, cells=cells, table_type=TableType.COMPOUND_PROPERTIES
        )

    @pytest.fixture
    def sample_experimental_records(self):
        """Sample experimental records."""
        return [
            ExperimentalRecord(
                compound="benzene",
                conditions={
                    "temperature": {"value": 80, "unit": "°C"},
                    "time": {"value": 2, "unit": "h"},
                },
                properties={
                    "yield": {"value": 85, "unit": "%"},
                    "melting_point": {"value": 5.5, "unit": "°C"},
                },
                confidence=0.9,
            )
        ]

    def test_initialization(self, mapper):
        """Test mapper initialization."""
        assert mapper is not None
        assert mapper.config is not None
        assert mapper.kg_schema is not None
        assert len(mapper.entity_type_mapping) > 0
        assert len(mapper.relationship_type_mapping) > 0

    def test_map_entities_basic(self, mapper, sample_entities):
        """Test basic entity mapping."""
        result = mapper.map_entities(sample_entities)

        assert isinstance(result, MappingResult)
        assert len(result.mapped_entities) == 3  # 3 above confidence threshold
        assert len(result.unmapped_data) == 1  # 1 below threshold

        # Check mapped entity structure
        entity = result.mapped_entities[0]
        assert isinstance(entity, ChemicalEntity)
        assert entity.name == "aspirin"
        assert entity.entity_type == KGEntityType.SMALL_MOLECULE
        assert "entity_" in entity.entity_id

    def test_map_entities_confidence_filtering(self, mapper, sample_entities):
        """Test confidence-based filtering."""
        # Lower threshold to include all
        mapper.config.confidence_threshold = 0.1
        result = mapper.map_entities(sample_entities)
        assert len(result.mapped_entities) == 4

        # Higher threshold to exclude more
        mapper.config.confidence_threshold = 0.85
        result = mapper.map_entities(sample_entities)
        assert len(result.mapped_entities) == 1  # Only aspirin with 0.9 confidence

    def test_entity_type_mapping(self, mapper):
        """Test entity type mapping from NER to KG types."""
        from chemi_accel.nlp.entity_recognition import EntityType as NEREntityType

        # Test various mappings
        assert (
            mapper.entity_type_mapping[NEREntityType.CHEMICAL]
            == KGEntityType.SMALL_MOLECULE
        )
        assert (
            mapper.entity_type_mapping[NEREntityType.REACTION] == KGEntityType.REACTION
        )
        assert (
            mapper.entity_type_mapping[NEREntityType.MATERIAL] == KGEntityType.MATERIAL
        )

    def test_extract_identifiers(self, mapper):
        """Test chemical identifier extraction."""
        # Test molecular formula
        identifiers = mapper._extract_identifiers("C6H6")
        assert "molecular_formula" in identifiers

        # Test SMILES-like string
        identifiers = mapper._extract_identifiers("CCO")
        assert len(identifiers) > 0

        # Test CAS number
        identifiers = mapper._extract_identifiers("50-78-2")
        assert "cas_number" in identifiers

        # Test regular name
        identifiers = mapper._extract_identifiers("aspirin")
        assert len(identifiers) > 0

    def test_extract_entity_properties(self, mapper):
        """Test property extraction from entity context."""
        entity = EntityExtraction(
            text="aspirin",
            label="CHEMICAL",
            start_idx=0,
            end_idx=7,
            confidence=0.9,
            properties={
                "context": "aspirin has a molecular weight of 180.16 g/mol and melting point of 135°C"
            },
        )

        properties = mapper._extract_entity_properties(entity)

        # Should extract molecular weight and temperature
        assert len(properties) > 0

        if "molecular_weight" in properties:
            assert properties["molecular_weight"]["value"] == 180.16
            assert properties["molecular_weight"]["unit"] == "g/mol"

    def test_map_relationships(self, mapper, sample_relationships):
        """Test relationship mapping."""
        result = mapper.map_relationships(sample_relationships)

        assert isinstance(result, MappingResult)
        assert len(result.mapped_relationships) == 1

        relationship = result.mapped_relationships[0]
        assert isinstance(relationship, ChemicalRelationship)
        assert relationship.relationship_type == RelationshipType.INHIBITS
        assert relationship.confidence == 0.8

    def test_map_table_data(self, mapper, sample_table):
        """Test table data mapping."""
        result = mapper.map_table_data(sample_table)

        assert isinstance(result, MappingResult)
        assert len(result.mapped_entities) > 0
        assert len(result.mapped_relationships) > 0

        # Should create entities for compounds
        compound_entities = [
            e
            for e in result.mapped_entities
            if "aspirin" in e.name or "caffeine" in e.name
        ]
        assert len(compound_entities) >= 2

        # Should create property relationships
        property_rels = [
            r
            for r in result.mapped_relationships
            if r.relationship_type == RelationshipType.HAS_PROPERTY
        ]
        assert len(property_rels) > 0

    def test_find_compound_columns(self, mapper):
        """Test compound column detection."""
        headers = ["Entry", "Compound Name", "Molecular Weight", "Activity"]
        compound_cols = mapper._find_compound_columns(headers)

        assert 1 in compound_cols  # "Compound Name" column

    def test_find_property_columns(self, mapper):
        """Test property column detection."""
        headers = ["Compound", "Molecular Weight", "Melting Point", "IC50", "Notes"]
        property_cols = mapper._find_property_columns(headers)

        # Should find MW, MP, and IC50 columns
        assert len(property_cols) >= 3
        assert 1 in property_cols  # Molecular Weight
        assert 2 in property_cols  # Melting Point
        assert 3 in property_cols  # IC50

    def test_map_experimental_records(self, mapper, sample_experimental_records):
        """Test experimental record mapping."""
        result = mapper.map_experimental_records(sample_experimental_records)

        assert isinstance(result, MappingResult)
        assert len(result.mapped_entities) > 0
        assert len(result.mapped_relationships) > 0

        # Should create compound entity
        compound_entities = [e for e in result.mapped_entities if e.name == "benzene"]
        assert len(compound_entities) == 1

        # Should create condition entities
        condition_entities = [
            e
            for e in result.mapped_entities
            if e.entity_type == KGEntityType.EXPERIMENTAL_CONDITION
        ]
        assert len(condition_entities) > 0

    def test_create_complete_mapping(
        self,
        mapper,
        sample_entities,
        sample_relationships,
        sample_table,
        sample_experimental_records,
    ):
        """Test complete mapping of all data types."""
        result = mapper.create_complete_mapping(
            sample_entities,
            sample_relationships,
            [sample_table],
            sample_experimental_records,
        )

        assert isinstance(result, MappingResult)
        assert len(result.mapped_entities) > 0
        assert len(result.mapped_relationships) > 0

        # Should have comprehensive statistics
        stats = result.mapping_statistics
        assert "entities" in stats
        assert "relationships" in stats
        assert "tables" in stats
        assert "experimental_records" in stats
        assert "summary" in stats

        summary = stats["summary"]
        assert "total_mapped_entities" in summary
        assert "total_mapped_relationships" in summary

    def test_validation_enabled(self):
        """Test entity validation when enabled."""
        config = MappingConfig(validate_mapped_entities=True)
        mapper = ICKGSchemaMapper(config)

        entities = [EntityExtraction("test_compound", "CHEMICAL", 0, 13, 0.8, {})]

        result = mapper.map_entities(entities)

        # Should have validation results
        assert hasattr(result, "validation_errors")
        assert isinstance(result.validation_errors, list)

    def test_validation_disabled(self):
        """Test when validation is disabled."""
        config = MappingConfig(validate_mapped_entities=False)
        mapper = ICKGSchemaMapper(config)

        entities = [EntityExtraction("test_compound", "CHEMICAL", 0, 13, 0.8, {})]

        result = mapper.map_entities(entities)

        # Should have empty validation errors
        assert len(result.validation_errors) == 0

    def test_error_handling(self, mapper):
        """Test error handling in mapping process."""
        # Test with malformed entity
        bad_entity = Mock()
        bad_entity.confidence = 0.9
        bad_entity.text = None  # This should cause an error
        bad_entity.label = "CHEMICAL"

        result = mapper.map_entities([bad_entity])

        # Should handle error gracefully
        assert len(result.unmapped_data) > 0

        # Error should be recorded
        error_item = result.unmapped_data[0]
        assert error_item["type"] == "entity"
        assert "error" in error_item["reason"]

    def test_identifier_generation(self, mapper):
        """Test identifier generation for entities without existing identifiers."""
        # Test with entity that has no obvious identifiers
        identifiers = mapper._generate_identifiers(
            "some_compound_name", KGEntityType.SMALL_MOLECULE
        )

        assert "generated_id" in identifiers
        assert "original_text" in identifiers
        assert identifiers["original_text"] == "some_compound_name"

    def test_property_relationship_creation(self, mapper):
        """Test creation of property relationships."""
        compound_cell = TableCell("aspirin", 0, 0, "text")
        property_cell = TableCell("180.16", 0, 1, "number")

        relationship = mapper._create_property_relationship(
            compound_cell, property_cell, "Compound", "Molecular Weight"
        )

        assert isinstance(relationship, ChemicalRelationship)
        assert relationship.relationship_type == RelationshipType.HAS_PROPERTY
        assert relationship.properties["compound_name"] == "aspirin"
        assert relationship.properties["property_value"] == "180.16"

    def test_mapping_statistics_generation(self, mapper, sample_entities):
        """Test generation of mapping statistics."""
        result = mapper.map_entities(sample_entities)
        stats = result.mapping_statistics

        assert "total_extracted" in stats
        assert "successfully_mapped" in stats
        assert "failed_mapping" in stats
        assert "entity_type_distribution" in stats

        assert stats["total_extracted"] == len(sample_entities)
        assert (
            stats["successfully_mapped"] + stats["failed_mapping"]
            == stats["total_extracted"]
        )


class TestMappingConfig:
    """Test mapping configuration options."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MappingConfig()

        assert config.confidence_threshold == 0.5
        assert config.create_missing_entities == True
        assert config.infer_relationships == True
        assert config.validate_mapped_entities == True
        assert config.generate_identifiers == True
        assert config.preserve_original_data == True

    def test_custom_config(self):
        """Test custom configuration."""
        config = MappingConfig(
            confidence_threshold=0.8,
            create_missing_entities=False,
            validate_mapped_entities=False,
        )

        assert config.confidence_threshold == 0.8
        assert config.create_missing_entities == False
        assert config.validate_mapped_entities == False


class TestMappingResult:
    """Test mapping result structure."""

    def test_mapping_result_structure(self):
        """Test MappingResult structure and properties."""
        result = MappingResult(
            mapped_entities=[],
            mapped_relationships=[],
            mapping_statistics={"test": 1},
            validation_errors=["error1"],
            unmapped_data=[{"type": "entity"}],
        )

        assert hasattr(result, "mapped_entities")
        assert hasattr(result, "mapped_relationships")
        assert hasattr(result, "mapping_statistics")
        assert hasattr(result, "validation_errors")
        assert hasattr(result, "unmapped_data")

        assert len(result.validation_errors) == 1
        assert len(result.unmapped_data) == 1


class TestSchemaIntegration:
    """Integration tests for schema mapping."""

    def test_real_extraction_mapping(self):
        """Test mapping of realistic extraction results."""
        # Simulate realistic NLP extraction results
        entities = [
            EntityExtraction("benzene", "CHEMICAL", 0, 7, 0.9, {"method": "ml_model"}),
            EntityExtraction("C6H6", "CHEMICAL", 9, 13, 0.8, {"method": "pattern"}),
            EntityExtraction("80°C", "VALUE", 20, 24, 0.7, {"method": "pattern"}),
            EntityExtraction(
                "synthesis", "REACTION", 30, 39, 0.6, {"method": "ml_model"}
            ),
        ]

        relationships = [
            RelationshipExtraction(
                source_entity=entities[0],
                target_entity=entities[1],
                relation_type="SIMILAR_TO",
                confidence=0.8,
                context="benzene C6H6",
                properties={"method": "co_occurrence"},
            )
        ]

        mapper = ICKGSchemaMapper()
        result = mapper.create_complete_mapping(entities, relationships, [], [])

        # Should successfully map most entities
        assert len(result.mapped_entities) >= 3  # At least 3 above threshold
        assert len(result.mapped_relationships) >= 1

        # Check entity types are correctly mapped
        entity_types = [e.entity_type for e in result.mapped_entities]
        assert KGEntityType.SMALL_MOLECULE in entity_types
        assert KGEntityType.REACTION in entity_types

        # Check statistics
        stats = result.mapping_statistics["summary"]
        assert stats["total_mapped_entities"] > 0
        assert stats["total_mapped_relationships"] > 0

    def test_complex_table_mapping(self):
        """Test mapping of complex table data."""
        # Create a realistic chemical data table
        headers = ["Compound", "SMILES", "MW", "LogP", "IC50 (µM)", "Selectivity"]
        cells = [
            [
                TableCell("Compound_1", 0, 0, "text"),
                TableCell("CCO", 0, 1, "chemical"),
                TableCell("46.07", 0, 2, "number"),
                TableCell("-0.31", 0, 3, "number"),
                TableCell("12.5", 0, 4, "number"),
                TableCell("2.3", 0, 5, "number"),
            ],
            [
                TableCell("Compound_2", 1, 0, "text"),
                TableCell("c1ccccc1", 1, 1, "chemical"),
                TableCell("78.11", 1, 2, "number"),
                TableCell("2.13", 1, 3, "number"),
                TableCell("5.8", 1, 4, "number"),
                TableCell("4.1", 1, 5, "number"),
            ],
        ]

        table = ExtractedTable(
            headers=headers,
            cells=cells,
            table_type=TableType.BIOLOGICAL_ACTIVITY,
            title="Biological activity of synthesized compounds",
        )

        mapper = ICKGSchemaMapper()
        result = mapper.map_table_data(table)

        # Should create entities for compounds
        assert len(result.mapped_entities) >= 2

        # Should create property relationships
        assert len(result.mapped_relationships) >= 8  # 2 compounds × 4 properties

        # Check that SMILES identifiers are recognized
        smiles_entities = [
            e for e in result.mapped_entities if "smiles" in e.identifiers
        ]
        assert len(smiles_entities) >= 2

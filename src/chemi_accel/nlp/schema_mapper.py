"""
IC-KG Schema Mapping for NLP Extracted Entities

Maps extracted entities and relationships from NLP processing to the
Integrated Chemistry Knowledge Graph (IC-KG) schema format.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from ..knowledge_graph.ontology import (
    ChemicalEntity,
    ChemicalProperty,
    ChemicalRelationship,
)
from ..knowledge_graph.schema import EntityType as KGEntityType
from ..knowledge_graph.schema import (
    KnowledgeGraphSchema,
    PropertyType,
    RelationshipType,
)
from .backends.base import EntityExtraction, RelationshipExtraction
from .entity_recognition import EntityType as NEREntityType
from .table_extractor import ExperimentalRecord, ExtractedTable

logger = logging.getLogger(__name__)


@dataclass
class MappingConfig:
    """Configuration for schema mapping."""

    confidence_threshold: float = 0.5
    create_missing_entities: bool = True
    infer_relationships: bool = True
    validate_mapped_entities: bool = True
    generate_identifiers: bool = True
    preserve_original_data: bool = True


@dataclass
class MappingResult:
    """Result of schema mapping operation."""

    mapped_entities: List[ChemicalEntity]
    mapped_relationships: List[ChemicalRelationship]
    mapping_statistics: Dict[str, Any]
    validation_errors: List[str] = field(default_factory=list)
    unmapped_data: List[Dict[str, Any]] = field(default_factory=list)


class ICKGSchemaMapper:
    """
    Maps NLP-extracted entities and relationships to IC-KG schema.

    Provides comprehensive mapping from NLP extraction results to the
    standardized Integrated Chemistry Knowledge Graph schema, including
    entity type mapping, relationship inference, and data validation.
    """

    def __init__(self, config: Optional[MappingConfig] = None):
        self.config = config or MappingConfig()
        self.kg_schema = KnowledgeGraphSchema()

        # Mapping from NER entity types to KG entity types
        self.entity_type_mapping = {
            NEREntityType.CHEMICAL: KGEntityType.SMALL_MOLECULE,
            NEREntityType.COMPOUND: KGEntityType.SMALL_MOLECULE,
            NEREntityType.MOLECULE: KGEntityType.MOLECULE,
            NEREntityType.REACTION: KGEntityType.REACTION,
            NEREntityType.MATERIAL: KGEntityType.MATERIAL,
            NEREntityType.CATALYST: KGEntityType.CATALYST,
            NEREntityType.SOLVENT: KGEntityType.SOLVENT,
            NEREntityType.PRODUCT: KGEntityType.SMALL_MOLECULE,
            NEREntityType.REAGENT: KGEntityType.SMALL_MOLECULE,
        }

        # Mapping from NLP relationship types to KG relationship types
        self.relationship_type_mapping = {
            "BINDS_TO": RelationshipType.BINDS_TO,
            "INHIBITS": RelationshipType.INHIBITS,
            "ACTIVATES": RelationshipType.ACTIVATES,
            "SIMILAR_TO": RelationshipType.SIMILAR_TO,
            "REACTS_WITH": RelationshipType.REACTS_WITH,
            "FORMS": RelationshipType.FORMS,
            "CATALYZES": RelationshipType.CATALYZES,
        }

        # Property type mapping for extracted values
        self.property_mapping = {
            "molecular_weight": PropertyType.MOLECULAR_WEIGHT,
            "melting_point": PropertyType.MELTING_POINT,
            "boiling_point": PropertyType.BOILING_POINT,
            "solubility": PropertyType.SOLUBILITY,
            "density": PropertyType.DENSITY,
            "logp": PropertyType.LOGP,
            "tpsa": PropertyType.TPSA,
            "ic50": PropertyType.IC50,
            "ki": PropertyType.KI,
            "purity": PropertyType.PURITY,
            "yield": PropertyType.YIELD,
        }

        # Chemical identifier patterns
        self.identifier_patterns = {
            "smiles": r"^[A-Za-z0-9@+\-\[\]()=#$:.\/\\%]+$",
            "inchi": r"^InChI=.+",
            "cas_number": r"^\d{2,7}-\d{2}-\d$",
            "pubchem_cid": r"^\d+$",
            "chembl_id": r"^CHEMBL\d+$",
        }

        logger.info("Initialized ICKGSchemaMapper")

    def map_entities(self, extracted_entities: List[EntityExtraction]) -> MappingResult:
        """
        Map extracted entities to IC-KG schema.

        Args:
            extracted_entities: List of entities from NLP extraction

        Returns:
            MappingResult containing mapped entities and statistics
        """
        mapped_entities = []
        unmapped_data = []
        mapping_stats = {
            "total_extracted": len(extracted_entities),
            "successfully_mapped": 0,
            "failed_mapping": 0,
            "entity_type_distribution": {},
            "confidence_distribution": {},
        }

        for entity in extracted_entities:
            try:
                if entity.confidence >= self.config.confidence_threshold:
                    mapped_entity = self._map_single_entity(entity)
                    if mapped_entity:
                        mapped_entities.append(mapped_entity)
                        mapping_stats["successfully_mapped"] += 1

                        # Update statistics
                        entity_type = entity.label
                        mapping_stats["entity_type_distribution"][entity_type] = (
                            mapping_stats["entity_type_distribution"].get(
                                entity_type, 0
                            )
                            + 1
                        )
                    else:
                        unmapped_data.append(
                            {
                                "type": "entity",
                                "data": entity,
                                "reason": "mapping_failed",
                            }
                        )
                        mapping_stats["failed_mapping"] += 1
                else:
                    unmapped_data.append(
                        {"type": "entity", "data": entity, "reason": "low_confidence"}
                    )
                    mapping_stats["failed_mapping"] += 1

            except Exception as e:
                logger.error(f"Error mapping entity {entity.text}: {e}")
                unmapped_data.append(
                    {"type": "entity", "data": entity, "reason": f"error: {str(e)}"}
                )
                mapping_stats["failed_mapping"] += 1

        # Validate mapped entities if enabled
        validation_errors = []
        if self.config.validate_mapped_entities:
            validation_errors = self._validate_entities(mapped_entities)

        return MappingResult(
            mapped_entities=mapped_entities,
            mapped_relationships=[],
            mapping_statistics=mapping_stats,
            validation_errors=validation_errors,
            unmapped_data=unmapped_data,
        )

    def _map_single_entity(self, entity: EntityExtraction) -> Optional[ChemicalEntity]:
        """Map a single extracted entity to IC-KG schema."""
        # Determine KG entity type
        ner_type = self._get_ner_entity_type(entity.label)
        kg_entity_type = self.entity_type_mapping.get(ner_type, KGEntityType.MOLECULE)

        # Generate entity ID
        entity_id = f"entity_{uuid.uuid4().hex[:8]}"

        # Extract identifiers
        identifiers = self._extract_identifiers(entity.text)
        if self.config.generate_identifiers and not identifiers:
            identifiers = self._generate_identifiers(entity.text, kg_entity_type)

        # Extract properties
        properties = self._extract_entity_properties(entity)

        # Create ChemicalEntity
        mapped_entity = ChemicalEntity(
            entity_id=entity_id,
            entity_type=kg_entity_type,
            name=entity.text,
            description=f"Entity extracted from text with confidence {entity.confidence:.2f}",
            identifiers=identifiers,
            properties=properties,
            metadata={
                "extraction_method": entity.properties.get("method", "unknown"),
                "original_label": entity.label,
                "confidence": entity.confidence,
                "start_idx": entity.start_idx,
                "end_idx": entity.end_idx,
                "context": entity.properties.get("context", ""),
            },
        )

        return mapped_entity

    def _get_ner_entity_type(self, label: str) -> NEREntityType:
        """Convert label string to NEREntityType."""
        try:
            return NEREntityType(label.lower())
        except ValueError:
            # Try to match common variations
            label_lower = label.lower()
            if "chemical" in label_lower or "compound" in label_lower:
                return NEREntityType.CHEMICAL
            elif "reaction" in label_lower:
                return NEREntityType.REACTION
            elif "material" in label_lower:
                return NEREntityType.MATERIAL
            else:
                return NEREntityType.CHEMICAL  # Default fallback

    def _extract_identifiers(self, text: str) -> Dict[str, str]:
        """Extract chemical identifiers from entity text."""
        identifiers = {}

        for identifier_type, pattern in self.identifier_patterns.items():
            if re.match(pattern, text, re.IGNORECASE):
                identifiers[identifier_type] = text
                break

        # Special handling for common chemical name patterns
        if not identifiers:
            # Check if it looks like a molecular formula
            if re.match(r"^[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*$", text):
                identifiers["molecular_formula"] = text
            # Check if it looks like an IUPAC name
            elif any(
                suffix in text.lower()
                for suffix in ["yl", "ine", "ide", "ate", "oic acid"]
            ):
                identifiers["iupac_name"] = text
            else:
                identifiers["common_name"] = text

        return identifiers

    def _generate_identifiers(
        self, text: str, entity_type: KGEntityType
    ) -> Dict[str, str]:
        """Generate basic identifiers for an entity."""
        identifiers = {}

        # Generate a simple identifier based on entity type and text
        clean_text = re.sub(r"[^a-zA-Z0-9]", "_", text.lower())
        base_id = f"{entity_type.value}_{clean_text}"

        identifiers["generated_id"] = base_id
        identifiers["original_text"] = text

        return identifiers

    def _extract_entity_properties(
        self, entity: EntityExtraction
    ) -> Dict[str, Dict[str, Any]]:
        """Extract properties for an entity from its context and metadata."""
        properties = {}

        # Look for numerical values in the entity context
        context = entity.properties.get("context", "")
        if context:
            # Extract temperature values
            temp_matches = re.finditer(r"(\d+(?:\.\d+)?)\s*(°C|K|°F)", context)
            for match in temp_matches:
                value = float(match.group(1))
                unit = match.group(2)
                properties["temperature"] = {
                    "value": value,
                    "unit": unit,
                    "confidence": 0.7,
                    "source": "context_extraction",
                }

            # Extract molecular weight
            mw_matches = re.finditer(
                r"(?:molecular weight|MW|mol\.?\s*wt\.?)[:\s]+(\d+(?:\.\d+)?)",
                context,
                re.IGNORECASE,
            )
            for match in mw_matches:
                value = float(match.group(1))
                properties["molecular_weight"] = {
                    "value": value,
                    "unit": "g/mol",
                    "confidence": 0.8,
                    "source": "context_extraction",
                }

            # Extract melting point
            mp_matches = re.finditer(
                r"(?:melting point|mp|m\.?\s*p\.?)[:\s]+(\d+(?:\.\d+)?)\s*(°C|K)?",
                context,
                re.IGNORECASE,
            )
            for match in mp_matches:
                value = float(match.group(1))
                unit = match.group(2) or "°C"
                properties["melting_point"] = {
                    "value": value,
                    "unit": unit,
                    "confidence": 0.8,
                    "source": "context_extraction",
                }

        return properties

    def map_relationships(
        self, extracted_relationships: List[RelationshipExtraction]
    ) -> MappingResult:
        """
        Map extracted relationships to IC-KG schema.

        Args:
            extracted_relationships: List of relationships from NLP extraction

        Returns:
            MappingResult containing mapped relationships and statistics
        """
        mapped_relationships = []
        unmapped_data = []
        mapping_stats = {
            "total_extracted": len(extracted_relationships),
            "successfully_mapped": 0,
            "failed_mapping": 0,
            "relationship_type_distribution": {},
        }

        for relationship in extracted_relationships:
            try:
                if relationship.confidence >= self.config.confidence_threshold:
                    mapped_relationship = self._map_single_relationship(relationship)
                    if mapped_relationship:
                        mapped_relationships.append(mapped_relationship)
                        mapping_stats["successfully_mapped"] += 1

                        # Update statistics
                        rel_type = relationship.relation_type
                        mapping_stats["relationship_type_distribution"][rel_type] = (
                            mapping_stats["relationship_type_distribution"].get(
                                rel_type, 0
                            )
                            + 1
                        )
                    else:
                        unmapped_data.append(
                            {
                                "type": "relationship",
                                "data": relationship,
                                "reason": "mapping_failed",
                            }
                        )
                        mapping_stats["failed_mapping"] += 1
                else:
                    unmapped_data.append(
                        {
                            "type": "relationship",
                            "data": relationship,
                            "reason": "low_confidence",
                        }
                    )
                    mapping_stats["failed_mapping"] += 1

            except Exception as e:
                logger.error(f"Error mapping relationship: {e}")
                unmapped_data.append(
                    {
                        "type": "relationship",
                        "data": relationship,
                        "reason": f"error: {str(e)}",
                    }
                )
                mapping_stats["failed_mapping"] += 1

        return MappingResult(
            mapped_entities=[],
            mapped_relationships=mapped_relationships,
            mapping_statistics=mapping_stats,
            validation_errors=[],
            unmapped_data=unmapped_data,
        )

    def _map_single_relationship(
        self, relationship: RelationshipExtraction
    ) -> Optional[ChemicalRelationship]:
        """Map a single extracted relationship to IC-KG schema."""
        # Map relationship type
        kg_relationship_type = self.relationship_type_mapping.get(
            relationship.relation_type, RelationshipType.INTERACTS_WITH
        )

        # Generate relationship ID
        relationship_id = f"rel_{uuid.uuid4().hex[:8]}"

        # Map source and target entities (simplified - in practice, need to resolve entity IDs)
        source_entity_id = (
            f"entity_{uuid.uuid4().hex[:8]}"  # Would resolve from mapped entities
        )
        target_entity_id = (
            f"entity_{uuid.uuid4().hex[:8]}"  # Would resolve from mapped entities
        )

        # Create ChemicalRelationship
        mapped_relationship = ChemicalRelationship(
            relationship_id=relationship_id,
            relationship_type=kg_relationship_type,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            properties={
                "context": relationship.context,
                "extraction_confidence": relationship.confidence,
                "extraction_method": relationship.properties.get("method", "unknown"),
            },
            confidence=relationship.confidence,
            source="nlp_extraction",
            evidence=[relationship.context],
            metadata={
                "original_relation_type": relationship.relation_type,
                "source_entity_text": relationship.source_entity.text,
                "target_entity_text": relationship.target_entity.text,
            },
        )

        return mapped_relationship

    def map_table_data(self, table: ExtractedTable) -> MappingResult:
        """
        Map structured table data to IC-KG entities and relationships.

        Args:
            table: Extracted table from NLP processing

        Returns:
            MappingResult containing mapped entities from table data
        """
        mapped_entities = []
        mapped_relationships = []
        unmapped_data = []

        # Find compound/chemical columns
        compound_columns = self._find_compound_columns(table.headers)
        property_columns = self._find_property_columns(table.headers)

        for row_idx, row in enumerate(table.cells):
            try:
                # Extract compound entities from compound columns
                for col_idx in compound_columns:
                    if col_idx < len(row):
                        cell = row[col_idx]
                        if cell.value and len(cell.value.strip()) > 2:
                            entity = self._create_entity_from_cell(
                                cell, table.headers[col_idx], row_idx
                            )
                            if entity:
                                mapped_entities.append(entity)

                # Extract property relationships
                if compound_columns and property_columns:
                    compound_col = compound_columns[0]  # Use first compound column
                    if compound_col < len(row):
                        compound_cell = row[compound_col]

                        for prop_col in property_columns:
                            if prop_col < len(row):
                                prop_cell = row[prop_col]
                                relationship = self._create_property_relationship(
                                    compound_cell,
                                    prop_cell,
                                    table.headers[compound_col],
                                    table.headers[prop_col],
                                )
                                if relationship:
                                    mapped_relationships.append(relationship)

            except Exception as e:
                logger.error(f"Error mapping table row {row_idx}: {e}")
                unmapped_data.append(
                    {"type": "table_row", "data": row, "reason": f"error: {str(e)}"}
                )

        mapping_stats = {
            "total_rows": len(table.cells),
            "mapped_entities": len(mapped_entities),
            "mapped_relationships": len(mapped_relationships),
            "table_type": table.table_type.value,
        }

        return MappingResult(
            mapped_entities=mapped_entities,
            mapped_relationships=mapped_relationships,
            mapping_statistics=mapping_stats,
            validation_errors=[],
            unmapped_data=unmapped_data,
        )

    def _find_compound_columns(self, headers: List[str]) -> List[int]:
        """Find columns that contain compound/chemical names."""
        compound_keywords = [
            "compound",
            "chemical",
            "molecule",
            "drug",
            "name",
            "structure",
        ]
        compound_columns = []

        for i, header in enumerate(headers):
            header_lower = header.lower()
            if any(keyword in header_lower for keyword in compound_keywords):
                compound_columns.append(i)

        return compound_columns

    def _find_property_columns(self, headers: List[str]) -> List[int]:
        """Find columns that contain chemical properties."""
        property_keywords = list(self.property_mapping.keys()) + [
            "activity",
            "value",
            "result",
            "measurement",
            "data",
        ]
        property_columns = []

        for i, header in enumerate(headers):
            header_lower = header.lower()
            if any(keyword in header_lower for keyword in property_keywords):
                property_columns.append(i)

        return property_columns

    def _create_entity_from_cell(
        self, cell, header: str, row_idx: int
    ) -> Optional[ChemicalEntity]:
        """Create a ChemicalEntity from a table cell."""
        entity_id = f"table_entity_{row_idx}_{cell.column}"

        # Determine entity type based on header
        entity_type = KGEntityType.SMALL_MOLECULE  # Default
        if "reaction" in header.lower():
            entity_type = KGEntityType.REACTION
        elif "material" in header.lower():
            entity_type = KGEntityType.MATERIAL

        # Extract identifiers
        identifiers = self._extract_identifiers(cell.value)
        if not identifiers:
            identifiers = {"table_value": cell.value}

        entity = ChemicalEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=cell.value,
            description=f"Entity from table cell ({header})",
            identifiers=identifiers,
            metadata={
                "source": "table_extraction",
                "table_header": header,
                "row_index": row_idx,
                "column_index": cell.column,
                "data_type": cell.data_type,
            },
        )

        return entity

    def _create_property_relationship(
        self, compound_cell, property_cell, compound_header: str, property_header: str
    ) -> Optional[ChemicalRelationship]:
        """Create a property relationship between compound and property value."""
        relationship_id = f"prop_rel_{uuid.uuid4().hex[:8]}"

        # Simplified - would need proper entity ID resolution
        source_entity_id = f"table_entity_{compound_cell.row}_{compound_cell.column}"
        target_entity_id = f"table_entity_{property_cell.row}_{property_cell.column}"

        relationship = ChemicalRelationship(
            relationship_id=relationship_id,
            relationship_type=RelationshipType.HAS_PROPERTY,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            properties={
                "property_name": property_header,
                "property_value": property_cell.value,
                "compound_name": compound_cell.value,
            },
            confidence=0.9,  # High confidence for table data
            source="table_extraction",
            metadata={
                "compound_header": compound_header,
                "property_header": property_header,
            },
        )

        return relationship

    def map_experimental_records(
        self, records: List[ExperimentalRecord]
    ) -> MappingResult:
        """
        Map experimental records to IC-KG entities and relationships.

        Args:
            records: List of experimental records from table extraction

        Returns:
            MappingResult containing mapped experimental data
        """
        mapped_entities = []
        mapped_relationships = []

        for i, record in enumerate(records):
            try:
                # Create compound entity
                if record.compound:
                    compound_entity = self._create_compound_entity_from_record(
                        record, i
                    )
                    mapped_entities.append(compound_entity)

                # Create condition entities and relationships
                for condition_name, condition_data in record.conditions.items():
                    condition_entity = self._create_condition_entity(
                        condition_data, condition_name, i
                    )
                    if condition_entity:
                        mapped_entities.append(condition_entity)

                        # Create relationship between compound and condition
                        if record.compound:
                            condition_rel = self._create_condition_relationship(
                                compound_entity.entity_id,
                                condition_entity.entity_id,
                                condition_name,
                            )
                            mapped_relationships.append(condition_rel)

                # Create property relationships
                for property_name, property_data in record.properties.items():
                    if record.compound:
                        property_rel = self._create_property_relationship_from_record(
                            compound_entity.entity_id, property_name, property_data
                        )
                        if property_rel:
                            mapped_relationships.append(property_rel)

            except Exception as e:
                logger.error(f"Error mapping experimental record {i}: {e}")

        mapping_stats = {
            "total_records": len(records),
            "mapped_entities": len(mapped_entities),
            "mapped_relationships": len(mapped_relationships),
        }

        return MappingResult(
            mapped_entities=mapped_entities,
            mapped_relationships=mapped_relationships,
            mapping_statistics=mapping_stats,
            validation_errors=[],
            unmapped_data=[],
        )

    def _create_compound_entity_from_record(
        self, record: ExperimentalRecord, record_idx: int
    ) -> ChemicalEntity:
        """Create a compound entity from experimental record."""
        entity_id = f"exp_compound_{record_idx}"

        identifiers = self._extract_identifiers(record.compound)
        if not identifiers:
            identifiers = {"experimental_name": record.compound}

        entity = ChemicalEntity(
            entity_id=entity_id,
            entity_type=KGEntityType.SMALL_MOLECULE,
            name=record.compound,
            description="Compound from experimental data",
            identifiers=identifiers,
            metadata={
                "source": "experimental_record",
                "record_index": record_idx,
                "confidence": record.confidence,
            },
        )

        return entity

    def _create_condition_entity(
        self, condition_data: Dict[str, Any], condition_name: str, record_idx: int
    ) -> Optional[ChemicalEntity]:
        """Create an entity representing experimental conditions."""
        if isinstance(condition_data, dict) and "value" in condition_data:
            entity_id = f"exp_condition_{record_idx}_{condition_name}"

            entity = ChemicalEntity(
                entity_id=entity_id,
                entity_type=KGEntityType.EXPERIMENTAL_CONDITION,
                name=f"{condition_name}: {condition_data.get('value', '')} {condition_data.get('unit', '')}",
                description=f"Experimental condition: {condition_name}",
                identifiers={"condition_type": condition_name},
                properties={
                    condition_name: {
                        "value": condition_data.get("value"),
                        "unit": condition_data.get("unit"),
                        "raw": condition_data.get("raw"),
                    }
                },
                metadata={
                    "source": "experimental_condition",
                    "record_index": record_idx,
                },
            )

            return entity

        return None

    def _create_condition_relationship(
        self, compound_id: str, condition_id: str, condition_name: str
    ) -> ChemicalRelationship:
        """Create relationship between compound and experimental condition."""
        relationship_id = f"cond_rel_{uuid.uuid4().hex[:8]}"

        relationship = ChemicalRelationship(
            relationship_id=relationship_id,
            relationship_type=RelationshipType.STUDIED_UNDER,
            source_entity_id=compound_id,
            target_entity_id=condition_id,
            properties={"condition_type": condition_name},
            confidence=0.9,
            source="experimental_record",
            metadata={"condition_name": condition_name},
        )

        return relationship

    def _create_property_relationship_from_record(
        self, compound_id: str, property_name: str, property_data: Dict[str, Any]
    ) -> Optional[ChemicalRelationship]:
        """Create property relationship from experimental record."""
        if isinstance(property_data, dict) and "value" in property_data:
            relationship_id = f"prop_rel_{uuid.uuid4().hex[:8]}"

            relationship = ChemicalRelationship(
                relationship_id=relationship_id,
                relationship_type=RelationshipType.HAS_PROPERTY,
                source_entity_id=compound_id,
                target_entity_id=f"property_{property_name}_{uuid.uuid4().hex[:8]}",
                properties={
                    "property_name": property_name,
                    "property_value": property_data.get("value"),
                    "property_unit": property_data.get("unit"),
                    "raw_value": property_data.get("raw"),
                },
                confidence=0.8,
                source="experimental_record",
                metadata={"property_type": property_name},
            )

            return relationship

        return None

    def _validate_entities(self, entities: List[ChemicalEntity]) -> List[str]:
        """Validate mapped entities against KG schema."""
        validation_errors = []

        for entity in entities:
            try:
                errors = self.kg_schema.validate_entity(entity)
                if errors:
                    validation_errors.extend(
                        [f"Entity {entity.entity_id}: {error}" for error in errors]
                    )
            except Exception as e:
                validation_errors.append(
                    f"Validation error for entity {entity.entity_id}: {str(e)}"
                )

        return validation_errors

    def create_complete_mapping(
        self,
        entities: List[EntityExtraction],
        relationships: List[RelationshipExtraction],
        tables: List[ExtractedTable],
        experimental_records: List[ExperimentalRecord],
    ) -> MappingResult:
        """
        Create a complete mapping of all NLP extraction results to IC-KG schema.

        Args:
            entities: Extracted entities
            relationships: Extracted relationships
            tables: Extracted tables
            experimental_records: Experimental records

        Returns:
            Complete mapping result
        """
        all_entities = []
        all_relationships = []
        all_unmapped = []
        combined_stats = {}
        all_validation_errors = []

        # Map entities
        if entities:
            entity_result = self.map_entities(entities)
            all_entities.extend(entity_result.mapped_entities)
            all_unmapped.extend(entity_result.unmapped_data)
            all_validation_errors.extend(entity_result.validation_errors)
            combined_stats["entities"] = entity_result.mapping_statistics

        # Map relationships
        if relationships:
            relationship_result = self.map_relationships(relationships)
            all_relationships.extend(relationship_result.mapped_relationships)
            all_unmapped.extend(relationship_result.unmapped_data)
            combined_stats["relationships"] = relationship_result.mapping_statistics

        # Map tables
        if tables:
            table_stats = []
            for table in tables:
                table_result = self.map_table_data(table)
                all_entities.extend(table_result.mapped_entities)
                all_relationships.extend(table_result.mapped_relationships)
                all_unmapped.extend(table_result.unmapped_data)
                table_stats.append(table_result.mapping_statistics)
            combined_stats["tables"] = table_stats

        # Map experimental records
        if experimental_records:
            exp_result = self.map_experimental_records(experimental_records)
            all_entities.extend(exp_result.mapped_entities)
            all_relationships.extend(exp_result.mapped_relationships)
            all_unmapped.extend(exp_result.unmapped_data)
            combined_stats["experimental_records"] = exp_result.mapping_statistics

        # Overall statistics
        combined_stats["summary"] = {
            "total_mapped_entities": len(all_entities),
            "total_mapped_relationships": len(all_relationships),
            "total_unmapped_items": len(all_unmapped),
            "total_validation_errors": len(all_validation_errors),
        }

        return MappingResult(
            mapped_entities=all_entities,
            mapped_relationships=all_relationships,
            mapping_statistics=combined_stats,
            validation_errors=all_validation_errors,
            unmapped_data=all_unmapped,
        )

"""
Knowledge Graph Schema Management and Validation

This module provides comprehensive schema management for the chemical knowledge graph,
including formal schema definitions, validation, and integrity checking.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

import jsonschema

from .ontology import (
    ChemicalEntity,
    ChemicalProperty,
    ChemicalRelationship,
    EntityType,
    PropertyType,
    RelationshipType,
)


class DataType(Enum):
    """Data types for property values."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATE = "date"
    DATETIME = "datetime"


@dataclass
class PropertySchema:
    """
    Schema definition for a property type.

    Defines the structure, constraints, and validation rules
    for a specific property type.
    """

    property_type: PropertyType
    data_type: DataType
    required: bool = False
    default_value: Optional[Any] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    unit_required: bool = False
    allowed_units: Optional[Set[str]] = None
    description: Optional[str] = None

    def validate_value(self, value: Any, unit: Optional[str] = None) -> List[str]:
        """
        Validate a property value against this schema.

        Returns:
            List of validation errors
        """
        errors = []

        # Check data type
        if not self._validate_data_type(value):
            errors.append(
                f"Value {value} does not match expected type {self.data_type.value}"
            )
            return errors  # Don't proceed with other validations if type is wrong

        # Check value constraints
        if self.min_value is not None and value < self.min_value:
            errors.append(f"Value {value} is below minimum {self.min_value}")

        if self.max_value is not None and value > self.max_value:
            errors.append(f"Value {value} is above maximum {self.max_value}")

        if self.allowed_values is not None and value not in self.allowed_values:
            errors.append(f"Value {value} not in allowed values {self.allowed_values}")

        if self.pattern and isinstance(value, str):
            import re

            if not re.match(self.pattern, value):
                errors.append(f"Value {value} does not match pattern {self.pattern}")

        # Check unit constraints
        if self.unit_required and not unit:
            errors.append(f"Unit is required for property {self.property_type.value}")

        if unit and self.allowed_units and unit not in self.allowed_units:
            errors.append(f"Unit {unit} not in allowed units {self.allowed_units}")

        return errors

    def _validate_data_type(self, value: Any) -> bool:
        """Check if value matches the expected data type."""
        type_validators = {
            DataType.STRING: lambda v: isinstance(v, str),
            DataType.INTEGER: lambda v: isinstance(v, int),
            DataType.FLOAT: lambda v: isinstance(v, (int, float)),
            DataType.BOOLEAN: lambda v: isinstance(v, bool),
            DataType.ARRAY: lambda v: isinstance(v, (list, tuple)),
            DataType.OBJECT: lambda v: isinstance(v, dict),
            DataType.DATE: lambda v: isinstance(v, str),  # Simplified validation
            DataType.DATETIME: lambda v: isinstance(v, str),  # Simplified validation
        }

        validator = type_validators.get(self.data_type)
        return validator(value) if validator else True

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {"type": self.data_type.value}

        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.allowed_values is not None:
            schema["enum"] = self.allowed_values
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.description:
            schema["description"] = self.description

        return schema


@dataclass
class EntitySchema:
    """
    Schema definition for an entity type.

    Defines the structure and constraints for entities of a specific type.
    """

    entity_type: EntityType
    required_properties: Set[PropertyType] = field(default_factory=set)
    optional_properties: Set[PropertyType] = field(default_factory=set)
    forbidden_properties: Set[PropertyType] = field(default_factory=set)
    required_identifiers: Set[str] = field(default_factory=set)
    property_schemas: Dict[PropertyType, PropertySchema] = field(default_factory=dict)
    description: Optional[str] = None

    def validate_entity(self, entity: ChemicalEntity) -> List[str]:
        """
        Validate an entity against this schema.

        Returns:
            List of validation errors
        """
        errors = []

        # Check entity type
        if entity.entity_type != self.entity_type:
            errors.append(
                f"Entity type {entity.entity_type.value} does not match "
                f"schema type {self.entity_type.value}"
            )

        # Check required properties
        for required_prop in self.required_properties:
            if not entity.has_property(required_prop):
                errors.append(f"Required property {required_prop.value} is missing")

        # Check forbidden properties
        for prop_type_str in entity.properties:
            try:
                prop_type = PropertyType(prop_type_str)
                if prop_type in self.forbidden_properties:
                    errors.append(f"Forbidden property {prop_type.value} is present")
            except ValueError:
                continue  # Unknown property type, handled elsewhere

        # Check required identifiers
        for required_id in self.required_identifiers:
            if required_id not in entity.identifiers:
                errors.append(f"Required identifier {required_id} is missing")

        # Validate individual properties
        for prop_type_str, prop_data in entity.properties.items():
            try:
                prop_type = PropertyType(prop_type_str)
                schema = self.property_schemas.get(prop_type)
                if schema:
                    prop_value = prop_data.get("value")
                    prop_unit = prop_data.get("unit")
                    prop_errors = schema.validate_value(prop_value, prop_unit)
                    errors.extend(
                        [
                            f"Property {prop_type.value}: {error}"
                            for error in prop_errors
                        ]
                    )
            except ValueError:
                continue  # Unknown property type

        return errors

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        properties = {}
        required = ["entity_id", "entity_type", "name"]

        # Add property schemas
        for prop_type, prop_schema in self.property_schemas.items():
            properties[prop_type.value] = prop_schema.to_json_schema()
            if prop_type in self.required_properties:
                required.append(prop_type.value)

        schema = {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "entity_type": {"type": "string", "const": self.entity_type.value},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "identifiers": {"type": "object"},
                "properties": {"type": "object", "properties": properties},
                "metadata": {"type": "object"},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"},
            },
            "required": required,
            "additionalProperties": False,
        }

        if self.description:
            schema["description"] = self.description

        return schema


@dataclass
class RelationshipSchema:
    """
    Schema definition for a relationship type.

    Defines the structure and constraints for relationships of a specific type.
    """

    relationship_type: RelationshipType
    source_entity_types: Set[EntityType] = field(default_factory=set)
    target_entity_types: Set[EntityType] = field(default_factory=set)
    required_properties: Dict[str, PropertyType] = field(default_factory=dict)
    confidence_required: bool = False
    evidence_required: bool = False
    is_symmetric: bool = False
    is_transitive: bool = False
    description: Optional[str] = None

    def validate_relationship(
        self,
        relationship: ChemicalRelationship,
        source_entity: ChemicalEntity,
        target_entity: ChemicalEntity,
    ) -> List[str]:
        """
        Validate a relationship against this schema.

        Returns:
            List of validation errors
        """
        errors = []

        # Check relationship type
        if relationship.relationship_type != self.relationship_type:
            errors.append(
                f"Relationship type {relationship.relationship_type.value} does not match "
                f"schema type {self.relationship_type.value}"
            )

        # Check source entity type
        if (
            self.source_entity_types
            and source_entity.entity_type not in self.source_entity_types
        ):
            errors.append(
                f"Source entity type {source_entity.entity_type.value} not allowed "
                f"for relationship {self.relationship_type.value}"
            )

        # Check target entity type
        if (
            self.target_entity_types
            and target_entity.entity_type not in self.target_entity_types
        ):
            errors.append(
                f"Target entity type {target_entity.entity_type.value} not allowed "
                f"for relationship {self.relationship_type.value}"
            )

        # Check required properties
        for prop_name, prop_type in self.required_properties.items():
            if prop_name not in relationship.properties:
                errors.append(f"Required property {prop_name} is missing")

        # Check confidence requirement
        if self.confidence_required and relationship.confidence is None:
            errors.append("Confidence score is required")

        # Check evidence requirement
        if self.evidence_required and not relationship.evidence:
            errors.append("Evidence is required")

        return errors

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        required = [
            "relationship_id",
            "relationship_type",
            "source_entity_id",
            "target_entity_id",
        ]

        if self.confidence_required:
            required.append("confidence")
        if self.evidence_required:
            required.append("evidence")

        schema = {
            "type": "object",
            "properties": {
                "relationship_id": {"type": "string"},
                "relationship_type": {
                    "type": "string",
                    "const": self.relationship_type.value,
                },
                "source_entity_id": {"type": "string"},
                "target_entity_id": {"type": "string"},
                "properties": {"type": "object"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "source": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "is_directional": {"type": "boolean"},
                "created_at": {"type": "string", "format": "date-time"},
            },
            "required": required,
            "additionalProperties": False,
        }

        if self.description:
            schema["description"] = self.description

        return schema


class KnowledgeGraphSchema:
    """
    Comprehensive schema manager for the chemical knowledge graph.

    Manages entity schemas, relationship schemas, and provides
    comprehensive validation capabilities.
    """

    def __init__(self):
        self.entity_schemas: Dict[EntityType, EntitySchema] = {}
        self.relationship_schemas: Dict[RelationshipType, RelationshipSchema] = {}
        self.property_schemas: Dict[PropertyType, PropertySchema] = {}

        # Initialize with default schemas
        self._initialize_default_schemas()

    def _initialize_default_schemas(self) -> None:
        """Initialize with default schema definitions."""

        # Define common property schemas
        self._define_property_schemas()

        # Define entity schemas
        self._define_entity_schemas()

        # Define relationship schemas
        self._define_relationship_schemas()

    def _define_property_schemas(self) -> None:
        """Define schemas for common property types."""

        # Molecular descriptors
        self.property_schemas[PropertyType.MOLECULAR_WEIGHT] = PropertySchema(
            property_type=PropertyType.MOLECULAR_WEIGHT,
            data_type=DataType.FLOAT,
            min_value=0.0,
            max_value=10000.0,
            unit_required=True,
            allowed_units={"g/mol", "Da", "kDa"},
            description="Molecular weight of the compound",
        )

        self.property_schemas[PropertyType.LOGP] = PropertySchema(
            property_type=PropertyType.LOGP,
            data_type=DataType.FLOAT,
            min_value=-10.0,
            max_value=10.0,
            description="Octanol-water partition coefficient",
        )

        self.property_schemas[PropertyType.TPSA] = PropertySchema(
            property_type=PropertyType.TPSA,
            data_type=DataType.FLOAT,
            min_value=0.0,
            max_value=1000.0,
            unit_required=True,
            allowed_units={"Ų", "Å²"},
            description="Topological polar surface area",
        )

        # Biological activities
        self.property_schemas[PropertyType.IC50] = PropertySchema(
            property_type=PropertyType.IC50,
            data_type=DataType.FLOAT,
            min_value=0.0,
            unit_required=True,
            allowed_units={"nM", "µM", "mM", "M"},
            description="Half maximal inhibitory concentration",
        )

        self.property_schemas[PropertyType.KI] = PropertySchema(
            property_type=PropertyType.KI,
            data_type=DataType.FLOAT,
            min_value=0.0,
            unit_required=True,
            allowed_units={"nM", "µM", "mM", "M"},
            description="Inhibition constant",
        )

        # Physical properties
        self.property_schemas[PropertyType.MELTING_POINT] = PropertySchema(
            property_type=PropertyType.MELTING_POINT,
            data_type=DataType.FLOAT,
            min_value=-273.15,
            unit_required=True,
            allowed_units={"°C", "K", "°F"},
            description="Melting point temperature",
        )

        self.property_schemas[PropertyType.SOLUBILITY] = PropertySchema(
            property_type=PropertyType.SOLUBILITY,
            data_type=DataType.FLOAT,
            min_value=0.0,
            unit_required=True,
            allowed_units={"mg/mL", "g/L", "M", "mM", "µM"},
            description="Aqueous solubility",
        )

    def _define_entity_schemas(self) -> None:
        """Define schemas for entity types."""

        # Small molecule schema
        self.entity_schemas[EntityType.SMALL_MOLECULE] = EntitySchema(
            entity_type=EntityType.SMALL_MOLECULE,
            required_properties={PropertyType.MOLECULAR_WEIGHT},
            optional_properties={
                PropertyType.LOGP,
                PropertyType.TPSA,
                PropertyType.HBD,
                PropertyType.HBA,
                PropertyType.ROTATABLE_BONDS,
                PropertyType.MELTING_POINT,
                PropertyType.SOLUBILITY,
            },
            required_identifiers={"smiles"},
            property_schemas={
                prop_type: self.property_schemas[prop_type]
                for prop_type in [
                    PropertyType.MOLECULAR_WEIGHT,
                    PropertyType.LOGP,
                    PropertyType.TPSA,
                    PropertyType.MELTING_POINT,
                    PropertyType.SOLUBILITY,
                ]
                if prop_type in self.property_schemas
            },
            description="Small organic molecule",
        )

        # Target protein schema
        self.entity_schemas[EntityType.PROTEIN_TARGET] = EntitySchema(
            entity_type=EntityType.PROTEIN_TARGET,
            required_properties={PropertyType.MOLECULAR_WEIGHT},
            required_identifiers={"uniprot_id"},
            property_schemas={
                PropertyType.MOLECULAR_WEIGHT: self.property_schemas[
                    PropertyType.MOLECULAR_WEIGHT
                ]
            },
            description="Protein target for drug discovery",
        )

        # Assay schema
        self.entity_schemas[EntityType.BIOASSAY] = EntitySchema(
            entity_type=EntityType.BIOASSAY,
            required_identifiers={"assay_id"},
            description="Biological assay for measuring molecular activity",
        )

    def _define_relationship_schemas(self) -> None:
        """Define schemas for relationship types."""

        # Binding relationship
        self.relationship_schemas[RelationshipType.BINDS_TO] = RelationshipSchema(
            relationship_type=RelationshipType.BINDS_TO,
            source_entity_types={EntityType.SMALL_MOLECULE, EntityType.MOLECULE},
            target_entity_types={
                EntityType.PROTEIN_TARGET,
                EntityType.ENZYME,
                EntityType.RECEPTOR,
            },
            confidence_required=True,
            evidence_required=True,
            description="Molecular binding interaction",
        )

        # Inhibition relationship
        self.relationship_schemas[RelationshipType.INHIBITS] = RelationshipSchema(
            relationship_type=RelationshipType.INHIBITS,
            source_entity_types={EntityType.SMALL_MOLECULE, EntityType.MOLECULE},
            target_entity_types={
                EntityType.ENZYME,
                EntityType.RECEPTOR,
                EntityType.ION_CHANNEL,
            },
            confidence_required=True,
            evidence_required=True,
            description="Inhibitory interaction",
        )

        # Similarity relationship
        self.relationship_schemas[RelationshipType.SIMILAR_TO] = RelationshipSchema(
            relationship_type=RelationshipType.SIMILAR_TO,
            source_entity_types={EntityType.SMALL_MOLECULE, EntityType.MOLECULE},
            target_entity_types={EntityType.SMALL_MOLECULE, EntityType.MOLECULE},
            is_symmetric=True,
            confidence_required=True,
            description="Structural or chemical similarity",
        )

    def add_entity_schema(self, schema: EntitySchema) -> None:
        """Add an entity schema."""
        self.entity_schemas[schema.entity_type] = schema

    def add_relationship_schema(self, schema: RelationshipSchema) -> None:
        """Add a relationship schema."""
        self.relationship_schemas[schema.relationship_type] = schema

    def add_property_schema(self, schema: PropertySchema) -> None:
        """Add a property schema."""
        self.property_schemas[schema.property_type] = schema

    def validate_entity(self, entity: ChemicalEntity) -> List[str]:
        """Validate an entity against its schema."""
        schema = self.entity_schemas.get(entity.entity_type)
        if not schema:
            return [f"No schema defined for entity type {entity.entity_type.value}"]

        return schema.validate_entity(entity)

    def validate_relationship(
        self,
        relationship: ChemicalRelationship,
        source_entity: ChemicalEntity,
        target_entity: ChemicalEntity,
    ) -> List[str]:
        """Validate a relationship against its schema."""
        schema = self.relationship_schemas.get(relationship.relationship_type)
        if not schema:
            return [
                f"No schema defined for relationship type {relationship.relationship_type.value}"
            ]

        return schema.validate_relationship(relationship, source_entity, target_entity)

    def export_json_schema(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export the complete schema as JSON Schema."""
        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            entity_schema.to_json_schema()
                            for entity_schema in self.entity_schemas.values()
                        ]
                    },
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            rel_schema.to_json_schema()
                            for rel_schema in self.relationship_schemas.values()
                        ]
                    },
                },
            },
            "required": ["entities", "relationships"],
            "additionalProperties": False,
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(schema, f, indent=2)

        return schema

    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the current schema definitions."""
        return {
            "entity_types": len(self.entity_schemas),
            "relationship_types": len(self.relationship_schemas),
            "property_types": len(self.property_schemas),
            "entity_schemas": {
                entity_type.value: {
                    "required_properties": len(schema.required_properties),
                    "optional_properties": len(schema.optional_properties),
                    "required_identifiers": len(schema.required_identifiers),
                }
                for entity_type, schema in self.entity_schemas.items()
            },
            "relationship_schemas": {
                rel_type.value: {
                    "source_types": len(schema.source_entity_types),
                    "target_types": len(schema.target_entity_types),
                    "confidence_required": schema.confidence_required,
                    "evidence_required": schema.evidence_required,
                }
                for rel_type, schema in self.relationship_schemas.items()
            },
        }


class SchemaValidator:
    """
    Comprehensive validator for knowledge graph components.

    Provides validation services for entities, relationships, and
    complete knowledge graphs against defined schemas.
    """

    def __init__(self, schema: KnowledgeGraphSchema):
        self.schema = schema

    def validate_entity(self, entity: ChemicalEntity) -> Tuple[bool, List[str]]:
        """
        Validate an entity.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = self.schema.validate_entity(entity)
        return len(errors) == 0, errors

    def validate_relationship(
        self,
        relationship: ChemicalRelationship,
        source_entity: ChemicalEntity,
        target_entity: ChemicalEntity,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a relationship.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = self.schema.validate_relationship(
            relationship, source_entity, target_entity
        )
        return len(errors) == 0, errors

    def validate_knowledge_graph(
        self, entities: List[ChemicalEntity], relationships: List[ChemicalRelationship]
    ) -> Dict[str, Any]:
        """
        Validate a complete knowledge graph.

        Returns:
            Validation report with statistics and errors
        """
        entity_errors = {}
        relationship_errors = {}
        entity_map = {entity.entity_id: entity for entity in entities}

        # Validate entities
        for entity in entities:
            is_valid, errors = self.validate_entity(entity)
            if not is_valid:
                entity_errors[entity.entity_id] = errors

        # Validate relationships
        for relationship in relationships:
            source_entity = entity_map.get(relationship.source_entity_id)
            target_entity = entity_map.get(relationship.target_entity_id)

            if not source_entity:
                relationship_errors[relationship.relationship_id] = [
                    f"Source entity {relationship.source_entity_id} not found"
                ]
                continue

            if not target_entity:
                relationship_errors[relationship.relationship_id] = [
                    f"Target entity {relationship.target_entity_id} not found"
                ]
                continue

            is_valid, errors = self.validate_relationship(
                relationship, source_entity, target_entity
            )
            if not is_valid:
                relationship_errors[relationship.relationship_id] = errors

        # Generate report
        report = {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "valid_entities": len(entities) - len(entity_errors),
            "valid_relationships": len(relationships) - len(relationship_errors),
            "entity_errors": entity_errors,
            "relationship_errors": relationship_errors,
            "is_valid": len(entity_errors) == 0 and len(relationship_errors) == 0,
            "validation_timestamp": datetime.utcnow().isoformat(),
        }

        return report

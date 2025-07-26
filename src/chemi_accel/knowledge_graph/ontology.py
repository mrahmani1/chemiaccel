"""
Chemical Ontology and Entity Type Definitions

This module defines the comprehensive ontology for chemical entities,
properties, and relationships in the ChemiAccel knowledge graph system.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, validator


class EntityType(Enum):
    """
    Comprehensive enumeration of chemical entity types.

    This hierarchy follows established chemical informatics ontologies
    and provides fine-grained classification of chemical entities.
    """

    # Molecular entities
    MOLECULE = "molecule"
    SMALL_MOLECULE = "small_molecule"
    MACROMOLECULE = "macromolecule"
    POLYMER = "polymer"
    PROTEIN = "protein"
    NUCLEIC_ACID = "nucleic_acid"
    CARBOHYDRATE = "carbohydrate"
    LIPID = "lipid"

    # Chemical components
    ATOM = "atom"
    BOND = "bond"
    FUNCTIONAL_GROUP = "functional_group"
    FRAGMENT = "fragment"
    SCAFFOLD = "scaffold"
    PHARMACOPHORE = "pharmacophore"

    # Reactions and processes
    REACTION = "reaction"
    CHEMICAL_REACTION = "chemical_reaction"
    ENZYMATIC_REACTION = "enzymatic_reaction"
    METABOLIC_PATHWAY = "metabolic_pathway"
    SYNTHESIS_ROUTE = "synthesis_route"

    # Biological entities
    TARGET = "target"
    PROTEIN_TARGET = "protein_target"
    ENZYME = "enzyme"
    RECEPTOR = "receptor"
    ION_CHANNEL = "ion_channel"
    TRANSPORTER = "transporter"

    # Assays and measurements
    ASSAY = "assay"
    BIOASSAY = "bioassay"
    PHYSICO_CHEMICAL_ASSAY = "physico_chemical_assay"
    ENDPOINT = "endpoint"
    MEASUREMENT = "measurement"

    # Publications and data sources
    PUBLICATION = "publication"
    DATASET = "dataset"
    DATABASE = "database"
    EXPERIMENTAL_DATA = "experimental_data"

    # Chemical spaces and collections
    CHEMICAL_LIBRARY = "chemical_library"
    COMPOUND_SERIES = "compound_series"
    CHEMICAL_SPACE = "chemical_space"

    # Conditions and contexts
    EXPERIMENTAL_CONDITION = "experimental_condition"
    BIOLOGICAL_CONTEXT = "biological_context"
    DISEASE = "disease"
    PHENOTYPE = "phenotype"


class PropertyType(Enum):
    """
    Types of properties that can be associated with chemical entities.
    """

    # Molecular descriptors
    MOLECULAR_WEIGHT = "molecular_weight"
    LOGP = "logp"
    TPSA = "tpsa"
    HBD = "hydrogen_bond_donors"
    HBA = "hydrogen_bond_acceptors"
    ROTATABLE_BONDS = "rotatable_bonds"

    # Physical properties
    MELTING_POINT = "melting_point"
    BOILING_POINT = "boiling_point"
    DENSITY = "density"
    SOLUBILITY = "solubility"
    PKA = "pka"

    # Biological activities
    IC50 = "ic50"
    EC50 = "ec50"
    KI = "ki"
    KD = "kd"
    BIOAVAILABILITY = "bioavailability"
    HALF_LIFE = "half_life"

    # ADMET properties
    ABSORPTION = "absorption"
    DISTRIBUTION = "distribution"
    METABOLISM = "metabolism"
    EXCRETION = "excretion"
    TOXICITY = "toxicity"

    # Structural properties
    STEREOCHEMISTRY = "stereochemistry"
    CONFORMATION = "conformation"
    FINGERPRINT = "fingerprint"
    SCAFFOLD_STRUCTURE = "scaffold_structure"

    # Reaction properties
    REACTION_YIELD = "reaction_yield"
    REACTION_TIME = "reaction_time"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    CATALYST = "catalyst"


class RelationshipType(Enum):
    """
    Types of relationships between chemical entities.
    """

    # Molecular relationships
    CONTAINS = "contains"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    ANALOG_OF = "analog_of"
    METABOLITE_OF = "metabolite_of"
    PRECURSOR_OF = "precursor_of"

    # Biological interactions
    BINDS_TO = "binds_to"
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    MODULATES = "modulates"
    TARGETS = "targets"
    INTERACTS_WITH = "interacts_with"

    # Chemical transformations
    REACTS_WITH = "reacts_with"
    PRODUCES = "produces"
    CATALYZES = "catalyzes"
    TRANSFORMS_TO = "transforms_to"

    # Data relationships
    MEASURED_IN = "measured_in"
    REPORTED_IN = "reported_in"
    DERIVED_FROM = "derived_from"
    VALIDATED_BY = "validated_by"

    # Structural relationships
    ISOMER_OF = "isomer_of"
    ENANTIOMER_OF = "enantiomer_of"
    TAUTOMER_OF = "tautomer_of"
    SUBSTRUCTURE_OF = "substructure_of"

    # Classification relationships
    IS_A = "is_a"
    INSTANCE_OF = "instance_of"
    CLASSIFIED_AS = "classified_as"
    BELONGS_TO = "belongs_to"


@dataclass
class ChemicalEntity:
    """
    Base class for all chemical entities in the knowledge graph.

    Attributes:
        entity_id: Unique identifier for the entity
        entity_type: Type of the entity from EntityType enum
        name: Human-readable name
        description: Detailed description
        identifiers: External identifiers (e.g., ChEMBL, PubChem, SMILES)
        properties: Associated properties and their values
        metadata: Additional metadata
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType = field()
    name: str = field()
    description: Optional[str] = field(default=None)
    identifiers: Dict[str, str] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_property(
        self,
        property_type: PropertyType,
        value: Any,
        unit: Optional[str] = None,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
    ) -> None:
        """Add a property to the entity."""
        property_data = {
            "value": value,
            "type": property_type.value,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if unit:
            property_data["unit"] = unit
        if confidence is not None:
            property_data["confidence"] = confidence
        if source:
            property_data["source"] = source

        self.properties[property_type.value] = property_data
        self.updated_at = datetime.utcnow()

    def add_identifier(self, identifier_type: str, identifier_value: str) -> None:
        """Add an external identifier."""
        self.identifiers[identifier_type] = identifier_value
        self.updated_at = datetime.utcnow()

    def get_property(self, property_type: PropertyType) -> Optional[Any]:
        """Get property value by type."""
        prop_data = self.properties.get(property_type.value)
        return prop_data.get("value") if prop_data else None

    def has_property(self, property_type: PropertyType) -> bool:
        """Check if entity has a specific property."""
        return property_type.value in self.properties

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "description": self.description,
            "identifiers": self.identifiers.copy(),
            "properties": self.properties.copy(),
            "metadata": self.metadata.copy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ChemicalProperty:
    """
    Represents a property associated with a chemical entity.

    Properties can be numerical values, categorical classifications,
    or complex objects with associated metadata.
    """

    property_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    property_type: PropertyType = field()
    value: Any = field()
    unit: Optional[str] = field(default=None)
    confidence: Optional[float] = field(default=None)
    source: Optional[str] = field(default=None)
    measurement_method: Optional[str] = field(default=None)
    conditions: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[float] = field(default=None)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert property to dictionary representation."""
        return {
            "property_id": self.property_id,
            "property_type": self.property_type.value,
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "source": self.source,
            "measurement_method": self.measurement_method,
            "conditions": self.conditions.copy(),
            "uncertainty": self.uncertainty,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ChemicalRelationship:
    """
    Represents a relationship between two chemical entities.

    Relationships can be directional or bidirectional and may
    have associated properties and confidence scores.
    """

    relationship_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    relationship_type: RelationshipType = field()
    source_entity_id: str = field()
    target_entity_id: str = field()
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = field(default=None)
    source: Optional[str] = field(default=None)
    evidence: List[str] = field(default_factory=list)
    is_directional: bool = field(default=True)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_evidence(self, evidence_id: str) -> None:
        """Add evidence supporting this relationship."""
        if evidence_id not in self.evidence:
            self.evidence.append(evidence_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            "relationship_id": self.relationship_id,
            "relationship_type": self.relationship_type.value,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "properties": self.properties.copy(),
            "confidence": self.confidence,
            "source": self.source,
            "evidence": self.evidence.copy(),
            "is_directional": self.is_directional,
            "created_at": self.created_at.isoformat(),
        }


class ChemicalOntology:
    """
    Main ontology class that manages the chemical entity type hierarchy
    and provides semantic reasoning capabilities.
    """

    def __init__(self):
        self._entity_hierarchy = self._build_entity_hierarchy()
        self._property_constraints = self._build_property_constraints()
        self._relationship_constraints = self._build_relationship_constraints()

    def _build_entity_hierarchy(self) -> Dict[EntityType, Set[EntityType]]:
        """Build the entity type hierarchy."""
        hierarchy = {
            EntityType.MOLECULE: {
                EntityType.SMALL_MOLECULE,
                EntityType.MACROMOLECULE,
                EntityType.POLYMER,
                EntityType.PROTEIN,
                EntityType.NUCLEIC_ACID,
                EntityType.CARBOHYDRATE,
                EntityType.LIPID,
            },
            EntityType.TARGET: {
                EntityType.PROTEIN_TARGET,
                EntityType.ENZYME,
                EntityType.RECEPTOR,
                EntityType.ION_CHANNEL,
                EntityType.TRANSPORTER,
            },
            EntityType.REACTION: {
                EntityType.CHEMICAL_REACTION,
                EntityType.ENZYMATIC_REACTION,
                EntityType.METABOLIC_PATHWAY,
                EntityType.SYNTHESIS_ROUTE,
            },
            EntityType.ASSAY: {EntityType.BIOASSAY, EntityType.PHYSICO_CHEMICAL_ASSAY},
            EntityType.PUBLICATION: {
                EntityType.DATASET,
                EntityType.DATABASE,
                EntityType.EXPERIMENTAL_DATA,
            },
        }
        return hierarchy

    def _build_property_constraints(self) -> Dict[EntityType, Set[PropertyType]]:
        """Define which properties are valid for each entity type."""
        constraints = {
            EntityType.MOLECULE: {
                PropertyType.MOLECULAR_WEIGHT,
                PropertyType.LOGP,
                PropertyType.TPSA,
                PropertyType.HBD,
                PropertyType.HBA,
                PropertyType.ROTATABLE_BONDS,
                PropertyType.MELTING_POINT,
                PropertyType.BOILING_POINT,
                PropertyType.SOLUBILITY,
                PropertyType.PKA,
                PropertyType.STEREOCHEMISTRY,
                PropertyType.FINGERPRINT,
            },
            EntityType.SMALL_MOLECULE: {
                PropertyType.MOLECULAR_WEIGHT,
                PropertyType.LOGP,
                PropertyType.TPSA,
                PropertyType.HBD,
                PropertyType.HBA,
                PropertyType.ROTATABLE_BONDS,
                PropertyType.IC50,
                PropertyType.EC50,
                PropertyType.KI,
                PropertyType.KD,
                PropertyType.BIOAVAILABILITY,
                PropertyType.ABSORPTION,
                PropertyType.DISTRIBUTION,
                PropertyType.METABOLISM,
                PropertyType.EXCRETION,
                PropertyType.TOXICITY,
            },
            EntityType.REACTION: {
                PropertyType.REACTION_YIELD,
                PropertyType.REACTION_TIME,
                PropertyType.TEMPERATURE,
                PropertyType.PRESSURE,
                PropertyType.CATALYST,
            },
            EntityType.TARGET: {
                PropertyType.MOLECULAR_WEIGHT,
                PropertyType.STEREOCHEMISTRY,
                PropertyType.CONFORMATION,
            },
        }
        return constraints

    def _build_relationship_constraints(
        self,
    ) -> Dict[RelationshipType, Dict[str, Set[EntityType]]]:
        """Define valid entity types for relationship sources and targets."""
        constraints = {
            RelationshipType.BINDS_TO: {
                "source": {EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                "target": {
                    EntityType.TARGET,
                    EntityType.PROTEIN_TARGET,
                    EntityType.ENZYME,
                },
            },
            RelationshipType.INHIBITS: {
                "source": {EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                "target": {
                    EntityType.ENZYME,
                    EntityType.RECEPTOR,
                    EntityType.ION_CHANNEL,
                },
            },
            RelationshipType.REACTS_WITH: {
                "source": {EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                "target": {EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
            },
            RelationshipType.METABOLITE_OF: {
                "source": {EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                "target": {EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
            },
            RelationshipType.CONTAINS: {
                "source": {EntityType.MOLECULE, EntityType.MACROMOLECULE},
                "target": {
                    EntityType.FUNCTIONAL_GROUP,
                    EntityType.FRAGMENT,
                    EntityType.SCAFFOLD,
                },
            },
            RelationshipType.MEASURED_IN: {
                "source": {EntityType.MOLECULE, EntityType.TARGET},
                "target": {EntityType.ASSAY, EntityType.BIOASSAY},
            },
            RelationshipType.REPORTED_IN: {
                "source": {EntityType.MOLECULE, EntityType.MEASUREMENT},
                "target": {EntityType.PUBLICATION, EntityType.DATASET},
            },
        }
        return constraints

    def is_valid_entity_type(self, entity_type: EntityType) -> bool:
        """Check if an entity type is valid."""
        return isinstance(entity_type, EntityType)

    def is_subtype_of(self, child_type: EntityType, parent_type: EntityType) -> bool:
        """Check if child_type is a subtype of parent_type."""
        return child_type in self._entity_hierarchy.get(parent_type, set())

    def get_subtypes(self, entity_type: EntityType) -> Set[EntityType]:
        """Get all direct subtypes of an entity type."""
        return self._entity_hierarchy.get(entity_type, set())

    def get_all_subtypes(self, entity_type: EntityType) -> Set[EntityType]:
        """Get all subtypes (recursive) of an entity type."""
        subtypes = set()
        direct_subtypes = self.get_subtypes(entity_type)
        subtypes.update(direct_subtypes)

        for subtype in direct_subtypes:
            subtypes.update(self.get_all_subtypes(subtype))

        return subtypes

    def get_valid_properties(self, entity_type: EntityType) -> Set[PropertyType]:
        """Get valid properties for an entity type."""
        properties = self._property_constraints.get(entity_type, set())

        # Add properties from parent types
        for parent_type, children in self._entity_hierarchy.items():
            if entity_type in children:
                properties.update(self._property_constraints.get(parent_type, set()))

        return properties

    def is_valid_property(
        self, entity_type: EntityType, property_type: PropertyType
    ) -> bool:
        """Check if a property is valid for an entity type."""
        return property_type in self.get_valid_properties(entity_type)

    def is_valid_relationship(
        self,
        relationship_type: RelationshipType,
        source_type: EntityType,
        target_type: EntityType,
    ) -> bool:
        """Check if a relationship is valid between two entity types."""
        constraints = self._relationship_constraints.get(relationship_type)
        if not constraints:
            return True  # No constraints defined, allow any

        valid_sources = constraints.get("source", set())
        valid_targets = constraints.get("target", set())

        source_valid = not valid_sources or source_type in valid_sources
        target_valid = not valid_targets or target_type in valid_targets

        return source_valid and target_valid

    def get_valid_relationship_targets(
        self, relationship_type: RelationshipType, source_type: EntityType
    ) -> Set[EntityType]:
        """Get valid target types for a relationship from a source type."""
        constraints = self._relationship_constraints.get(relationship_type)
        if not constraints:
            return set(EntityType)  # No constraints, all types valid

        valid_sources = constraints.get("source", set())
        if valid_sources and source_type not in valid_sources:
            return set()  # Source type not valid for this relationship

        return constraints.get("target", set(EntityType))

    def validate_entity(self, entity: ChemicalEntity) -> List[str]:
        """Validate an entity against the ontology."""
        errors = []

        # Check entity type
        if not self.is_valid_entity_type(entity.entity_type):
            errors.append(f"Invalid entity type: {entity.entity_type}")

        # Check properties
        for prop_type_str in entity.properties:
            try:
                prop_type = PropertyType(prop_type_str)
                if not self.is_valid_property(entity.entity_type, prop_type):
                    errors.append(
                        f"Property {prop_type_str} not valid for entity type {entity.entity_type.value}"
                    )
            except ValueError:
                errors.append(f"Unknown property type: {prop_type_str}")

        return errors

    def validate_relationship(
        self,
        relationship: ChemicalRelationship,
        source_entity: ChemicalEntity,
        target_entity: ChemicalEntity,
    ) -> List[str]:
        """Validate a relationship against the ontology."""
        errors = []

        # Check relationship type validity
        if not self.is_valid_relationship(
            relationship.relationship_type,
            source_entity.entity_type,
            target_entity.entity_type,
        ):
            errors.append(
                f"Relationship {relationship.relationship_type.value} not valid "
                f"between {source_entity.entity_type.value} and {target_entity.entity_type.value}"
            )

        return errors

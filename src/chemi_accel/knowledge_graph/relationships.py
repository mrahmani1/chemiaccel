"""
Chemical Relationship Management and Predefined Relationships

This module provides comprehensive management of relationships between
chemical entities, including predefined relationship types, constraints,
and validation logic.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .ontology import (
    ChemicalEntity,
    ChemicalRelationship,
    EntityType,
    PropertyType,
    RelationshipType,
)


@dataclass
class RelationshipConstraints:
    """
    Defines constraints for a specific relationship type.

    Attributes:
        relationship_type: The type of relationship
        valid_source_types: Set of valid source entity types
        valid_target_types: Set of valid target entity types
        is_symmetric: Whether the relationship is symmetric (A->B implies B->A)
        is_transitive: Whether the relationship is transitive (A->B, B->C implies A->C)
        is_reflexive: Whether entities can have this relationship with themselves
        required_properties: Properties that must exist on entities for this relationship
        confidence_required: Whether confidence scores are required
        evidence_required: Whether evidence is required
        max_instances: Maximum number of instances allowed (None for unlimited)
    """

    relationship_type: RelationshipType
    valid_source_types: Set[EntityType] = field(default_factory=set)
    valid_target_types: Set[EntityType] = field(default_factory=set)
    is_symmetric: bool = False
    is_transitive: bool = False
    is_reflexive: bool = False
    required_properties: Dict[str, Set[PropertyType]] = field(
        default_factory=dict
    )  # 'source' or 'target'
    confidence_required: bool = False
    evidence_required: bool = False
    max_instances: Optional[int] = None
    description: Optional[str] = None

    def validates_entities(
        self, source_entity: ChemicalEntity, target_entity: ChemicalEntity
    ) -> Tuple[bool, List[str]]:
        """
        Validate if two entities can have this relationship.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check entity types
        if (
            self.valid_source_types
            and source_entity.entity_type not in self.valid_source_types
        ):
            errors.append(
                f"Source entity type {source_entity.entity_type.value} not valid for "
                f"relationship {self.relationship_type.value}"
            )

        if (
            self.valid_target_types
            and target_entity.entity_type not in self.valid_target_types
        ):
            errors.append(
                f"Target entity type {target_entity.entity_type.value} not valid for "
                f"relationship {self.relationship_type.value}"
            )

        # Check reflexivity
        if not self.is_reflexive and source_entity.entity_id == target_entity.entity_id:
            errors.append(
                f"Relationship {self.relationship_type.value} cannot be reflexive"
            )

        # Check required properties
        for role, required_props in self.required_properties.items():
            entity = source_entity if role == "source" else target_entity
            for prop_type in required_props:
                if not entity.has_property(prop_type):
                    errors.append(
                        f"{role.title()} entity must have property {prop_type.value} "
                        f"for relationship {self.relationship_type.value}"
                    )

        return len(errors) == 0, errors

    def validates_relationship(
        self, relationship: ChemicalRelationship
    ) -> Tuple[bool, List[str]]:
        """
        Validate a relationship instance against these constraints.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check confidence requirement
        if self.confidence_required and relationship.confidence is None:
            errors.append(
                f"Confidence score required for relationship {self.relationship_type.value}"
            )

        # Check evidence requirement
        if self.evidence_required and not relationship.evidence:
            errors.append(
                f"Evidence required for relationship {self.relationship_type.value}"
            )

        return len(errors) == 0, errors


class PreDefinedRelationships:
    """
    Registry of predefined relationship types with their constraints.

    This class provides a comprehensive set of commonly used chemical
    relationships with appropriate validation constraints.
    """

    @staticmethod
    def get_all_constraints() -> Dict[RelationshipType, RelationshipConstraints]:
        """Get all predefined relationship constraints."""
        return {
            # Molecular binding relationships
            RelationshipType.BINDS_TO: RelationshipConstraints(
                relationship_type=RelationshipType.BINDS_TO,
                valid_source_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                valid_target_types={
                    EntityType.TARGET,
                    EntityType.PROTEIN_TARGET,
                    EntityType.ENZYME,
                    EntityType.RECEPTOR,
                    EntityType.ION_CHANNEL,
                    EntityType.TRANSPORTER,
                },
                is_symmetric=False,
                is_transitive=False,
                is_reflexive=False,
                required_properties={
                    "source": {PropertyType.MOLECULAR_WEIGHT},
                    "target": {PropertyType.MOLECULAR_WEIGHT},
                },
                confidence_required=True,
                evidence_required=True,
                description="Molecular binding interaction between a ligand and target",
            ),
            # Inhibition relationships
            RelationshipType.INHIBITS: RelationshipConstraints(
                relationship_type=RelationshipType.INHIBITS,
                valid_source_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                valid_target_types={
                    EntityType.ENZYME,
                    EntityType.RECEPTOR,
                    EntityType.ION_CHANNEL,
                    EntityType.TRANSPORTER,
                    EntityType.PROTEIN_TARGET,
                },
                is_symmetric=False,
                is_transitive=False,
                is_reflexive=False,
                confidence_required=True,
                evidence_required=True,
                description="Inhibitory effect of a molecule on a biological target",
            ),
            # Activation relationships
            RelationshipType.ACTIVATES: RelationshipConstraints(
                relationship_type=RelationshipType.ACTIVATES,
                valid_source_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                valid_target_types={
                    EntityType.ENZYME,
                    EntityType.RECEPTOR,
                    EntityType.ION_CHANNEL,
                    EntityType.TRANSPORTER,
                    EntityType.PROTEIN_TARGET,
                },
                is_symmetric=False,
                is_transitive=False,
                is_reflexive=False,
                confidence_required=True,
                evidence_required=True,
                description="Activating effect of a molecule on a biological target",
            ),
            # Chemical reaction relationships
            RelationshipType.REACTS_WITH: RelationshipConstraints(
                relationship_type=RelationshipType.REACTS_WITH,
                valid_source_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                valid_target_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                is_symmetric=True,
                is_transitive=False,
                is_reflexive=False,
                description="Chemical reaction between two molecules",
            ),
            # Production relationships
            RelationshipType.PRODUCES: RelationshipConstraints(
                relationship_type=RelationshipType.PRODUCES,
                valid_source_types={
                    EntityType.REACTION,
                    EntityType.CHEMICAL_REACTION,
                    EntityType.ENZYMATIC_REACTION,
                },
                valid_target_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                is_symmetric=False,
                is_transitive=False,
                is_reflexive=False,
                description="Product formation in a chemical reaction",
            ),
            # Structural similarity relationships
            RelationshipType.SIMILAR_TO: RelationshipConstraints(
                relationship_type=RelationshipType.SIMILAR_TO,
                valid_source_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                valid_target_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                is_symmetric=True,
                is_transitive=False,
                is_reflexive=True,
                confidence_required=True,
                description="Structural or chemical similarity between molecules",
            ),
            # Analog relationships
            RelationshipType.ANALOG_OF: RelationshipConstraints(
                relationship_type=RelationshipType.ANALOG_OF,
                valid_source_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                valid_target_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                is_symmetric=True,
                is_transitive=False,
                is_reflexive=False,
                description="Chemical analog relationship",
            ),
            # Metabolic relationships
            RelationshipType.METABOLITE_OF: RelationshipConstraints(
                relationship_type=RelationshipType.METABOLITE_OF,
                valid_source_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                valid_target_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                is_symmetric=False,
                is_transitive=False,
                is_reflexive=False,
                description="Metabolite relationship between parent and metabolite",
            ),
            # Structural containment
            RelationshipType.CONTAINS: RelationshipConstraints(
                relationship_type=RelationshipType.CONTAINS,
                valid_source_types={
                    EntityType.MOLECULE,
                    EntityType.SMALL_MOLECULE,
                    EntityType.MACROMOLECULE,
                    EntityType.POLYMER,
                },
                valid_target_types={
                    EntityType.FUNCTIONAL_GROUP,
                    EntityType.FRAGMENT,
                    EntityType.SCAFFOLD,
                    EntityType.PHARMACOPHORE,
                    EntityType.ATOM,
                },
                is_symmetric=False,
                is_transitive=True,
                is_reflexive=False,
                description="Structural containment relationship",
            ),
            # Part-of relationships (inverse of contains)
            RelationshipType.PART_OF: RelationshipConstraints(
                relationship_type=RelationshipType.PART_OF,
                valid_source_types={
                    EntityType.FUNCTIONAL_GROUP,
                    EntityType.FRAGMENT,
                    EntityType.SCAFFOLD,
                    EntityType.PHARMACOPHORE,
                    EntityType.ATOM,
                },
                valid_target_types={
                    EntityType.MOLECULE,
                    EntityType.SMALL_MOLECULE,
                    EntityType.MACROMOLECULE,
                    EntityType.POLYMER,
                },
                is_symmetric=False,
                is_transitive=True,
                is_reflexive=False,
                description="Part-of structural relationship",
            ),
            # Stereochemical relationships
            RelationshipType.ENANTIOMER_OF: RelationshipConstraints(
                relationship_type=RelationshipType.ENANTIOMER_OF,
                valid_source_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                valid_target_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                is_symmetric=True,
                is_transitive=False,
                is_reflexive=False,
                required_properties={
                    "source": {PropertyType.STEREOCHEMISTRY},
                    "target": {PropertyType.STEREOCHEMISTRY},
                },
                description="Enantiomeric relationship between chiral molecules",
            ),
            RelationshipType.ISOMER_OF: RelationshipConstraints(
                relationship_type=RelationshipType.ISOMER_OF,
                valid_source_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                valid_target_types={EntityType.MOLECULE, EntityType.SMALL_MOLECULE},
                is_symmetric=True,
                is_transitive=False,
                is_reflexive=False,
                description="Isomeric relationship between molecules",
            ),
            # Measurement and data relationships
            RelationshipType.MEASURED_IN: RelationshipConstraints(
                relationship_type=RelationshipType.MEASURED_IN,
                valid_source_types={
                    EntityType.MOLECULE,
                    EntityType.SMALL_MOLECULE,
                    EntityType.TARGET,
                    EntityType.PROTEIN_TARGET,
                },
                valid_target_types={
                    EntityType.ASSAY,
                    EntityType.BIOASSAY,
                    EntityType.PHYSICO_CHEMICAL_ASSAY,
                },
                is_symmetric=False,
                is_transitive=False,
                is_reflexive=False,
                evidence_required=True,
                description="Entity measured in a specific assay",
            ),
            RelationshipType.REPORTED_IN: RelationshipConstraints(
                relationship_type=RelationshipType.REPORTED_IN,
                valid_source_types={
                    EntityType.MOLECULE,
                    EntityType.MEASUREMENT,
                    EntityType.EXPERIMENTAL_DATA,
                },
                valid_target_types={
                    EntityType.PUBLICATION,
                    EntityType.DATASET,
                    EntityType.DATABASE,
                },
                is_symmetric=False,
                is_transitive=False,
                is_reflexive=False,
                evidence_required=True,
                description="Data or measurement reported in a publication or dataset",
            ),
            # Classification relationships
            RelationshipType.IS_A: RelationshipConstraints(
                relationship_type=RelationshipType.IS_A,
                valid_source_types=set(EntityType),  # Any entity type
                valid_target_types=set(EntityType),  # Any entity type
                is_symmetric=False,
                is_transitive=True,
                is_reflexive=True,
                description="Taxonomic is-a relationship",
            ),
            RelationshipType.INSTANCE_OF: RelationshipConstraints(
                relationship_type=RelationshipType.INSTANCE_OF,
                valid_source_types=set(EntityType),
                valid_target_types=set(EntityType),
                is_symmetric=False,
                is_transitive=False,
                is_reflexive=False,
                description="Instance-of relationship for classification",
            ),
        }

    @staticmethod
    def get_constraints(
        relationship_type: RelationshipType,
    ) -> Optional[RelationshipConstraints]:
        """Get constraints for a specific relationship type."""
        all_constraints = PreDefinedRelationships.get_all_constraints()
        return all_constraints.get(relationship_type)

    @staticmethod
    def get_symmetric_relationships() -> Set[RelationshipType]:
        """Get all symmetric relationship types."""
        symmetric_rels = set()
        for (
            rel_type,
            constraints,
        ) in PreDefinedRelationships.get_all_constraints().items():
            if constraints.is_symmetric:
                symmetric_rels.add(rel_type)
        return symmetric_rels

    @staticmethod
    def get_transitive_relationships() -> Set[RelationshipType]:
        """Get all transitive relationship types."""
        transitive_rels = set()
        for (
            rel_type,
            constraints,
        ) in PreDefinedRelationships.get_all_constraints().items():
            if constraints.is_transitive:
                transitive_rels.add(rel_type)
        return transitive_rels


class ChemicalRelationshipManager:
    """
    Manages chemical relationships with validation, inference, and querying capabilities.

    This class provides a high-level interface for working with chemical relationships,
    including automatic validation, symmetry handling, and transitive closure computation.
    """

    def __init__(self, enable_inference: bool = True):
        """
        Initialize the relationship manager.

        Args:
            enable_inference: Whether to enable automatic relationship inference
        """
        self.relationships: Dict[str, ChemicalRelationship] = {}
        self.entities: Dict[str, ChemicalEntity] = {}
        self.constraints = PreDefinedRelationships.get_all_constraints()
        self.enable_inference = enable_inference

        # Index for efficient querying
        self._source_index: Dict[str, Set[str]] = {}  # entity_id -> relationship_ids
        self._target_index: Dict[str, Set[str]] = {}  # entity_id -> relationship_ids
        self._type_index: Dict[
            RelationshipType, Set[str]
        ] = {}  # rel_type -> relationship_ids

    def add_entity(self, entity: ChemicalEntity) -> None:
        """Add an entity to the manager."""
        self.entities[entity.entity_id] = entity
        if entity.entity_id not in self._source_index:
            self._source_index[entity.entity_id] = set()
        if entity.entity_id not in self._target_index:
            self._target_index[entity.entity_id] = set()

    def add_relationship(
        self, relationship: ChemicalRelationship, validate: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Add a relationship with optional validation.

        Args:
            relationship: The relationship to add
            validate: Whether to validate the relationship

        Returns:
            Tuple of (success, list_of_errors)
        """
        errors = []

        if validate:
            validation_errors = self.validate_relationship(relationship)
            if validation_errors:
                return False, validation_errors

        # Add the relationship
        self.relationships[relationship.relationship_id] = relationship

        # Update indices
        self._source_index.setdefault(relationship.source_entity_id, set()).add(
            relationship.relationship_id
        )
        self._target_index.setdefault(relationship.target_entity_id, set()).add(
            relationship.relationship_id
        )
        self._type_index.setdefault(relationship.relationship_type, set()).add(
            relationship.relationship_id
        )

        # Handle symmetric relationships
        if self.enable_inference:
            self._handle_symmetric_relationship(relationship)
            self._handle_transitive_relationships(relationship)

        return True, []

    def validate_relationship(self, relationship: ChemicalRelationship) -> List[str]:
        """
        Validate a relationship against all constraints.

        Returns:
            List of validation errors
        """
        errors = []

        # Check if entities exist
        source_entity = self.entities.get(relationship.source_entity_id)
        target_entity = self.entities.get(relationship.target_entity_id)

        if not source_entity:
            errors.append(f"Source entity {relationship.source_entity_id} not found")
            return errors

        if not target_entity:
            errors.append(f"Target entity {relationship.target_entity_id} not found")
            return errors

        # Check predefined constraints
        constraints = self.constraints.get(relationship.relationship_type)
        if constraints:
            entity_valid, entity_errors = constraints.validates_entities(
                source_entity, target_entity
            )
            errors.extend(entity_errors)

            rel_valid, rel_errors = constraints.validates_relationship(relationship)
            errors.extend(rel_errors)

        return errors

    def _handle_symmetric_relationship(
        self, relationship: ChemicalRelationship
    ) -> None:
        """Handle symmetric relationships by creating the inverse if needed."""
        constraints = self.constraints.get(relationship.relationship_type)
        if not constraints or not constraints.is_symmetric:
            return

        # Check if inverse relationship already exists
        inverse_id = f"inv_{relationship.relationship_id}"
        if inverse_id in self.relationships:
            return

        # Create inverse relationship
        inverse_relationship = ChemicalRelationship(
            relationship_id=inverse_id,
            relationship_type=relationship.relationship_type,
            source_entity_id=relationship.target_entity_id,
            target_entity_id=relationship.source_entity_id,
            properties=relationship.properties.copy(),
            confidence=relationship.confidence,
            source=relationship.source,
            evidence=relationship.evidence.copy(),
            is_directional=relationship.is_directional,
        )

        # Add without validation to avoid infinite recursion
        self.add_relationship(inverse_relationship, validate=False)

    def _handle_transitive_relationships(
        self, relationship: ChemicalRelationship
    ) -> None:
        """Handle transitive relationships by inferring new relationships."""
        constraints = self.constraints.get(relationship.relationship_type)
        if not constraints or not constraints.is_transitive:
            return

        # Find all relationships of the same type where this relationship's target
        # is the source of another relationship
        target_relationships = self.get_relationships_from_source(
            relationship.target_entity_id, relationship.relationship_type
        )

        for target_rel in target_relationships:
            # Create transitive relationship A->B, B->C => A->C
            transitive_id = (
                f"trans_{relationship.relationship_id}_{target_rel.relationship_id}"
            )

            if transitive_id not in self.relationships:
                transitive_relationship = ChemicalRelationship(
                    relationship_id=transitive_id,
                    relationship_type=relationship.relationship_type,
                    source_entity_id=relationship.source_entity_id,
                    target_entity_id=target_rel.target_entity_id,
                    properties={
                        "inferred": True,
                        "transitive_source": [
                            relationship.relationship_id,
                            target_rel.relationship_id,
                        ],
                    },
                    confidence=min(
                        relationship.confidence or 1.0, target_rel.confidence or 1.0
                    ),
                    source="transitive_inference",
                    evidence=[relationship.relationship_id, target_rel.relationship_id],
                    is_directional=True,
                )

                # Add without validation to avoid infinite recursion
                self.add_relationship(transitive_relationship, validate=False)

    def get_relationships_from_source(
        self,
        source_entity_id: str,
        relationship_type: Optional[RelationshipType] = None,
    ) -> List[ChemicalRelationship]:
        """Get all relationships originating from a source entity."""
        relationship_ids = self._source_index.get(source_entity_id, set())
        relationships = [self.relationships[rid] for rid in relationship_ids]

        if relationship_type:
            relationships = [
                r for r in relationships if r.relationship_type == relationship_type
            ]

        return relationships

    def get_relationships_to_target(
        self,
        target_entity_id: str,
        relationship_type: Optional[RelationshipType] = None,
    ) -> List[ChemicalRelationship]:
        """Get all relationships targeting an entity."""
        relationship_ids = self._target_index.get(target_entity_id, set())
        relationships = [self.relationships[rid] for rid in relationship_ids]

        if relationship_type:
            relationships = [
                r for r in relationships if r.relationship_type == relationship_type
            ]

        return relationships

    def get_relationships_by_type(
        self, relationship_type: RelationshipType
    ) -> List[ChemicalRelationship]:
        """Get all relationships of a specific type."""
        relationship_ids = self._type_index.get(relationship_type, set())
        return [self.relationships[rid] for rid in relationship_ids]

    def find_paths(
        self,
        source_entity_id: str,
        target_entity_id: str,
        max_depth: int = 3,
        relationship_types: Optional[Set[RelationshipType]] = None,
    ) -> List[List[ChemicalRelationship]]:
        """
        Find relationship paths between two entities.

        Args:
            source_entity_id: Starting entity ID
            target_entity_id: Target entity ID
            max_depth: Maximum path length
            relationship_types: Optional filter for relationship types

        Returns:
            List of relationship paths (each path is a list of relationships)
        """
        paths = []

        def dfs(
            current_id: str,
            target_id: str,
            current_path: List[ChemicalRelationship],
            visited: Set[str],
            depth: int,
        ):
            if depth > max_depth:
                return

            if current_id == target_id and current_path:
                paths.append(current_path.copy())
                return

            for relationship in self.get_relationships_from_source(current_id):
                if relationship.target_entity_id not in visited:
                    if (
                        not relationship_types
                        or relationship.relationship_type in relationship_types
                    ):
                        visited.add(relationship.target_entity_id)
                        current_path.append(relationship)
                        dfs(
                            relationship.target_entity_id,
                            target_id,
                            current_path,
                            visited,
                            depth + 1,
                        )
                        current_path.pop()
                        visited.remove(relationship.target_entity_id)

        dfs(source_entity_id, target_entity_id, [], {source_entity_id}, 0)
        return paths

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the relationship graph."""
        stats = {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "relationship_type_counts": {},
            "entities_by_type": {},
            "average_degree": 0.0,
            "most_connected_entities": [],
        }

        # Count relationships by type
        for relationship in self.relationships.values():
            rel_type = relationship.relationship_type.value
            stats["relationship_type_counts"][rel_type] = (
                stats["relationship_type_counts"].get(rel_type, 0) + 1
            )

        # Count entities by type
        for entity in self.entities.values():
            entity_type = entity.entity_type.value
            stats["entities_by_type"][entity_type] = (
                stats["entities_by_type"].get(entity_type, 0) + 1
            )

        # Calculate average degree
        total_connections = sum(len(rels) for rels in self._source_index.values())
        if self.entities:
            stats["average_degree"] = total_connections / len(self.entities)

        # Find most connected entities
        entity_degrees = []
        for entity_id in self.entities:
            out_degree = len(self._source_index.get(entity_id, set()))
            in_degree = len(self._target_index.get(entity_id, set()))
            total_degree = out_degree + in_degree
            entity_degrees.append((entity_id, total_degree))

        entity_degrees.sort(key=lambda x: x[1], reverse=True)
        stats["most_connected_entities"] = entity_degrees[:10]

        return stats

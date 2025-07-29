"""
Knowledge Graph Validation and Anomaly Detection

This module implements algorithms to detect inconsistencies, contradictions,
and data gaps within the IC-KG, and suggests targeted experiments to resolve conflicts.
"""

import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .graph_database import GraphDatabaseBackend, GraphQuery, GraphQueryResult
from .ontology import ChemicalEntity, ChemicalProperty, EntityType, PropertyType
from .reasoning_rules import AdvancedReasoningRule, ChemicalReasoningRuleLibrary
from .schema import EntitySchema, ICKGSchema, PropertySchema, SchemaValidator

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected in the knowledge graph."""

    INCONSISTENCY = "inconsistency"
    CONTRADICTION = "contradiction"
    DATA_GAP = "data_gap"
    OUTLIER = "outlier"
    SCHEMA_VIOLATION = "schema_violation"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    RELATIONSHIP_ANOMALY = "relationship_anomaly"
    DUPLICATE_ENTITY = "duplicate_entity"
    ORPHANED_NODE = "orphaned_node"
    CIRCULAR_DEPENDENCY = "circular_dependency"


class SeverityLevel(Enum):
    """Severity levels for detected anomalies."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DetectedAnomaly:
    """Represents a detected anomaly in the knowledge graph."""

    anomaly_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    description: str
    affected_entities: List[str]
    evidence: Dict[str, Any]
    confidence: float
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_suggestion: Optional[str] = None
    experimental_validation_needed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Comprehensive validation report for the knowledge graph."""

    report_id: str
    generated_at: datetime
    total_entities: int
    total_relationships: int
    anomalies_detected: List[DetectedAnomaly]
    validation_coverage: float
    data_quality_score: float
    recommendations: List[str]
    experimental_suggestions: List[str]
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityMetrics:
    """Metrics for assessing data quality in the knowledge graph."""

    completeness: float  # Percentage of complete entity profiles
    consistency: float  # Consistency score across data sources
    accuracy: float  # Accuracy based on validation rules
    freshness: float  # Recency of data updates
    validity: float  # Schema conformance score
    uniqueness: float  # Duplication detection score

    def overall_score(self) -> float:
        """Calculate overall data quality score."""
        weights = {
            "completeness": 0.2,
            "consistency": 0.25,
            "accuracy": 0.25,
            "freshness": 0.1,
            "validity": 0.1,
            "uniqueness": 0.1,
        }

        return sum(getattr(self, metric) * weight for metric, weight in weights.items())


class KnowledgeGraphValidator:
    """
    Main validator class for detecting anomalies and inconsistencies
    in the chemical knowledge graph.
    """

    def __init__(
        self,
        database: GraphDatabaseBackend,
        schema: ICKGSchema,
        reasoning_rules: ChemicalReasoningRuleLibrary,
        confidence_threshold: float = 0.7,
        enable_experimental_suggestions: bool = True,
    ):
        self.database = database
        self.schema = schema
        self.reasoning_rules = reasoning_rules
        self.confidence_threshold = confidence_threshold
        self.enable_experimental_suggestions = enable_experimental_suggestions

        # Validation statistics
        self.validation_history = []
        self.anomaly_patterns = defaultdict(list)

        # Configure anomaly detectors
        self._setup_anomaly_detectors()

    def _setup_anomaly_detectors(self) -> None:
        """Initialize anomaly detection algorithms."""
        self.detectors = {
            AnomalyType.INCONSISTENCY: self._detect_logical_inconsistencies,
            AnomalyType.CONTRADICTION: self._detect_contradictions,
            AnomalyType.DATA_GAP: self._detect_data_gaps,
            AnomalyType.OUTLIER: self._detect_statistical_outliers,
            AnomalyType.SCHEMA_VIOLATION: self._detect_schema_violations,
            AnomalyType.TEMPORAL_INCONSISTENCY: self._detect_temporal_inconsistencies,
            AnomalyType.RELATIONSHIP_ANOMALY: self._detect_relationship_anomalies,
            AnomalyType.DUPLICATE_ENTITY: self._detect_duplicate_entities,
            AnomalyType.ORPHANED_NODE: self._detect_orphaned_nodes,
            AnomalyType.CIRCULAR_DEPENDENCY: self._detect_circular_dependencies,
        }

    def validate_full_graph(self) -> ValidationReport:
        """
        Perform comprehensive validation of the entire knowledge graph.

        Returns:
            Complete validation report with all detected anomalies
        """
        logger.info("Starting comprehensive knowledge graph validation")
        start_time = datetime.now()

        # Gather basic statistics
        stats = self._gather_graph_statistics()

        # Detect all types of anomalies
        all_anomalies = []
        for anomaly_type, detector in self.detectors.items():
            logger.info(f"Running detector for {anomaly_type.value}")
            try:
                anomalies = detector()
                all_anomalies.extend(anomalies)
                logger.info(f"Found {len(anomalies)} {anomaly_type.value} anomalies")
            except Exception as e:
                logger.error(f"Error in {anomaly_type.value} detector: {e}")

        # Calculate data quality metrics
        quality_metrics = self._calculate_data_quality_metrics(all_anomalies, stats)

        # Generate recommendations
        recommendations = self._generate_recommendations(all_anomalies, quality_metrics)
        experimental_suggestions = self._generate_experimental_suggestions(
            all_anomalies
        )

        # Create validation report
        report = ValidationReport(
            report_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=start_time,
            total_entities=stats.get("total_entities", 0),
            total_relationships=stats.get("total_relationships", 0),
            anomalies_detected=all_anomalies,
            validation_coverage=self._calculate_validation_coverage(stats),
            data_quality_score=quality_metrics.overall_score(),
            recommendations=recommendations,
            experimental_suggestions=experimental_suggestions,
            statistics=stats,
        )

        # Store validation history
        self.validation_history.append(report)

        logger.info(
            f"Validation completed in {datetime.now() - start_time}. "
            f"Found {len(all_anomalies)} anomalies. "
            f"Data quality score: {quality_metrics.overall_score():.2f}"
        )

        return report

    def validate_entity(self, entity_id: str) -> List[DetectedAnomaly]:
        """
        Validate a specific entity and its relationships.

        Args:
            entity_id: Unique identifier of the entity to validate

        Returns:
            List of anomalies detected for this entity
        """
        logger.info(f"Validating entity: {entity_id}")

        anomalies = []

        # Get entity data
        entity_data = self._get_entity_data(entity_id)
        if not entity_data:
            anomalies.append(
                DetectedAnomaly(
                    anomaly_id=f"missing_entity_{entity_id}",
                    anomaly_type=AnomalyType.DATA_GAP,
                    severity=SeverityLevel.HIGH,
                    description=f"Entity {entity_id} not found in knowledge graph",
                    affected_entities=[entity_id],
                    evidence={"query_result": None},
                    confidence=1.0,
                )
            )
            return anomalies

        # Schema validation
        schema_anomalies = self._validate_entity_schema(entity_id, entity_data)
        anomalies.extend(schema_anomalies)

        # Property consistency validation
        property_anomalies = self._validate_entity_properties(entity_id, entity_data)
        anomalies.extend(property_anomalies)

        # Relationship validation
        relationship_anomalies = self._validate_entity_relationships(entity_id)
        anomalies.extend(relationship_anomalies)

        return anomalies

    def _detect_logical_inconsistencies(self) -> List[DetectedAnomaly]:
        """Detect logical inconsistencies using reasoning rules."""
        anomalies = []

        # Apply reasoning rules to detect inconsistencies
        for rule in self.reasoning_rules.get_all_rules():
            try:
                inconsistencies = self._apply_rule_for_validation(rule)
                anomalies.extend(inconsistencies)
            except Exception as e:
                logger.error(f"Error applying rule {rule.rule_id}: {e}")

        return anomalies

    def _detect_contradictions(self) -> List[DetectedAnomaly]:
        """Detect contradictory facts in the knowledge graph."""
        anomalies = []

        # Query for contradictory property values
        contradiction_query = GraphQuery(
            query="""
            MATCH (e:ChemicalEntity)
            WHERE exists(e.molecular_weight) AND exists(e.exact_mass)
            AND abs(e.molecular_weight - e.exact_mass) > 50
            RETURN e.id as entity_id, e.molecular_weight, e.exact_mass
            """,
            query_type="cypher",
        )

        try:
            result = self.database.execute_query(contradiction_query)
            for record in result.records:
                anomalies.append(
                    DetectedAnomaly(
                        anomaly_id=f"mass_contradiction_{record['entity_id']}",
                        anomaly_type=AnomalyType.CONTRADICTION,
                        severity=SeverityLevel.HIGH,
                        description="Large discrepancy between molecular weight and exact mass",
                        affected_entities=[record["entity_id"]],
                        evidence={
                            "molecular_weight": record["molecular_weight"],
                            "exact_mass": record["exact_mass"],
                            "difference": abs(
                                record["molecular_weight"] - record["exact_mass"]
                            ),
                        },
                        confidence=0.9,
                        experimental_validation_needed=True,
                    )
                )
        except Exception as e:
            logger.error(f"Error detecting contradictions: {e}")

        return anomalies

    def _detect_data_gaps(self) -> List[DetectedAnomaly]:
        """Detect missing data and incomplete entity profiles."""
        anomalies = []

        # Check for entities missing critical properties
        for entity_type in self.schema.entity_schemas:
            schema = self.schema.entity_schemas[entity_type]
            for required_prop in schema.required_properties:
                gap_query = GraphQuery(
                    query=f"""
                    MATCH (e:{entity_type.value})
                    WHERE NOT exists(e.{required_prop.value})
                    RETURN e.id as entity_id
                    LIMIT 100
                    """,
                    query_type="cypher",
                )

                try:
                    result = self.database.execute_query(gap_query)
                    for record in result.records:
                        anomalies.append(
                            DetectedAnomaly(
                                anomaly_id=f"missing_property_{record['entity_id']}_{required_prop.value}",
                                anomaly_type=AnomalyType.DATA_GAP,
                                severity=SeverityLevel.MEDIUM,
                                description=f"Missing required property: {required_prop.value}",
                                affected_entities=[record["entity_id"]],
                                evidence={"missing_property": required_prop.value},
                                confidence=1.0,
                                experimental_validation_needed=True,
                            )
                        )
                except Exception as e:
                    logger.error(
                        f"Error detecting data gaps for {required_prop.value}: {e}"
                    )

        return anomalies

    def _detect_statistical_outliers(self) -> List[DetectedAnomaly]:
        """Detect statistical outliers in numerical properties."""
        anomalies = []

        numerical_properties = ["molecular_weight", "logp", "tpsa", "hbd", "hba"]

        for prop in numerical_properties:
            outlier_query = GraphQuery(
                query=f"""
                MATCH (e:ChemicalEntity)
                WHERE exists(e.{prop}) AND e.{prop} IS NOT NULL
                RETURN e.id as entity_id, e.{prop} as value
                """,
                query_type="cypher",
            )

            try:
                result = self.database.execute_query(outlier_query)
                values = [record["value"] for record in result.records]

                if len(values) > 10:  # Need sufficient data for statistical analysis
                    outliers = self._detect_outliers_iqr(values, result.records)

                    for outlier in outliers:
                        anomalies.append(
                            DetectedAnomaly(
                                anomaly_id=f"outlier_{outlier['entity_id']}_{prop}",
                                anomaly_type=AnomalyType.OUTLIER,
                                severity=SeverityLevel.LOW,
                                description=f"Statistical outlier for property {prop}",
                                affected_entities=[outlier["entity_id"]],
                                evidence={
                                    "property": prop,
                                    "value": outlier["value"],
                                    "z_score": outlier["z_score"],
                                },
                                confidence=outlier["confidence"],
                                experimental_validation_needed=True,
                            )
                        )

            except Exception as e:
                logger.error(f"Error detecting outliers for {prop}: {e}")

        return anomalies

    def _detect_schema_violations(self) -> List[DetectedAnomaly]:
        """Detect violations of the defined schema."""
        anomalies = []

        for entity_type, schema in self.schema.entity_schemas.items():
            # Check for forbidden properties
            for forbidden_prop in schema.forbidden_properties:
                violation_query = GraphQuery(
                    query=f"""
                    MATCH (e:{entity_type.value})
                    WHERE exists(e.{forbidden_prop.value})
                    RETURN e.id as entity_id
                    """,
                    query_type="cypher",
                )

                try:
                    result = self.database.execute_query(violation_query)
                    for record in result.records:
                        anomalies.append(
                            DetectedAnomaly(
                                anomaly_id=f"schema_violation_{record['entity_id']}_{forbidden_prop.value}",
                                anomaly_type=AnomalyType.SCHEMA_VIOLATION,
                                severity=SeverityLevel.MEDIUM,
                                description=f"Entity has forbidden property: {forbidden_prop.value}",
                                affected_entities=[record["entity_id"]],
                                evidence={"forbidden_property": forbidden_prop.value},
                                confidence=1.0,
                            )
                        )
                except Exception as e:
                    logger.error(f"Error detecting schema violations: {e}")

        return anomalies

    def _detect_temporal_inconsistencies(self) -> List[DetectedAnomaly]:
        """Detect temporal inconsistencies in time-dependent data."""
        anomalies = []

        # Check for future dates in historical data
        temporal_query = GraphQuery(
            query="""
            MATCH (e:ChemicalEntity)
            WHERE exists(e.discovery_date)
            AND date(e.discovery_date) > date()
            RETURN e.id as entity_id, e.discovery_date
            """,
            query_type="cypher",
        )

        try:
            result = self.database.execute_query(temporal_query)
            for record in result.records:
                anomalies.append(
                    DetectedAnomaly(
                        anomaly_id=f"future_date_{record['entity_id']}",
                        anomaly_type=AnomalyType.TEMPORAL_INCONSISTENCY,
                        severity=SeverityLevel.HIGH,
                        description="Discovery date is in the future",
                        affected_entities=[record["entity_id"]],
                        evidence={"discovery_date": record["discovery_date"]},
                        confidence=1.0,
                    )
                )
        except Exception as e:
            logger.error(f"Error detecting temporal inconsistencies: {e}")

        return anomalies

    def _detect_relationship_anomalies(self) -> List[DetectedAnomaly]:
        """Detect anomalous relationships between entities."""
        anomalies = []

        # Check for duplicate relationships
        duplicate_query = GraphQuery(
            query="""
            MATCH (a)-[r1:SIMILAR_TO]->(b), (a)-[r2:SIMILAR_TO]->(b)
            WHERE id(r1) < id(r2)
            RETURN a.id as entity1, b.id as entity2, count(*) as duplicates
            """,
            query_type="cypher",
        )

        try:
            result = self.database.execute_query(duplicate_query)
            for record in result.records:
                anomalies.append(
                    DetectedAnomaly(
                        anomaly_id=f"duplicate_relationship_{record['entity1']}_{record['entity2']}",
                        anomaly_type=AnomalyType.RELATIONSHIP_ANOMALY,
                        severity=SeverityLevel.MEDIUM,
                        description="Duplicate relationships between entities",
                        affected_entities=[record["entity1"], record["entity2"]],
                        evidence={"duplicate_count": record["duplicates"]},
                        confidence=1.0,
                    )
                )
        except Exception as e:
            logger.error(f"Error detecting relationship anomalies: {e}")

        return anomalies

    def _detect_duplicate_entities(self) -> List[DetectedAnomaly]:
        """Detect potential duplicate entities."""
        anomalies = []

        # Check for entities with identical SMILES
        duplicate_query = GraphQuery(
            query="""
            MATCH (e1:ChemicalEntity), (e2:ChemicalEntity)
            WHERE e1.smiles = e2.smiles
            AND e1.id < e2.id
            AND e1.smiles IS NOT NULL
            RETURN e1.id as entity1, e2.id as entity2, e1.smiles as smiles
            """,
            query_type="cypher",
        )

        try:
            result = self.database.execute_query(duplicate_query)
            for record in result.records:
                anomalies.append(
                    DetectedAnomaly(
                        anomaly_id=f"duplicate_smiles_{record['entity1']}_{record['entity2']}",
                        anomaly_type=AnomalyType.DUPLICATE_ENTITY,
                        severity=SeverityLevel.HIGH,
                        description="Entities with identical SMILES strings",
                        affected_entities=[record["entity1"], record["entity2"]],
                        evidence={"smiles": record["smiles"]},
                        confidence=0.95,
                    )
                )
        except Exception as e:
            logger.error(f"Error detecting duplicate entities: {e}")

        return anomalies

    def _detect_orphaned_nodes(self) -> List[DetectedAnomaly]:
        """Detect orphaned nodes with no relationships."""
        anomalies = []

        orphan_query = GraphQuery(
            query="""
            MATCH (e:ChemicalEntity)
            WHERE NOT (e)--()
            RETURN e.id as entity_id
            LIMIT 100
            """,
            query_type="cypher",
        )

        try:
            result = self.database.execute_query(orphan_query)
            for record in result.records:
                anomalies.append(
                    DetectedAnomaly(
                        anomaly_id=f"orphaned_node_{record['entity_id']}",
                        anomaly_type=AnomalyType.ORPHANED_NODE,
                        severity=SeverityLevel.LOW,
                        description="Entity has no relationships",
                        affected_entities=[record["entity_id"]],
                        evidence={},
                        confidence=1.0,
                    )
                )
        except Exception as e:
            logger.error(f"Error detecting orphaned nodes: {e}")

        return anomalies

    def _detect_circular_dependencies(self) -> List[DetectedAnomaly]:
        """Detect circular dependencies in hierarchical relationships."""
        anomalies = []

        # Check for circular parent-child relationships
        circular_query = GraphQuery(
            query="""
            MATCH path = (a:ChemicalEntity)-[:PARENT_OF*]->(a)
            WHERE length(path) > 1
            RETURN a.id as entity_id, length(path) as cycle_length
            """,
            query_type="cypher",
        )

        try:
            result = self.database.execute_query(circular_query)
            for record in result.records:
                anomalies.append(
                    DetectedAnomaly(
                        anomaly_id=f"circular_dependency_{record['entity_id']}",
                        anomaly_type=AnomalyType.CIRCULAR_DEPENDENCY,
                        severity=SeverityLevel.HIGH,
                        description="Circular dependency in hierarchical relationships",
                        affected_entities=[record["entity_id"]],
                        evidence={"cycle_length": record["cycle_length"]},
                        confidence=1.0,
                    )
                )
        except Exception as e:
            logger.error(f"Error detecting circular dependencies: {e}")

        return anomalies

    def _detect_outliers_iqr(
        self, values: List[float], records: List[Dict]
    ) -> List[Dict]:
        """Detect outliers using the IQR method."""
        if len(values) < 4:
            return []

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mean_val = np.mean(values)
        std_val = np.std(values)

        outliers = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
                confidence = min(0.95, max(0.5, z_score / 3.0))

                outliers.append(
                    {
                        "entity_id": records[i]["entity_id"],
                        "value": value,
                        "z_score": z_score,
                        "confidence": confidence,
                    }
                )

        return outliers

    def _gather_graph_statistics(self) -> Dict[str, Any]:
        """Gather basic statistics about the knowledge graph."""
        stats = {}

        try:
            # Count entities by type
            entity_count_query = GraphQuery(
                query="MATCH (n) RETURN labels(n) as labels, count(n) as count",
                query_type="cypher",
            )
            result = self.database.execute_query(entity_count_query)

            entity_counts = {}
            total_entities = 0
            for record in result.records:
                labels = record["labels"]
                count = record["count"]
                entity_counts[str(labels)] = count
                total_entities += count

            stats["entity_counts"] = entity_counts
            stats["total_entities"] = total_entities

            # Count relationships
            rel_count_query = GraphQuery(
                query="MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count",
                query_type="cypher",
            )
            result = self.database.execute_query(rel_count_query)

            relationship_counts = {}
            total_relationships = 0
            for record in result.records:
                rel_type = record["rel_type"]
                count = record["count"]
                relationship_counts[rel_type] = count
                total_relationships += count

            stats["relationship_counts"] = relationship_counts
            stats["total_relationships"] = total_relationships

        except Exception as e:
            logger.error(f"Error gathering graph statistics: {e}")
            stats = {"total_entities": 0, "total_relationships": 0}

        return stats

    def _calculate_data_quality_metrics(
        self, anomalies: List[DetectedAnomaly], stats: Dict[str, Any]
    ) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics."""

        total_entities = stats.get("total_entities", 1)

        # Count anomalies by severity
        severity_counts = Counter(anomaly.severity for anomaly in anomalies)

        # Calculate individual metrics
        completeness = 1.0 - (
            len([a for a in anomalies if a.anomaly_type == AnomalyType.DATA_GAP])
            / total_entities
        )
        completeness = max(0.0, min(1.0, completeness))

        consistency = 1.0 - (
            len([a for a in anomalies if a.anomaly_type == AnomalyType.INCONSISTENCY])
            / total_entities
        )
        consistency = max(0.0, min(1.0, consistency))

        accuracy = 1.0 - (
            len([a for a in anomalies if a.anomaly_type == AnomalyType.CONTRADICTION])
            / total_entities
        )
        accuracy = max(0.0, min(1.0, accuracy))

        validity = 1.0 - (
            len(
                [a for a in anomalies if a.anomaly_type == AnomalyType.SCHEMA_VIOLATION]
            )
            / total_entities
        )
        validity = max(0.0, min(1.0, validity))

        uniqueness = 1.0 - (
            len(
                [a for a in anomalies if a.anomaly_type == AnomalyType.DUPLICATE_ENTITY]
            )
            / total_entities
        )
        uniqueness = max(0.0, min(1.0, uniqueness))

        # For freshness, assume 0.8 as a baseline (would need temporal data to calculate properly)
        freshness = 0.8

        return DataQualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            freshness=freshness,
            validity=validity,
            uniqueness=uniqueness,
        )

    def _calculate_validation_coverage(self, stats: Dict[str, Any]) -> float:
        """Calculate the percentage of the graph that was validated."""
        # For now, assume full coverage
        return 1.0

    def _generate_recommendations(
        self, anomalies: List[DetectedAnomaly], quality_metrics: DataQualityMetrics
    ) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []

        # Count anomalies by type
        anomaly_counts = Counter(anomaly.anomaly_type for anomaly in anomalies)

        if anomaly_counts[AnomalyType.DATA_GAP] > 10:
            recommendations.append(
                "High number of data gaps detected. Consider implementing "
                "automated data completion from external sources."
            )

        if anomaly_counts[AnomalyType.DUPLICATE_ENTITY] > 5:
            recommendations.append(
                "Multiple duplicate entities found. Implement entity resolution "
                "and deduplication procedures."
            )

        if anomaly_counts[AnomalyType.CONTRADICTION] > 3:
            recommendations.append(
                "Contradictory data detected. Review data source reliability "
                "and implement conflict resolution strategies."
            )

        if quality_metrics.overall_score() < 0.7:
            recommendations.append(
                "Overall data quality is below acceptable threshold. "
                "Priority should be given to data curation efforts."
            )

        return recommendations

    def _generate_experimental_suggestions(
        self, anomalies: List[DetectedAnomaly]
    ) -> List[str]:
        """Generate suggestions for experimental validation of anomalies."""
        suggestions = []

        experimental_anomalies = [
            a
            for a in anomalies
            if a.experimental_validation_needed
            and a.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        ]

        if experimental_anomalies:
            # Group by entity type for targeted suggestions
            entity_groups = defaultdict(list)
            for anomaly in experimental_anomalies:
                for entity_id in anomaly.affected_entities:
                    entity_groups[self._get_entity_type(entity_id)].append(anomaly)

            for entity_type, type_anomalies in entity_groups.items():
                if len(type_anomalies) > 5:
                    suggestions.append(
                        f"High-priority experimental validation needed for {len(type_anomalies)} "
                        f"{entity_type} entities with data inconsistencies."
                    )

        return suggestions

    def _get_entity_type(self, entity_id: str) -> str:
        """Get the type of an entity."""
        # Simplified implementation - would need actual query
        return "ChemicalEntity"

    def _get_entity_data(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve entity data from the database."""
        try:
            query = GraphQuery(
                query="MATCH (e) WHERE e.id = $entity_id RETURN e",
                parameters={"entity_id": entity_id},
                query_type="cypher",
            )
            result = self.database.execute_query(query)

            if result.records:
                return result.records[0]["e"]
            return None

        except Exception as e:
            logger.error(f"Error retrieving entity data for {entity_id}: {e}")
            return None

    def _validate_entity_schema(
        self, entity_id: str, entity_data: Dict[str, Any]
    ) -> List[DetectedAnomaly]:
        """Validate entity against schema."""
        # Simplified implementation
        return []

    def _validate_entity_properties(
        self, entity_id: str, entity_data: Dict[str, Any]
    ) -> List[DetectedAnomaly]:
        """Validate entity properties for consistency."""
        # Simplified implementation
        return []

    def _validate_entity_relationships(self, entity_id: str) -> List[DetectedAnomaly]:
        """Validate entity relationships."""
        # Simplified implementation
        return []

    def _apply_rule_for_validation(
        self, rule: AdvancedReasoningRule
    ) -> List[DetectedAnomaly]:
        """Apply a reasoning rule to detect inconsistencies."""
        # Simplified implementation
        return []


class ExperimentSuggestionEngine:
    """
    Engine for suggesting targeted experiments to resolve data conflicts
    and fill knowledge gaps in the chemical knowledge graph.
    """

    def __init__(self, validator: KnowledgeGraphValidator):
        self.validator = validator
        self.suggestion_templates = self._load_suggestion_templates()

    def suggest_experiments(
        self, anomalies: List[DetectedAnomaly]
    ) -> List[Dict[str, Any]]:
        """
        Generate specific experimental suggestions to resolve anomalies.

        Args:
            anomalies: List of detected anomalies

        Returns:
            List of experimental suggestions with protocols and priorities
        """
        suggestions = []

        # Group anomalies by type and entity
        grouped_anomalies = self._group_anomalies_for_experiments(anomalies)

        for group_key, group_anomalies in grouped_anomalies.items():
            suggestion = self._generate_experiment_suggestion(
                group_key, group_anomalies
            )
            if suggestion:
                suggestions.append(suggestion)

        # Prioritize suggestions
        suggestions.sort(key=lambda x: x["priority"], reverse=True)

        return suggestions

    def _group_anomalies_for_experiments(
        self, anomalies: List[DetectedAnomaly]
    ) -> Dict[str, List[DetectedAnomaly]]:
        """Group anomalies that can be resolved by similar experiments."""
        groups = defaultdict(list)

        for anomaly in anomalies:
            if not anomaly.experimental_validation_needed:
                continue

            # Create grouping key based on anomaly type and affected properties
            group_key = f"{anomaly.anomaly_type.value}"
            if "property" in anomaly.evidence:
                group_key += f"_{anomaly.evidence['property']}"

            groups[group_key].append(anomaly)

        return groups

    def _generate_experiment_suggestion(
        self, group_key: str, anomalies: List[DetectedAnomaly]
    ) -> Optional[Dict[str, Any]]:
        """Generate an experimental suggestion for a group of anomalies."""

        if not anomalies:
            return None

        # Calculate priority based on severity and count
        priority = self._calculate_experiment_priority(anomalies)

        # Generate suggestion based on anomaly type
        anomaly_type = anomalies[0].anomaly_type

        if anomaly_type == AnomalyType.CONTRADICTION:
            return self._suggest_property_measurement_experiment(anomalies, priority)
        elif anomaly_type == AnomalyType.DATA_GAP:
            return self._suggest_property_completion_experiment(anomalies, priority)
        elif anomaly_type == AnomalyType.OUTLIER:
            return self._suggest_outlier_validation_experiment(anomalies, priority)

        return None

    def _calculate_experiment_priority(self, anomalies: List[DetectedAnomaly]) -> float:
        """Calculate priority score for experimental suggestions."""
        severity_weights = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.6,
            SeverityLevel.LOW: 0.4,
            SeverityLevel.INFO: 0.2,
        }

        total_weight = sum(severity_weights[a.severity] for a in anomalies)
        count_bonus = min(
            0.3, len(anomalies) * 0.05
        )  # Bonus for affecting many entities

        return min(1.0, total_weight / len(anomalies) + count_bonus)

    def _suggest_property_measurement_experiment(
        self, anomalies: List[DetectedAnomaly], priority: float
    ) -> Dict[str, Any]:
        """Suggest experiments to resolve property contradictions."""

        affected_entities = set()
        for anomaly in anomalies:
            affected_entities.update(anomaly.affected_entities)

        return {
            "experiment_id": f"property_measurement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "property_measurement",
            "title": "Resolve Property Value Contradictions",
            "description": f"Experimental validation needed for {len(affected_entities)} entities with contradictory property values",
            "affected_entities": list(affected_entities),
            "priority": priority,
            "estimated_cost": "Medium",
            "estimated_time": "2-4 weeks",
            "protocol_suggestions": [
                "High-resolution mass spectrometry for exact mass determination",
                "Analytical balance measurement for molecular weight confirmation",
                "NMR spectroscopy for structural verification",
            ],
            "expected_outcomes": [
                "Resolution of mass value contradictions",
                "Improved data quality and reliability",
                "Updated knowledge graph with validated values",
            ],
        }

    def _suggest_property_completion_experiment(
        self, anomalies: List[DetectedAnomaly], priority: float
    ) -> Dict[str, Any]:
        """Suggest experiments to fill data gaps."""

        missing_properties = set()
        affected_entities = set()

        for anomaly in anomalies:
            if "missing_property" in anomaly.evidence:
                missing_properties.add(anomaly.evidence["missing_property"])
            affected_entities.update(anomaly.affected_entities)

        return {
            "experiment_id": f"property_completion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "property_completion",
            "title": "Fill Critical Data Gaps",
            "description": f"Measurement of missing properties for {len(affected_entities)} entities",
            "missing_properties": list(missing_properties),
            "affected_entities": list(affected_entities),
            "priority": priority,
            "estimated_cost": "Low to Medium",
            "estimated_time": "1-3 weeks",
            "protocol_suggestions": self._get_measurement_protocols(missing_properties),
            "expected_outcomes": [
                "Complete entity profiles in knowledge graph",
                "Enhanced predictive model accuracy",
                "Reduced uncertainty in AI-driven predictions",
            ],
        }

    def _suggest_outlier_validation_experiment(
        self, anomalies: List[DetectedAnomaly], priority: float
    ) -> Dict[str, Any]:
        """Suggest experiments to validate statistical outliers."""

        outlier_entities = set()
        outlier_properties = set()

        for anomaly in anomalies:
            outlier_entities.update(anomaly.affected_entities)
            if "property" in anomaly.evidence:
                outlier_properties.add(anomaly.evidence["property"])

        return {
            "experiment_id": f"outlier_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "outlier_validation",
            "title": "Validate Statistical Outliers",
            "description": f"Experimental validation of outlier values for {len(outlier_entities)} entities",
            "outlier_properties": list(outlier_properties),
            "affected_entities": list(outlier_entities),
            "priority": priority,
            "estimated_cost": "Medium",
            "estimated_time": "2-3 weeks",
            "protocol_suggestions": [
                "Repeat measurements with independent methods",
                "Cross-validation with reference standards",
                "Interlaboratory comparison studies",
            ],
            "expected_outcomes": [
                "Confirmation or correction of outlier values",
                "Improved statistical models",
                "Enhanced data reliability",
            ],
        }

    def _get_measurement_protocols(self, properties: Set[str]) -> List[str]:
        """Get measurement protocols for specific properties."""
        protocol_map = {
            "molecular_weight": "Mass spectrometry or analytical balance",
            "logp": "Shake-flask method or HPLC",
            "tpsa": "Computational calculation with experimental validation",
            "melting_point": "Differential scanning calorimetry (DSC)",
            "boiling_point": "Distillation or thermogravimetric analysis",
            "solubility": "Saturation shake-flask method",
        }

        protocols = []
        for prop in properties:
            if prop in protocol_map:
                protocols.append(f"{prop}: {protocol_map[prop]}")
            else:
                protocols.append(f"{prop}: Standard analytical method")

        return protocols

    def _load_suggestion_templates(self) -> Dict[str, Any]:
        """Load experiment suggestion templates."""
        # Placeholder for loading templates from configuration
        return {}

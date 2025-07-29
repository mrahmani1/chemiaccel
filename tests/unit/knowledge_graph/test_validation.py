"""
Tests for Knowledge Graph Validation and Anomaly Detection Module
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from chemi_accel.knowledge_graph.graph_database import GraphQuery, GraphQueryResult
from chemi_accel.knowledge_graph.reasoning_rules import ChemicalReasoningRuleLibrary
from chemi_accel.knowledge_graph.schema import ICKGSchema
from chemi_accel.knowledge_graph.validation import (
    AnomalyType,
    DataQualityMetrics,
    DetectedAnomaly,
    ExperimentSuggestionEngine,
    KnowledgeGraphValidator,
    SeverityLevel,
    ValidationReport,
)


class TestDetectedAnomaly:
    """Test DetectedAnomaly dataclass."""

    def test_anomaly_creation(self):
        """Test creation of DetectedAnomaly."""
        anomaly = DetectedAnomaly(
            anomaly_id="test_001",
            anomaly_type=AnomalyType.INCONSISTENCY,
            severity=SeverityLevel.HIGH,
            description="Test anomaly",
            affected_entities=["entity_1", "entity_2"],
            evidence={"test_key": "test_value"},
            confidence=0.9,
        )

        assert anomaly.anomaly_id == "test_001"
        assert anomaly.anomaly_type == AnomalyType.INCONSISTENCY
        assert anomaly.severity == SeverityLevel.HIGH
        assert len(anomaly.affected_entities) == 2
        assert anomaly.confidence == 0.9
        assert not anomaly.resolved


class TestDataQualityMetrics:
    """Test DataQualityMetrics calculations."""

    def test_overall_score_calculation(self):
        """Test overall data quality score calculation."""
        metrics = DataQualityMetrics(
            completeness=0.9,
            consistency=0.8,
            accuracy=0.85,
            freshness=0.7,
            validity=0.95,
            uniqueness=0.9,
        )

        # Expected: 0.9*0.2 + 0.8*0.25 + 0.85*0.25 + 0.7*0.1 + 0.95*0.1 + 0.9*0.1 = 0.8425
        expected_score = 0.8425
        assert abs(metrics.overall_score() - expected_score) < 0.001

    def test_perfect_quality_score(self):
        """Test perfect quality score."""
        metrics = DataQualityMetrics(
            completeness=1.0,
            consistency=1.0,
            accuracy=1.0,
            freshness=1.0,
            validity=1.0,
            uniqueness=1.0,
        )

        assert metrics.overall_score() == 1.0


class TestKnowledgeGraphValidator:
    """Test KnowledgeGraphValidator functionality."""

    @pytest.fixture
    def mock_database(self):
        """Mock database backend."""
        database = MagicMock()
        return database

    @pytest.fixture
    def mock_schema(self):
        """Mock ICKG schema."""
        schema = MagicMock(spec=ICKGSchema)
        schema.entity_schemas = {}
        return schema

    @pytest.fixture
    def mock_reasoning_rules(self):
        """Mock reasoning rules library."""
        rules = MagicMock(spec=ChemicalReasoningRuleLibrary)
        rules.get_all_rules.return_value = []
        return rules

    @pytest.fixture
    def validator(self, mock_database, mock_schema, mock_reasoning_rules):
        """Create validator instance."""
        return KnowledgeGraphValidator(
            database=mock_database,
            schema=mock_schema,
            reasoning_rules=mock_reasoning_rules,
        )

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.confidence_threshold == 0.7
        assert validator.enable_experimental_suggestions is True
        assert len(validator.detectors) == 10  # Number of anomaly types

    def test_detect_contradictions(self, validator, mock_database):
        """Test contradiction detection."""
        # Mock database response for contradictions
        mock_result = GraphQueryResult(
            records=[
                {
                    "entity_id": "entity_001",
                    "molecular_weight": 200.0,
                    "exact_mass": 150.0,
                }
            ],
            summary={},
            execution_time=0.1,
            query=GraphQuery(query="test"),
        )
        mock_database.execute_query.return_value = mock_result

        anomalies = validator._detect_contradictions()

        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.CONTRADICTION
        assert anomalies[0].severity == SeverityLevel.HIGH
        assert "entity_001" in anomalies[0].affected_entities
        assert anomalies[0].experimental_validation_needed is True

    def test_detect_duplicate_entities(self, validator, mock_database):
        """Test duplicate entity detection."""
        mock_result = GraphQueryResult(
            records=[
                {"entity1": "entity_001", "entity2": "entity_002", "smiles": "CCO"}
            ],
            summary={},
            execution_time=0.1,
            query=GraphQuery(query="test"),
        )
        mock_database.execute_query.return_value = mock_result

        anomalies = validator._detect_duplicate_entities()

        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.DUPLICATE_ENTITY
        assert anomalies[0].severity == SeverityLevel.HIGH
        assert len(anomalies[0].affected_entities) == 2

    def test_detect_outliers_iqr(self, validator):
        """Test IQR outlier detection."""
        values = [1, 2, 3, 4, 5, 100]  # 100 is clear outlier
        records = [{"entity_id": f"entity_{i}"} for i in range(len(values))]

        outliers = validator._detect_outliers_iqr(values, records)

        assert len(outliers) == 1
        assert outliers[0]["entity_id"] == "entity_5"
        assert outliers[0]["value"] == 100
        assert outliers[0]["confidence"] > 0.5

    def test_gather_graph_statistics(self, validator, mock_database):
        """Test graph statistics gathering."""
        # Mock entity count query
        entity_result = GraphQueryResult(
            records=[
                {"labels": ["ChemicalEntity"], "count": 100},
                {"labels": ["Drug"], "count": 50},
            ],
            summary={},
            execution_time=0.1,
            query=GraphQuery(query="test"),
        )

        # Mock relationship count query
        rel_result = GraphQueryResult(
            records=[
                {"rel_type": "SIMILAR_TO", "count": 200},
                {"rel_type": "TARGETS", "count": 150},
            ],
            summary={},
            execution_time=0.1,
            query=GraphQuery(query="test"),
        )

        mock_database.execute_query.side_effect = [entity_result, rel_result]

        stats = validator._gather_graph_statistics()

        assert stats["total_entities"] == 150
        assert stats["total_relationships"] == 350
        assert "entity_counts" in stats
        assert "relationship_counts" in stats

    def test_calculate_data_quality_metrics(self, validator):
        """Test data quality metrics calculation."""
        anomalies = [
            DetectedAnomaly(
                anomaly_id="test_1",
                anomaly_type=AnomalyType.DATA_GAP,
                severity=SeverityLevel.MEDIUM,
                description="Test",
                affected_entities=["entity_1"],
                evidence={},
                confidence=0.9,
            ),
            DetectedAnomaly(
                anomaly_id="test_2",
                anomaly_type=AnomalyType.INCONSISTENCY,
                severity=SeverityLevel.HIGH,
                description="Test",
                affected_entities=["entity_2"],
                evidence={},
                confidence=0.8,
            ),
        ]

        stats = {"total_entities": 100}

        metrics = validator._calculate_data_quality_metrics(anomalies, stats)

        assert 0 <= metrics.completeness <= 1
        assert 0 <= metrics.consistency <= 1
        assert 0 <= metrics.overall_score() <= 1

    def test_generate_recommendations(self, validator):
        """Test recommendation generation."""
        anomalies = [
            DetectedAnomaly(
                anomaly_id=f"gap_{i}",
                anomaly_type=AnomalyType.DATA_GAP,
                severity=SeverityLevel.MEDIUM,
                description="Test gap",
                affected_entities=[f"entity_{i}"],
                evidence={},
                confidence=0.9,
            )
            for i in range(15)  # More than 10 data gaps
        ]

        quality_metrics = DataQualityMetrics(
            completeness=0.5,  # Low completeness
            consistency=0.8,
            accuracy=0.9,
            freshness=0.7,
            validity=0.9,
            uniqueness=0.95,
        )

        recommendations = validator._generate_recommendations(
            anomalies, quality_metrics
        )

        assert len(recommendations) > 0
        assert any("data gaps" in rec.lower() for rec in recommendations)


class TestExperimentSuggestionEngine:
    """Test ExperimentSuggestionEngine functionality."""

    @pytest.fixture
    def mock_validator(self):
        """Mock validator."""
        validator = MagicMock(spec=KnowledgeGraphValidator)
        return validator

    @pytest.fixture
    def suggestion_engine(self, mock_validator):
        """Create suggestion engine instance."""
        return ExperimentSuggestionEngine(validator=mock_validator)

    def test_suggest_experiments(self, suggestion_engine):
        """Test experiment suggestion generation."""
        anomalies = [
            DetectedAnomaly(
                anomaly_id="contradiction_1",
                anomaly_type=AnomalyType.CONTRADICTION,
                severity=SeverityLevel.HIGH,
                description="Mass contradiction",
                affected_entities=["entity_1"],
                evidence={"property": "molecular_weight"},
                confidence=0.9,
                experimental_validation_needed=True,
            ),
            DetectedAnomaly(
                anomaly_id="gap_1",
                anomaly_type=AnomalyType.DATA_GAP,
                severity=SeverityLevel.MEDIUM,
                description="Missing property",
                affected_entities=["entity_2"],
                evidence={"missing_property": "logp"},
                confidence=1.0,
                experimental_validation_needed=True,
            ),
        ]

        suggestions = suggestion_engine.suggest_experiments(anomalies)

        assert len(suggestions) >= 1
        assert all("experiment_id" in suggestion for suggestion in suggestions)
        assert all("priority" in suggestion for suggestion in suggestions)
        assert all("protocol_suggestions" in suggestion for suggestion in suggestions)

    def test_calculate_experiment_priority(self, suggestion_engine):
        """Test experiment priority calculation."""
        high_severity_anomalies = [
            DetectedAnomaly(
                anomaly_id="test_1",
                anomaly_type=AnomalyType.CONTRADICTION,
                severity=SeverityLevel.CRITICAL,
                description="Test",
                affected_entities=["entity_1"],
                evidence={},
                confidence=0.9,
            )
        ]

        low_severity_anomalies = [
            DetectedAnomaly(
                anomaly_id="test_2",
                anomaly_type=AnomalyType.OUTLIER,
                severity=SeverityLevel.LOW,
                description="Test",
                affected_entities=["entity_2"],
                evidence={},
                confidence=0.7,
            )
        ]

        high_priority = suggestion_engine._calculate_experiment_priority(
            high_severity_anomalies
        )
        low_priority = suggestion_engine._calculate_experiment_priority(
            low_severity_anomalies
        )

        assert high_priority > low_priority
        assert 0 <= high_priority <= 1
        assert 0 <= low_priority <= 1

    def test_property_measurement_suggestion(self, suggestion_engine):
        """Test property measurement experiment suggestion."""
        anomalies = [
            DetectedAnomaly(
                anomaly_id="mass_contradiction",
                anomaly_type=AnomalyType.CONTRADICTION,
                severity=SeverityLevel.HIGH,
                description="Mass values don't match",
                affected_entities=["entity_1", "entity_2"],
                evidence={"property": "molecular_weight"},
                confidence=0.9,
            )
        ]

        suggestion = suggestion_engine._suggest_property_measurement_experiment(
            anomalies, 0.8
        )

        assert suggestion["type"] == "property_measurement"
        assert suggestion["priority"] == 0.8
        assert len(suggestion["affected_entities"]) == 2
        assert "mass spectrometry" in suggestion["protocol_suggestions"][0].lower()

    def test_get_measurement_protocols(self, suggestion_engine):
        """Test measurement protocol suggestions."""
        properties = {"molecular_weight", "logp", "unknown_property"}

        protocols = suggestion_engine._get_measurement_protocols(properties)

        assert len(protocols) == 3
        assert any("mass spectrometry" in protocol.lower() for protocol in protocols)
        assert any("shake-flask" in protocol.lower() for protocol in protocols)
        assert any(
            "standard analytical method" in protocol.lower() for protocol in protocols
        )


class TestValidationIntegration:
    """Integration tests for validation functionality."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        # This would be an integration test with actual database
        # For now, just test that the workflow doesn't crash

        mock_database = MagicMock()
        mock_database.execute_query.return_value = GraphQueryResult(
            records=[], summary={}, execution_time=0.1, query=GraphQuery(query="test")
        )

        mock_schema = MagicMock(spec=ICKGSchema)
        mock_schema.entity_schemas = {}

        mock_rules = MagicMock(spec=ChemicalReasoningRuleLibrary)
        mock_rules.get_all_rules.return_value = []

        validator = KnowledgeGraphValidator(
            database=mock_database, schema=mock_schema, reasoning_rules=mock_rules
        )

        # This should not raise an exception
        report = validator.validate_full_graph()

        assert isinstance(report, ValidationReport)
        assert report.report_id is not None
        assert report.generated_at is not None
        assert isinstance(report.anomalies_detected, list)
        assert 0 <= report.data_quality_score <= 1


if __name__ == "__main__":
    pytest.main([__file__])

"""
Tests for Data Validation and Quality Control System

This module contains comprehensive tests for the data validation system including
validation rules, quality metrics, and error detection capabilities.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import numpy as np
import pytest
from chemi_accel.autonomous.data_ingestion import ExperimentalDataPoint
from chemi_accel.autonomous.data_validation import (
    DataValidator,
    InstrumentSpecificValidationRule,
    QualityControlRule,
    RangeValidationRule,
    SchemaValidationRule,
    StatisticalValidationRule,
    TimestampValidationRule,
    ValidationResult,
    create_assay_validator,
    create_lcms_validator,
    create_nmr_validator,
)


class TestValidationResult:
    """Test cases for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test basic validation result creation."""
        result = ValidationResult(
            is_valid=True,
            confidence_score=0.95,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            quality_metrics={"metric1": 0.8, "metric2": 0.9},
            corrected_data={"corrected_field": "value"},
        )

        assert result.is_valid is True
        assert result.confidence_score == 0.95
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert len(result.quality_metrics) == 2
        assert result.corrected_data["corrected_field"] == "value"

    def test_validation_result_defaults(self):
        """Test validation result with default values."""
        result = ValidationResult(is_valid=False, confidence_score=0.3)

        assert result.is_valid is False
        assert result.confidence_score == 0.3
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.quality_metrics) == 0
        assert result.corrected_data is None


class TestQualityControlRule:
    """Test cases for QualityControlRule dataclass."""

    def test_quality_control_rule_creation(self):
        """Test quality control rule creation."""
        rule = QualityControlRule(
            rule_id="test_rule_001",
            rule_type="range_check",
            parameters={"min_value": 0.0, "max_value": 100.0},
            severity="error",
            description="Test range validation rule",
            is_active=True,
        )

        assert rule.rule_id == "test_rule_001"
        assert rule.rule_type == "range_check"
        assert rule.parameters["min_value"] == 0.0
        assert rule.severity == "error"
        assert rule.description == "Test range validation rule"
        assert rule.is_active is True


class TestSchemaValidationRule:
    """Test cases for SchemaValidationRule."""

    @pytest.fixture
    def sample_schema(self) -> Dict[str, Any]:
        """Create sample schema for testing."""
        return {
            "required_fields": ["data_field", "timestamp_field"],
            "field_types": {
                "data_field": "list",
                "timestamp_field": "string",
                "optional_field": "number",
            },
            "value_ranges": {"optional_field": (0.0, 100.0)},
        }

    @pytest.fixture
    def schema_rule(self, sample_schema) -> SchemaValidationRule:
        """Create schema validation rule."""
        return SchemaValidationRule(sample_schema)

    @pytest.fixture
    def valid_data_point(self) -> ExperimentalDataPoint:
        """Create valid data point for testing."""
        return ExperimentalDataPoint(
            data_id="test_123",
            experiment_id="exp_456",
            instrument_id="test_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="test",
            raw_data={
                "data_field": [1, 2, 3, 4, 5],
                "timestamp_field": "2024-01-01T12:00:00Z",
                "optional_field": 50.0,
            },
        )

    @pytest.fixture
    def invalid_data_point(self) -> ExperimentalDataPoint:
        """Create invalid data point for testing."""
        return ExperimentalDataPoint(
            data_id="test_456",
            experiment_id="exp_789",
            instrument_id="test_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="test",
            raw_data={
                "data_field": "should_be_list",  # Wrong type
                "optional_field": 150.0,  # Out of range
                # Missing required field: timestamp_field
            },
        )

    @pytest.mark.asyncio
    async def test_valid_data_validation(self, schema_rule, valid_data_point):
        """Test validation of valid data."""
        result = await schema_rule.validate(valid_data_point)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.confidence_score == 1.0

    @pytest.mark.asyncio
    async def test_invalid_data_validation(self, schema_rule, invalid_data_point):
        """Test validation of invalid data."""
        result = await schema_rule.validate(invalid_data_point)

        assert result.is_valid is False
        assert len(result.errors) >= 2  # Missing field + wrong type + out of range
        assert any("Missing required field" in error for error in result.errors)
        assert any("incorrect type" in error for error in result.errors)
        assert any("outside valid range" in error for error in result.errors)

    def test_type_checking(self, schema_rule):
        """Test type checking functionality."""
        assert schema_rule._check_type("test", "string") is True
        assert schema_rule._check_type(123, "number") is True
        assert schema_rule._check_type(123, "integer") is True
        assert schema_rule._check_type(123.5, "float") is True
        assert schema_rule._check_type([1, 2, 3], "list") is True
        assert schema_rule._check_type({"key": "value"}, "dict") is True
        assert schema_rule._check_type(True, "boolean") is True

        # Test incorrect types
        assert schema_rule._check_type("test", "number") is False
        assert schema_rule._check_type(123, "string") is False
        assert schema_rule._check_type(123.5, "integer") is False

    def test_get_rule_info(self, schema_rule, sample_schema):
        """Test rule information retrieval."""
        info = schema_rule.get_rule_info()

        assert info["rule_type"] == "schema_validation"
        assert "Validates data structure" in info["description"]
        assert info["schema"] == sample_schema


class TestRangeValidationRule:
    """Test cases for RangeValidationRule."""

    @pytest.fixture
    def range_rule(self) -> RangeValidationRule:
        """Create range validation rule."""
        return RangeValidationRule(
            {
                "temperature": (0.0, 100.0),
                "pressure": (0.0, 10.0),
                "data_points": (10, 1000),
            }
        )

    @pytest.fixture
    def valid_range_data(self) -> ExperimentalDataPoint:
        """Create data point with valid ranges."""
        return ExperimentalDataPoint(
            data_id="test_123",
            experiment_id="exp_456",
            instrument_id="test_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="test",
            raw_data={
                "temperature": 25.0,
                "pressure": 1.0,
                "data_points": [1, 2, 3, 4, 5] * 10,  # 50 points
            },
        )

    @pytest.fixture
    def invalid_range_data(self) -> ExperimentalDataPoint:
        """Create data point with invalid ranges."""
        return ExperimentalDataPoint(
            data_id="test_456",
            experiment_id="exp_789",
            instrument_id="test_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="test",
            raw_data={
                "temperature": 150.0,  # Too high
                "pressure": -1.0,  # Too low
                "data_points": [200.0] * 200,  # Many out of range values
            },
        )

    @pytest.mark.asyncio
    async def test_valid_range_validation(self, range_rule, valid_range_data):
        """Test validation of data within valid ranges."""
        result = await range_rule.validate(valid_range_data)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.quality_metrics) > 0
        assert all(score >= 0.0 for score in result.quality_metrics.values())

    @pytest.mark.asyncio
    async def test_invalid_range_validation(self, range_rule, invalid_range_data):
        """Test validation of data outside valid ranges."""
        result = await range_rule.validate(invalid_range_data)

        assert result.is_valid is False
        assert len(result.errors) >= 2  # Temperature and pressure errors
        assert any("above maximum" in error for error in result.errors)
        assert any("below minimum" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_list_range_validation(self, range_rule):
        """Test range validation for list data."""
        data_point = ExperimentalDataPoint(
            data_id="test_789",
            experiment_id="exp_123",
            instrument_id="test_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="test",
            raw_data={"data_points": [5, 15, 25, 1500, 35]},  # One value out of range
        )

        result = await range_rule.validate(data_point)

        # Should generate warning or error for out of range values
        assert len(result.errors) > 0 or len(result.warnings) > 0
        assert "data_points_range_compliance" in result.quality_metrics

    def test_get_rule_info(self, range_rule):
        """Test range rule information."""
        info = range_rule.get_rule_info()

        assert info["rule_type"] == "range_validation"
        assert "temperature" in info["field_ranges"]
        assert info["field_ranges"]["temperature"] == (0.0, 100.0)


class TestStatisticalValidationRule:
    """Test cases for StatisticalValidationRule."""

    @pytest.fixture
    def stats_rule(self) -> StatisticalValidationRule:
        """Create statistical validation rule."""
        return StatisticalValidationRule(
            {
                "signal_data": {
                    "outlier_threshold": 2.0,
                    "min_snr": 3.0,
                    "min_data_points": 10,
                    "expected_distribution": "normal",
                },
                "intensity_data": {
                    "outlier_threshold": 3.0,
                    "min_snr": 5.0,
                    "min_data_points": 5,
                },
            }
        )

    @pytest.mark.asyncio
    async def test_good_statistical_data(self, stats_rule):
        """Test validation of statistically good data."""
        # Generate normal distributed data
        np.random.seed(42)
        signal_data = np.random.normal(100, 10, 100).tolist()

        data_point = ExperimentalDataPoint(
            data_id="test_123",
            experiment_id="exp_456",
            instrument_id="test_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="test",
            raw_data={
                "signal_data": signal_data,
                "intensity_data": [100, 105, 95, 110, 90] * 4,  # 20 points
            },
        )

        result = await stats_rule.validate(data_point)

        assert result.is_valid is True
        assert len(result.quality_metrics) > 0
        assert "signal_data_outlier_score" in result.quality_metrics
        assert "signal_data_snr" in result.quality_metrics

    @pytest.mark.asyncio
    async def test_poor_statistical_data(self, stats_rule):
        """Test validation of statistically poor data."""
        # Generate data with many outliers
        signal_data = [50] * 20 + [1000] * 5 + [2000] * 3  # Many outliers

        data_point = ExperimentalDataPoint(
            data_id="test_456",
            experiment_id="exp_789",
            instrument_id="test_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="test",
            raw_data={
                "signal_data": signal_data,
                "intensity_data": [1, 2],  # Too few data points
            },
        )

        result = await stats_rule.validate(data_point)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("outliers" in error.lower() for error in result.errors)
        assert any(
            "insufficient data points" in error.lower() for error in result.errors
        )

    @pytest.mark.asyncio
    async def test_snr_calculation(self, stats_rule):
        """Test signal-to-noise ratio calculation."""
        # Create data with known SNR
        signal_level = 100
        noise_level = 10
        data = np.random.normal(signal_level, noise_level, 50).tolist()

        data_point = ExperimentalDataPoint(
            data_id="test_snr",
            experiment_id="exp_snr",
            instrument_id="test_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="test",
            raw_data={"signal_data": data},
        )

        result = await stats_rule.validate(data_point)

        assert "signal_data_snr" in result.quality_metrics
        # SNR should be approximately signal_level / noise_level = 10
        assert result.quality_metrics["signal_data_snr"] > 0.5  # Should be reasonable

    def test_get_rule_info(self, stats_rule):
        """Test statistical rule information."""
        info = stats_rule.get_rule_info()

        assert info["rule_type"] == "statistical_validation"
        assert "signal_data" in info["parameters"]
        assert info["parameters"]["signal_data"]["outlier_threshold"] == 2.0


class TestTimestampValidationRule:
    """Test cases for TimestampValidationRule."""

    @pytest.fixture
    def timestamp_rule(self) -> TimestampValidationRule:
        """Create timestamp validation rule."""
        return TimestampValidationRule(max_age_hours=24.0, future_tolerance_minutes=5.0)

    @pytest.mark.asyncio
    async def test_recent_timestamp_validation(self, timestamp_rule):
        """Test validation of recent timestamps."""
        recent_time = datetime.now(timezone.utc) - timedelta(minutes=30)

        data_point = ExperimentalDataPoint(
            data_id="test_123",
            experiment_id="exp_456",
            instrument_id="test_instrument",
            timestamp=recent_time,
            data_type="test",
            raw_data={
                "acquisition_time": recent_time.isoformat(),
                "measurement_time": (recent_time + timedelta(minutes=5)).isoformat(),
            },
        )

        result = await timestamp_rule.validate(data_point)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert "timestamp_recency" in result.quality_metrics
        assert result.quality_metrics["timestamp_recency"] > 0.8

    @pytest.mark.asyncio
    async def test_old_timestamp_validation(self, timestamp_rule):
        """Test validation of old timestamps."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)

        data_point = ExperimentalDataPoint(
            data_id="test_456",
            experiment_id="exp_789",
            instrument_id="test_instrument",
            timestamp=old_time,
            data_type="test",
            raw_data={},
        )

        result = await timestamp_rule.validate(data_point)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("too old" in error.lower() for error in result.errors)
        assert result.quality_metrics["timestamp_recency"] == 0.0

    @pytest.mark.asyncio
    async def test_future_timestamp_validation(self, timestamp_rule):
        """Test validation of future timestamps."""
        future_time = datetime.now(timezone.utc) + timedelta(minutes=30)

        data_point = ExperimentalDataPoint(
            data_id="test_789",
            experiment_id="exp_123",
            instrument_id="test_instrument",
            timestamp=future_time,
            data_type="test",
            raw_data={},
        )

        result = await timestamp_rule.validate(data_point)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("future" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_invalid_embedded_timestamps(self, timestamp_rule):
        """Test handling of invalid embedded timestamps."""
        data_point = ExperimentalDataPoint(
            data_id="test_invalid",
            experiment_id="exp_invalid",
            instrument_id="test_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="test",
            raw_data={
                "acquisition_time": "invalid_timestamp_format",
                "measurement_time": 12345,  # Invalid type
            },
        )

        result = await timestamp_rule.validate(data_point)

        assert result.is_valid is True  # Main timestamp is valid
        assert len(result.warnings) > 0
        assert any(
            "invalid timestamp format" in warning.lower() for warning in result.warnings
        )

    def test_get_rule_info(self, timestamp_rule):
        """Test timestamp rule information."""
        info = timestamp_rule.get_rule_info()

        assert info["rule_type"] == "timestamp_validation"
        assert info["max_age_hours"] == 24.0
        assert info["future_tolerance_minutes"] == 5.0


class TestInstrumentSpecificValidationRule:
    """Test cases for InstrumentSpecificValidationRule."""

    @pytest.fixture
    def instrument_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create instrument configurations."""
        return {
            "nmr": {
                "required_fields": ["chemical_shifts", "intensities"],
                "min_chemical_shift": -2.0,
                "max_chemical_shift": 12.0,
                "min_data_points": 50,
            },
            "lcms": {
                "min_chromatogram_points": 20,
                "max_retention_time": 60.0,
                "max_mass": 2000.0,
            },
            "assay": {
                "required_result_fields": ["compound_id", "result_value"],
                "allowed_plate_formats": ["96", "384"],
                "min_result_value": -100.0,
                "max_result_value": 100.0,
            },
        }

    @pytest.fixture
    def instrument_rule(self, instrument_configs) -> InstrumentSpecificValidationRule:
        """Create instrument-specific validation rule."""
        return InstrumentSpecificValidationRule(instrument_configs)

    @pytest.mark.asyncio
    async def test_valid_nmr_data(self, instrument_rule):
        """Test validation of valid NMR data."""
        data_point = ExperimentalDataPoint(
            data_id="nmr_test_123",
            experiment_id="exp_456",
            instrument_id="nmr_400",
            timestamp=datetime.now(timezone.utc),
            data_type="nmr",
            raw_data={
                "chemical_shifts": [7.5, 7.2, 3.8, 2.1, 1.5] * 10,  # 50 points
                "intensities": [100, 80, 120, 90, 75] * 10,
                "spectrum_type": "1H_NMR",
            },
        )

        result = await instrument_rule.validate(data_point)

        assert result.is_valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_invalid_nmr_data(self, instrument_rule):
        """Test validation of invalid NMR data."""
        data_point = ExperimentalDataPoint(
            data_id="nmr_test_456",
            experiment_id="exp_789",
            instrument_id="nmr_400",
            timestamp=datetime.now(timezone.utc),
            data_type="nmr",
            raw_data={
                "chemical_shifts": [15.0, -5.0, 7.2],  # Out of range values
                "intensities": [100, 80],  # Different length
                # Missing required field: intensities don't match shifts
            },
        )

        result = await instrument_rule.validate(data_point)

        assert result.is_valid is False
        assert len(result.errors) >= 2
        assert any("out of range" in error.lower() for error in result.errors)
        assert any("different lengths" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_valid_lcms_data(self, instrument_rule):
        """Test validation of valid LC-MS data."""
        chromatogram = [
            {"retention_time": i * 0.1, "intensity": 1000 + i * 10}
            for i in range(30)  # 30 points
        ]

        data_point = ExperimentalDataPoint(
            data_id="lcms_test_123",
            experiment_id="exp_456",
            instrument_id="lcms_qtof",
            timestamp=datetime.now(timezone.utc),
            data_type="lcms",
            raw_data={
                "chromatogram": chromatogram,
                "mass_spectra": [
                    {"masses": [200.1, 300.2, 400.3], "intensities": [1000, 800, 600]}
                ],
            },
        )

        result = await instrument_rule.validate(data_point)

        assert result.is_valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_invalid_lcms_data(self, instrument_rule):
        """Test validation of invalid LC-MS data."""
        data_point = ExperimentalDataPoint(
            data_id="lcms_test_456",
            experiment_id="exp_789",
            instrument_id="lcms_qtof",
            timestamp=datetime.now(timezone.utc),
            data_type="lcms",
            raw_data={
                "chromatogram": [
                    {"retention_time": 1.0, "intensity": 1000}
                ],  # Too short
                "mass_spectra": [
                    {
                        "masses": [3000.0, 4000.0],  # Exceeds max mass
                        "intensities": [1000],  # Different length
                    }
                ],
            },
        )

        result = await instrument_rule.validate(data_point)

        assert result.is_valid is False
        assert len(result.errors) >= 2
        assert any("too short" in error.lower() for error in result.errors)
        assert any("exceeds expected range" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_valid_assay_data(self, instrument_rule):
        """Test validation of valid assay data."""
        data_point = ExperimentalDataPoint(
            data_id="assay_test_123",
            experiment_id="exp_456",
            instrument_id="plate_reader",
            timestamp=datetime.now(timezone.utc),
            data_type="assay",
            raw_data={
                "results": [
                    {"compound_id": "COMP001", "result_value": 1.5, "well_id": "A1"},
                    {"compound_id": "COMP002", "result_value": -2.3, "well_id": "A2"},
                ],
                "plate_id": "PLATE_96_001",
            },
        )

        result = await instrument_rule.validate(data_point)

        assert result.is_valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_unknown_instrument_type(self, instrument_rule):
        """Test handling of unknown instrument types."""
        data_point = ExperimentalDataPoint(
            data_id="unknown_test",
            experiment_id="exp_unknown",
            instrument_id="unknown_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="unknown_type",
            raw_data={"some_data": "value"},
        )

        result = await instrument_rule.validate(data_point)

        assert result.is_valid is True  # Should pass with warning
        assert len(result.warnings) > 0
        assert any(
            "no specific validation rules" in warning.lower()
            for warning in result.warnings
        )

    def test_get_rule_info(self, instrument_rule):
        """Test instrument rule information."""
        info = instrument_rule.get_rule_info()

        assert info["rule_type"] == "instrument_specific_validation"
        assert "nmr" in info["supported_instruments"]
        assert "lcms" in info["supported_instruments"]
        assert "assay" in info["supported_instruments"]


class TestDataValidator:
    """Test cases for DataValidator main class."""

    @pytest.fixture
    def data_validator(self) -> DataValidator:
        """Create data validator instance."""
        return DataValidator()

    @pytest.fixture
    def sample_data_point(self) -> ExperimentalDataPoint:
        """Create sample data point for testing."""
        return ExperimentalDataPoint(
            data_id="test_123",
            experiment_id="exp_456",
            instrument_id="test_instrument",
            timestamp=datetime.now(timezone.utc),
            data_type="nmr",
            raw_data={
                "chemical_shifts": [7.5, 7.2, 3.8, 2.1, 1.5] * 20,  # 100 points
                "intensities": [100, 80, 120, 90, 75] * 20,
                "spectrum_type": "1H_NMR",
                "temperature": 298.0,
            },
            metadata={"solvent": "CDCl3", "frequency": 400.0},
        )

    def test_initialization(self, data_validator):
        """Test validator initialization."""
        assert len(data_validator.validation_rules) > 0
        assert data_validator.quality_thresholds["minimum_confidence"] == 0.7
        assert data_validator.quality_thresholds["error_tolerance"] == 0
        assert data_validator.quality_thresholds["warning_tolerance"] == 3

    def test_add_validation_rule(self, data_validator):
        """Test adding validation rules."""
        initial_count = len(data_validator.validation_rules)

        custom_rule = SchemaValidationRule({"required_fields": ["test_field"]})
        data_validator.add_validation_rule(custom_rule)

        assert len(data_validator.validation_rules) == initial_count + 1

    def test_remove_validation_rule(self, data_validator):
        """Test removing validation rules."""
        # Add a rule first
        custom_rule = SchemaValidationRule({"required_fields": ["test_field"]})
        data_validator.add_validation_rule(custom_rule)

        # Remove rules of specific type
        removed = data_validator.remove_validation_rule("schema_validation")

        assert removed is True

    def test_set_quality_thresholds(self, data_validator):
        """Test setting quality thresholds."""
        new_thresholds = {
            "minimum_confidence": 0.8,
            "error_tolerance": 1,
            "warning_tolerance": 5,
        }

        data_validator.set_quality_thresholds(new_thresholds)

        assert data_validator.quality_thresholds["minimum_confidence"] == 0.8
        assert data_validator.quality_thresholds["error_tolerance"] == 1
        assert data_validator.quality_thresholds["warning_tolerance"] == 5

    @pytest.mark.asyncio
    async def test_validate_data_point(self, data_validator, sample_data_point):
        """Test single data point validation."""
        result = await data_validator.validate_data_point(sample_data_point)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.quality_metrics, dict)

    @pytest.mark.asyncio
    async def test_validate_batch(self, data_validator):
        """Test batch validation."""
        data_points = []
        for i in range(5):
            data_point = ExperimentalDataPoint(
                data_id=f"test_{i}",
                experiment_id="exp_batch",
                instrument_id="test_instrument",
                timestamp=datetime.now(timezone.utc),
                data_type="nmr",
                raw_data={
                    "chemical_shifts": [7.5, 7.2, 3.8] * 30,
                    "intensities": [100, 80, 120] * 30,
                },
            )
            data_points.append(data_point)

        results = await data_validator.validate_batch(data_points)

        assert len(results) == 5
        assert all(isinstance(result, ValidationResult) for result in results)

    def test_get_validation_statistics(self, data_validator):
        """Test validation statistics."""
        stats = data_validator.get_validation_statistics()

        assert "total_rules" in stats
        assert "rule_types" in stats
        assert "quality_thresholds" in stats
        assert "rules_info" in stats
        assert stats["total_rules"] > 0
        assert len(stats["rule_types"]) > 0


class TestSpecializedValidators:
    """Test cases for specialized validator factories."""

    def test_create_nmr_validator(self):
        """Test NMR-specific validator creation."""
        validator = create_nmr_validator()

        assert isinstance(validator, DataValidator)
        assert len(validator.validation_rules) > 0

        # Should have NMR-specific rules
        rule_types = [
            rule.get_rule_info()["rule_type"] for rule in validator.validation_rules
        ]
        assert "schema_validation" in rule_types
        assert "range_validation" in rule_types
        assert "statistical_validation" in rule_types

    def test_create_lcms_validator(self):
        """Test LC-MS specific validator creation."""
        validator = create_lcms_validator()

        assert isinstance(validator, DataValidator)
        assert len(validator.validation_rules) > 0

        # Check for LC-MS specific configuration
        rule_types = [
            rule.get_rule_info()["rule_type"] for rule in validator.validation_rules
        ]
        assert "schema_validation" in rule_types
        assert "range_validation" in rule_types

    def test_create_assay_validator(self):
        """Test assay-specific validator creation."""
        validator = create_assay_validator()

        assert isinstance(validator, DataValidator)
        assert len(validator.validation_rules) > 0

        # Should have more lenient thresholds
        assert validator.quality_thresholds["minimum_confidence"] == 0.6
        assert validator.quality_thresholds["error_tolerance"] == 1
        assert validator.quality_thresholds["warning_tolerance"] == 5


class TestIntegrationScenarios:
    """Integration test scenarios for validation system."""

    @pytest.mark.asyncio
    async def test_comprehensive_nmr_validation(self):
        """Test comprehensive NMR data validation."""
        validator = create_nmr_validator()

        # Create realistic NMR data
        np.random.seed(42)
        chemical_shifts = np.linspace(0.5, 8.0, 200).tolist()
        intensities = np.random.gamma(
            2, 100, 200
        ).tolist()  # Realistic intensity distribution

        data_point = ExperimentalDataPoint(
            data_id="nmr_integration_test",
            experiment_id="exp_integration",
            instrument_id="nmr_400mhz",
            timestamp=datetime.now(timezone.utc),
            data_type="nmr",
            raw_data={
                "chemical_shifts": chemical_shifts,
                "intensities": intensities,
                "spectrum_type": "1H_NMR",
                "integration_values": [1.0, 2.0, 3.0] * 66 + [1.0, 2.0],  # 200 values
                "temperature": 298.0,
                "frequency": 400.13,
            },
            metadata={
                "solvent": "CDCl3",
                "experiment_date": datetime.now(timezone.utc).isoformat(),
                "sample_id": "SAMPLE_001",
            },
        )

        result = await validator.validate_data_point(data_point)

        # Should pass comprehensive validation
        assert result.is_valid is True
        assert result.confidence_score > 0.7
        assert len(result.quality_metrics) > 0

    @pytest.mark.asyncio
    async def test_multi_instrument_validation(self):
        """Test validation across multiple instrument types."""
        validator = DataValidator()

        # Create data points for different instruments
        nmr_data = ExperimentalDataPoint(
            data_id="nmr_multi_test",
            experiment_id="exp_multi",
            instrument_id="nmr_400",
            timestamp=datetime.now(timezone.utc),
            data_type="nmr",
            raw_data={
                "chemical_shifts": [7.5, 7.2, 3.8] * 30,
                "intensities": [100, 80, 120] * 30,
            },
        )

        lcms_data = ExperimentalDataPoint(
            data_id="lcms_multi_test",
            experiment_id="exp_multi",
            instrument_id="lcms_qtof",
            timestamp=datetime.now(timezone.utc),
            data_type="lcms",
            raw_data={
                "chromatogram": [
                    {"retention_time": i * 0.1, "intensity": 1000 + i * 10}
                    for i in range(50)
                ]
            },
        )

        assay_data = ExperimentalDataPoint(
            data_id="assay_multi_test",
            experiment_id="exp_multi",
            instrument_id="plate_reader",
            timestamp=datetime.now(timezone.utc),
            data_type="assay",
            raw_data={
                "results": [
                    {
                        "compound_id": f"COMP{i:03d}",
                        "result_value": i * 0.1,
                        "unit": "uM",
                    }
                    for i in range(96)
                ]
            },
        )

        # Validate all data points
        results = await validator.validate_batch([nmr_data, lcms_data, assay_data])

        assert len(results) == 3
        assert all(isinstance(result, ValidationResult) for result in results)

        # Each should have different validation characteristics
        confidence_scores = [result.confidence_score for result in results]
        assert all(score > 0.0 for score in confidence_scores)

    @pytest.mark.asyncio
    async def test_error_cascade_handling(self):
        """Test handling of cascading validation errors."""
        validator = DataValidator()

        # Create data with multiple types of errors
        problematic_data = ExperimentalDataPoint(
            data_id="error_cascade_test",
            experiment_id="exp_errors",
            instrument_id="problematic_instrument",
            timestamp=datetime.now(timezone.utc) - timedelta(hours=48),  # Too old
            data_type="nmr",
            raw_data={
                "chemical_shifts": "not_a_list",  # Wrong type
                "intensities": [1000] * 5,  # Too few points
                "temperature": -300.0,  # Impossible value
                "invalid_field": "should_not_be_here",
            },
        )

        result = await validator.validate_data_point(problematic_data)

        # Should accumulate multiple errors
        assert result.is_valid is False
        assert len(result.errors) >= 3  # Multiple validation failures
        assert result.confidence_score < 0.5  # Low confidence due to errors

    @pytest.mark.asyncio
    async def test_validation_performance(self):
        """Test validation performance with large datasets."""
        validator = DataValidator()

        # Create large batch of data points
        data_points = []
        for i in range(100):
            data_point = ExperimentalDataPoint(
                data_id=f"perf_test_{i}",
                experiment_id="exp_performance",
                instrument_id=f"instrument_{i % 5}",
                timestamp=datetime.now(timezone.utc),
                data_type="nmr",
                raw_data={
                    "chemical_shifts": [j * 0.1 for j in range(100)],
                    "intensities": [100 + j for j in range(100)],
                },
            )
            data_points.append(data_point)

        import time

        start_time = time.time()

        results = await validator.validate_batch(data_points)

        end_time = time.time()
        validation_time = end_time - start_time

        assert len(results) == 100
        assert validation_time < 10.0  # Should complete within 10 seconds
        assert all(isinstance(result, ValidationResult) for result in results)

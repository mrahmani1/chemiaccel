"""
Data Validation and Quality Control System

This module implements comprehensive data validation and error checking for incoming
experimental data to ensure data quality and integrity.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

# Avoid circular import - import within functions when needed


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation process."""

    is_valid: bool
    confidence_score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    corrected_data: Optional[Dict[str, Any]] = None


@dataclass
class QualityControlRule:
    """Quality control rule for data validation."""

    rule_id: str
    rule_type: str
    parameters: Dict[str, Any]
    severity: str  # "error", "warning", "info"
    description: str
    is_active: bool = True


class DataValidationRule(ABC):
    """Abstract base class for data validation rules."""

    @abstractmethod
    async def validate(self, data_point) -> ValidationResult:
        """Validate a data point against this rule."""
        pass

    @abstractmethod
    def get_rule_info(self) -> Dict[str, Any]:
        """Get information about this validation rule."""
        pass


class SchemaValidationRule(DataValidationRule):
    """Validates data against expected schema structure."""

    def __init__(self, expected_schema: Dict[str, Any]):
        self.expected_schema = expected_schema

    async def validate(self, data_point) -> ValidationResult:
        """Validate data structure against schema."""
        errors = []
        warnings = []

        # Check required fields
        required_fields = self.expected_schema.get("required_fields", [])
        for field in required_fields:
            if field not in data_point.raw_data:
                errors.append(f"Missing required field: {field}")

        # Check data types
        field_types = self.expected_schema.get("field_types", {})
        for field, expected_type in field_types.items():
            if field in data_point.raw_data:
                actual_value = data_point.raw_data[field]
                if not self._check_type(actual_value, expected_type):
                    errors.append(
                        f"Field '{field}' has incorrect type: expected {expected_type}, got {type(actual_value).__name__}"
                    )

        # Check value ranges
        value_ranges = self.expected_schema.get("value_ranges", {})
        for field, (min_val, max_val) in value_ranges.items():
            if field in data_point.raw_data:
                value = data_point.raw_data[field]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        errors.append(
                            f"Field '{field}' value {value} outside valid range [{min_val}, {max_val}]"
                        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            confidence_score=1.0 - (len(errors) * 0.2 + len(warnings) * 0.1),
            errors=errors,
            warnings=warnings,
        )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "float": float,
            "boolean": bool,
            "list": list,
            "dict": dict,
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, pass validation

        return isinstance(value, expected_python_type)

    def get_rule_info(self) -> Dict[str, Any]:
        """Get schema validation rule information."""
        return {
            "rule_type": "schema_validation",
            "description": "Validates data structure against expected schema",
            "schema": self.expected_schema,
        }


class RangeValidationRule(DataValidationRule):
    """Validates numerical values are within expected ranges."""

    def __init__(self, field_ranges: Dict[str, Tuple[float, float]]):
        self.field_ranges = field_ranges

    async def validate(self, data_point) -> ValidationResult:
        """Validate numerical ranges."""
        errors = []
        warnings = []
        quality_metrics = {}

        for field, (min_val, max_val) in self.field_ranges.items():
            if field in data_point.raw_data:
                value = data_point.raw_data[field]

                if isinstance(value, (int, float)):
                    if value < min_val:
                        errors.append(
                            f"Field '{field}' value {value} below minimum {min_val}"
                        )
                    elif value > max_val:
                        errors.append(
                            f"Field '{field}' value {value} above maximum {max_val}"
                        )
                    else:
                        # Calculate quality score based on position within range
                        range_size = max_val - min_val
                        position = (value - min_val) / range_size
                        quality_metrics[f"{field}_range_quality"] = (
                            1.0 - abs(0.5 - position) * 2
                        )

                elif isinstance(value, list):
                    # Validate list of numerical values
                    out_of_range_count = 0
                    for i, item in enumerate(value):
                        if isinstance(item, (int, float)):
                            if item < min_val or item > max_val:
                                out_of_range_count += 1

                    if out_of_range_count > 0:
                        ratio = out_of_range_count / len(value)
                        if ratio > 0.1:  # More than 10% out of range
                            errors.append(
                                f"Field '{field}' has {out_of_range_count}/{len(value)} values out of range"
                            )
                        else:
                            warnings.append(
                                f"Field '{field}' has {out_of_range_count} values out of range"
                            )

                    quality_metrics[f"{field}_range_compliance"] = 1.0 - (
                        out_of_range_count / len(value)
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            confidence_score=(
                np.mean(list(quality_metrics.values())) if quality_metrics else 0.8
            ),
            errors=errors,
            warnings=warnings,
            quality_metrics=quality_metrics,
        )

    def get_rule_info(self) -> Dict[str, Any]:
        """Get range validation rule information."""
        return {
            "rule_type": "range_validation",
            "description": "Validates numerical values are within expected ranges",
            "field_ranges": self.field_ranges,
        }


class StatisticalValidationRule(DataValidationRule):
    """Validates data using statistical analysis."""

    def __init__(self, statistical_parameters: Dict[str, Dict[str, float]]):
        self.statistical_parameters = statistical_parameters

    async def validate(self, data_point) -> ValidationResult:
        """Validate using statistical methods."""
        errors = []
        warnings = []
        quality_metrics = {}

        for field, stats_params in self.statistical_parameters.items():
            if field in data_point.raw_data:
                value = data_point.raw_data[field]

                if isinstance(value, list) and len(value) > 1:
                    data_array = np.array(value, dtype=float)

                    # Check for outliers using Z-score
                    if "outlier_threshold" in stats_params:
                        z_scores = np.abs(stats.zscore(data_array))
                        outlier_threshold = stats_params["outlier_threshold"]
                        outliers = np.where(z_scores > outlier_threshold)[0]

                        if len(outliers) > 0:
                            outlier_ratio = len(outliers) / len(data_array)
                            if outlier_ratio > 0.05:  # More than 5% outliers
                                errors.append(
                                    f"Field '{field}' has {len(outliers)} outliers ({outlier_ratio:.1%})"
                                )
                            else:
                                warnings.append(
                                    f"Field '{field}' has {len(outliers)} potential outliers"
                                )

                        quality_metrics[f"{field}_outlier_score"] = 1.0 - (
                            len(outliers) / len(data_array)
                        )

                    # Check signal-to-noise ratio
                    if "min_snr" in stats_params:
                        signal = np.mean(data_array)
                        noise = np.std(data_array)
                        snr = signal / noise if noise > 0 else float("inf")
                        min_snr = stats_params["min_snr"]

                        if snr < min_snr:
                            errors.append(
                                f"Field '{field}' has poor signal-to-noise ratio: {snr:.2f} < {min_snr}"
                            )

                        quality_metrics[f"{field}_snr"] = (
                            min(snr / min_snr, 2.0) / 2.0
                        )  # Normalize to 0-1

                    # Check for minimum data points
                    if "min_data_points" in stats_params:
                        min_points = stats_params["min_data_points"]
                        if len(data_array) < min_points:
                            errors.append(
                                f"Field '{field}' has insufficient data points: {len(data_array)} < {min_points}"
                            )

                    # Check data distribution
                    if "expected_distribution" in stats_params:
                        distribution = stats_params["expected_distribution"]
                        if distribution == "normal":
                            # Shapiro-Wilk test for normality
                            if len(data_array) >= 3:
                                _, p_value = stats.shapiro(data_array)
                                if p_value < 0.05:
                                    warnings.append(
                                        f"Field '{field}' data may not be normally distributed (p={p_value:.3f})"
                                    )
                                quality_metrics[f"{field}_normality_p"] = p_value

        return ValidationResult(
            is_valid=len(errors) == 0,
            confidence_score=(
                np.mean(list(quality_metrics.values())) if quality_metrics else 0.7
            ),
            errors=errors,
            warnings=warnings,
            quality_metrics=quality_metrics,
        )

    def get_rule_info(self) -> Dict[str, Any]:
        """Get statistical validation rule information."""
        return {
            "rule_type": "statistical_validation",
            "description": "Validates data using statistical analysis",
            "parameters": self.statistical_parameters,
        }


class TimestampValidationRule(DataValidationRule):
    """Validates timestamp-related data."""

    def __init__(
        self, max_age_hours: float = 24.0, future_tolerance_minutes: float = 5.0
    ):
        self.max_age_hours = max_age_hours
        self.future_tolerance_minutes = future_tolerance_minutes

    async def validate(self, data_point) -> ValidationResult:
        """Validate timestamp data."""
        errors = []
        warnings = []
        quality_metrics = {}

        now = datetime.now(timezone.utc)

        # Validate main timestamp
        timestamp = data_point.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        age_hours = (now - timestamp).total_seconds() / 3600

        if age_hours > self.max_age_hours:
            errors.append(
                f"Data point is too old: {age_hours:.1f} hours > {self.max_age_hours} hours"
            )

        future_minutes = (timestamp - now).total_seconds() / 60
        if future_minutes > self.future_tolerance_minutes:
            errors.append(
                f"Data point timestamp is too far in future: {future_minutes:.1f} minutes"
            )

        # Quality score based on recency
        quality_metrics["timestamp_recency"] = max(
            0.0, 1.0 - (age_hours / self.max_age_hours)
        )

        # Check for timestamp fields in raw data
        timestamp_fields = [
            "acquisition_time",
            "measurement_time",
            "start_time",
            "end_time",
        ]
        for field in timestamp_fields:
            if field in data_point.raw_data:
                try:
                    field_timestamp = datetime.fromisoformat(
                        str(data_point.raw_data[field])
                    )
                    if field_timestamp.tzinfo is None:
                        field_timestamp = field_timestamp.replace(tzinfo=timezone.utc)

                    field_age_hours = (now - field_timestamp).total_seconds() / 3600
                    if (
                        field_age_hours > self.max_age_hours * 2
                    ):  # More lenient for embedded timestamps
                        warnings.append(
                            f"Field '{field}' timestamp is very old: {field_age_hours:.1f} hours"
                        )

                except (ValueError, TypeError):
                    warnings.append(
                        f"Field '{field}' contains invalid timestamp format"
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            confidence_score=quality_metrics.get("timestamp_recency", 0.5),
            errors=errors,
            warnings=warnings,
            quality_metrics=quality_metrics,
        )

    def get_rule_info(self) -> Dict[str, Any]:
        """Get timestamp validation rule information."""
        return {
            "rule_type": "timestamp_validation",
            "description": "Validates timestamp-related data",
            "max_age_hours": self.max_age_hours,
            "future_tolerance_minutes": self.future_tolerance_minutes,
        }


class InstrumentSpecificValidationRule(DataValidationRule):
    """Validates data specific to instrument types."""

    def __init__(self, instrument_validations: Dict[str, Dict[str, Any]]):
        self.instrument_validations = instrument_validations

    async def validate(self, data_point) -> ValidationResult:
        """Validate instrument-specific data."""
        errors = []
        warnings = []
        quality_metrics = {}

        instrument_type = data_point.data_type.lower()
        validation_config = self.instrument_validations.get(instrument_type, {})

        if not validation_config:
            return ValidationResult(
                is_valid=True,
                confidence_score=0.8,
                warnings=[
                    f"No specific validation rules for instrument type: {instrument_type}"
                ],
            )

        # NMR-specific validations
        if instrument_type == "nmr":
            errors.extend(await self._validate_nmr_data(data_point, validation_config))

        # LC-MS specific validations
        elif instrument_type == "lcms":
            errors.extend(await self._validate_lcms_data(data_point, validation_config))

        # Assay-specific validations
        elif instrument_type == "assay":
            errors.extend(
                await self._validate_assay_data(data_point, validation_config)
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            confidence_score=1.0 - len(errors) * 0.15,
            errors=errors,
            warnings=warnings,
            quality_metrics=quality_metrics,
        )

    async def _validate_nmr_data(self, data_point, config: Dict[str, Any]) -> List[str]:
        """Validate NMR-specific data."""
        errors = []
        raw_data = data_point.raw_data

        # Check for required NMR fields
        required_fields = config.get(
            "required_fields", ["chemical_shifts", "intensities"]
        )
        for field in required_fields:
            if field not in raw_data:
                errors.append(f"NMR data missing required field: {field}")

        # Validate chemical shift ranges
        if "chemical_shifts" in raw_data:
            shifts = raw_data["chemical_shifts"]
            if isinstance(shifts, list):
                min_shift = config.get("min_chemical_shift", -5.0)
                max_shift = config.get("max_chemical_shift", 20.0)

                out_of_range = [
                    s
                    for s in shifts
                    if isinstance(s, (int, float)) and (s < min_shift or s > max_shift)
                ]
                if out_of_range:
                    errors.append(
                        f"NMR chemical shifts out of range: {len(out_of_range)} peaks"
                    )

        # Validate spectrum resolution
        if "intensities" in raw_data and "chemical_shifts" in raw_data:
            intensities = raw_data["intensities"]
            shifts = raw_data["chemical_shifts"]

            if len(intensities) != len(shifts):
                errors.append(
                    "NMR data: chemical_shifts and intensities arrays have different lengths"
                )

            min_points = config.get("min_data_points", 100)
            if len(intensities) < min_points:
                errors.append(
                    f"NMR spectrum has insufficient resolution: {len(intensities)} < {min_points} points"
                )

        return errors

    async def _validate_lcms_data(
        self, data_point, config: Dict[str, Any]
    ) -> List[str]:
        """Validate LC-MS specific data."""
        errors = []
        raw_data = data_point.raw_data

        # Check for chromatogram data
        if "chromatogram" in raw_data:
            chromatogram = raw_data["chromatogram"]
            if isinstance(chromatogram, list):
                min_points = config.get("min_chromatogram_points", 50)
                if len(chromatogram) < min_points:
                    errors.append(
                        f"LC-MS chromatogram too short: {len(chromatogram)} < {min_points} points"
                    )

                # Validate retention time range
                if chromatogram and "retention_time" in chromatogram[0]:
                    retention_times = [
                        point.get("retention_time", 0) for point in chromatogram
                    ]
                    max_rt = max(retention_times)
                    min_rt = min(retention_times)

                    expected_max_rt = config.get(
                        "max_retention_time", 60.0
                    )  # 60 minutes
                    if max_rt > expected_max_rt:
                        errors.append(
                            f"LC-MS retention time exceeds expected maximum: {max_rt} > {expected_max_rt}"
                        )

                    if min_rt < 0:
                        errors.append("LC-MS contains negative retention times")

        # Validate mass spectra if present
        if "mass_spectra" in raw_data:
            spectra = raw_data["mass_spectra"]
            if isinstance(spectra, list):
                for i, spectrum in enumerate(spectra):
                    if "masses" in spectrum and "intensities" in spectrum:
                        masses = spectrum["masses"]
                        intensities = spectrum["intensities"]

                        if len(masses) != len(intensities):
                            errors.append(
                                f"LC-MS spectrum {i}: masses and intensities arrays have different lengths"
                            )

                        # Check mass range
                        if masses:
                            max_mass = max(masses)
                            expected_max_mass = config.get("max_mass", 2000.0)
                            if max_mass > expected_max_mass:
                                errors.append(
                                    f"LC-MS spectrum {i}: mass exceeds expected range: {max_mass}"
                                )

        return errors

    async def _validate_assay_data(
        self, data_point, config: Dict[str, Any]
    ) -> List[str]:
        """Validate assay-specific data."""
        errors = []
        raw_data = data_point.raw_data

        # Check for assay results structure
        if "results" in raw_data:
            results = raw_data["results"]
            if isinstance(results, list):
                required_result_fields = config.get(
                    "required_result_fields", ["compound_id", "result_value"]
                )

                for i, result in enumerate(results):
                    for field in required_result_fields:
                        if field not in result:
                            errors.append(
                                f"Assay result {i} missing required field: {field}"
                            )

                    # Validate result values
                    if "result_value" in result:
                        value = result["result_value"]
                        if isinstance(value, (int, float)):
                            min_val = config.get("min_result_value", -float("inf"))
                            max_val = config.get("max_result_value", float("inf"))

                            if value < min_val or value > max_val:
                                errors.append(
                                    f"Assay result {i} value out of range: {value}"
                                )

                    # Check for control wells
                    if "well_id" in result:
                        well_id = result["well_id"]
                        if isinstance(well_id, str):
                            # Control wells typically have specific naming patterns
                            if well_id.lower() in ["blank", "control", "standard"]:
                                if "control_type" not in result:
                                    errors.append(
                                        f"Control well {well_id} missing control_type specification"
                                    )

        # Validate plate layout if present
        if "plate_id" in raw_data:
            plate_id = raw_data["plate_id"]
            expected_plate_formats = config.get(
                "allowed_plate_formats", ["96", "384", "1536"]
            )

            # Extract plate format from ID or metadata
            plate_format = None
            if "plate_format" in raw_data:
                plate_format = str(raw_data["plate_format"])
            elif isinstance(plate_id, str) and any(
                fmt in plate_id for fmt in expected_plate_formats
            ):
                plate_format = next(
                    fmt for fmt in expected_plate_formats if fmt in plate_id
                )

            if plate_format and plate_format not in expected_plate_formats:
                errors.append(f"Unsupported plate format: {plate_format}")

        return errors

    def get_rule_info(self) -> Dict[str, Any]:
        """Get instrument-specific validation rule information."""
        return {
            "rule_type": "instrument_specific_validation",
            "description": "Validates data specific to instrument types",
            "supported_instruments": list(self.instrument_validations.keys()),
        }


class DataValidator:
    """
    Main data validation system that orchestrates multiple validation rules.

    Provides comprehensive data quality checking and error detection for
    incoming experimental data.
    """

    def __init__(self):
        self.validation_rules: List[DataValidationRule] = []
        self.quality_thresholds = {
            "minimum_confidence": 0.7,
            "error_tolerance": 0,
            "warning_tolerance": 3,
        }

        # Initialize default validation rules
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize default validation rules."""
        # Schema validation for common data structures
        default_schemas = {
            "nmr": {
                "required_fields": ["chemical_shifts", "intensities"],
                "field_types": {
                    "chemical_shifts": "list",
                    "intensities": "list",
                    "spectrum_type": "string",
                },
                "value_ranges": {},
            },
            "lcms": {
                "required_fields": ["chromatogram"],
                "field_types": {"chromatogram": "list", "total_ion_current": "number"},
                "value_ranges": {"total_ion_current": (0, 1e12)},
            },
            "assay": {
                "required_fields": ["results"],
                "field_types": {"results": "list"},
                "value_ranges": {},
            },
        }

        for data_type, schema in default_schemas.items():
            self.add_validation_rule(SchemaValidationRule(schema))

        # Range validation
        common_ranges = {
            "temperature": (-273.15, 1000.0),  # Celsius
            "pressure": (0.0, 1000.0),  # atmospheres
            "ph": (0.0, 14.0),
            "concentration": (0.0, 1000.0),  # mM
        }
        self.add_validation_rule(RangeValidationRule(common_ranges))

        # Statistical validation
        statistical_params = {
            "intensities": {
                "outlier_threshold": 3.0,
                "min_snr": 3.0,
                "min_data_points": 10,
            },
            "chemical_shifts": {"outlier_threshold": 2.5, "min_data_points": 3},
        }
        self.add_validation_rule(StatisticalValidationRule(statistical_params))

        # Timestamp validation
        self.add_validation_rule(TimestampValidationRule(max_age_hours=48.0))

        # Instrument-specific validation
        instrument_configs = {
            "nmr": {
                "required_fields": ["chemical_shifts", "intensities"],
                "min_chemical_shift": -5.0,
                "max_chemical_shift": 20.0,
                "min_data_points": 100,
            },
            "lcms": {
                "min_chromatogram_points": 50,
                "max_retention_time": 120.0,
                "max_mass": 3000.0,
            },
            "assay": {
                "required_result_fields": ["compound_id", "result_value"],
                "allowed_plate_formats": ["96", "384", "1536"],
                "min_result_value": -1000.0,
                "max_result_value": 1000.0,
            },
        }
        self.add_validation_rule(InstrumentSpecificValidationRule(instrument_configs))

    def add_validation_rule(self, rule: DataValidationRule) -> None:
        """Add a validation rule to the system."""
        self.validation_rules.append(rule)
        logger.info(f"Added validation rule: {rule.get_rule_info()['rule_type']}")

    def remove_validation_rule(self, rule_type: str) -> bool:
        """Remove validation rules of a specific type."""
        initial_count = len(self.validation_rules)
        self.validation_rules = [
            rule
            for rule in self.validation_rules
            if rule.get_rule_info()["rule_type"] != rule_type
        ]
        removed_count = initial_count - len(self.validation_rules)

        if removed_count > 0:
            logger.info(f"Removed {removed_count} rules of type: {rule_type}")
            return True
        return False

    def set_quality_thresholds(self, thresholds: Dict[str, Union[int, float]]) -> None:
        """Set quality thresholds for validation."""
        self.quality_thresholds.update(thresholds)
        logger.info(f"Updated quality thresholds: {self.quality_thresholds}")

    async def validate_data_point(self, data_point) -> ValidationResult:
        """
        Validate a single data point against all rules.

        Args:
            data_point: The experimental data point to validate

        Returns:
            Comprehensive validation result
        """
        all_errors = []
        all_warnings = []
        all_quality_metrics = {}
        confidence_scores = []

        # Run all validation rules
        for rule in self.validation_rules:
            try:
                result = await rule.validate(data_point)

                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                all_quality_metrics.update(result.quality_metrics)
                confidence_scores.append(result.confidence_score)

            except Exception as e:
                logger.error(f"Validation rule error: {e}")
                all_errors.append(f"Validation rule failure: {str(e)}")

        # Calculate overall confidence score
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        # Apply quality thresholds
        is_valid = (
            len(all_errors) <= self.quality_thresholds["error_tolerance"]
            and len(all_warnings) <= self.quality_thresholds["warning_tolerance"]
            and overall_confidence >= self.quality_thresholds["minimum_confidence"]
        )

        return ValidationResult(
            is_valid=is_valid,
            confidence_score=overall_confidence,
            errors=all_errors,
            warnings=all_warnings,
            quality_metrics=all_quality_metrics,
        )

    async def validate_batch(self, data_points: List) -> List[ValidationResult]:
        """
        Validate a batch of data points.

        Args:
            data_points: List of data points to validate

        Returns:
            List of validation results
        """
        results = []

        for data_point in data_points:
            result = await self.validate_data_point(data_point)
            results.append(result)

        # Log batch statistics
        valid_count = sum(1 for r in results if r.is_valid)
        avg_confidence = np.mean([r.confidence_score for r in results])

        logger.info(
            f"Batch validation: {valid_count}/{len(data_points)} valid, "
            f"avg confidence: {avg_confidence:.3f}"
        )

        return results

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about validation rules and thresholds."""
        rule_info = [rule.get_rule_info() for rule in self.validation_rules]

        return {
            "total_rules": len(self.validation_rules),
            "rule_types": [info["rule_type"] for info in rule_info],
            "quality_thresholds": self.quality_thresholds,
            "rules_info": rule_info,
        }


# Factory functions for creating specialized validators


def create_nmr_validator() -> DataValidator:
    """Create a validator specialized for NMR data."""
    validator = DataValidator()

    # Add NMR-specific rules
    nmr_schema = {
        "required_fields": ["chemical_shifts", "intensities", "spectrum_type"],
        "field_types": {
            "chemical_shifts": "list",
            "intensities": "list",
            "spectrum_type": "string",
            "integration_values": "list",
        },
        "value_ranges": {},
    }

    nmr_ranges = {
        "chemical_shifts": (-10.0, 25.0),  # Typical NMR range
        "temperature": (243, 373),  # Kelvin, typical NMR temperature range
        "frequency": (200, 1000),  # MHz, typical NMR frequencies
    }

    nmr_statistics = {
        "chemical_shifts": {
            "outlier_threshold": 2.5,
            "min_data_points": 5,
            "expected_distribution": "normal",
        },
        "intensities": {
            "outlier_threshold": 3.0,
            "min_snr": 5.0,
            "min_data_points": 10,
        },
    }

    validator.add_validation_rule(SchemaValidationRule(nmr_schema))
    validator.add_validation_rule(RangeValidationRule(nmr_ranges))
    validator.add_validation_rule(StatisticalValidationRule(nmr_statistics))

    return validator


def create_lcms_validator() -> DataValidator:
    """Create a validator specialized for LC-MS data."""
    validator = DataValidator()

    # Add LC-MS specific rules
    lcms_schema = {
        "required_fields": ["chromatogram"],
        "field_types": {
            "chromatogram": "list",
            "mass_spectra": "list",
            "total_ion_current": "number",
        },
    }

    lcms_ranges = {
        "retention_time": (0.0, 180.0),  # minutes
        "mass": (50.0, 5000.0),  # m/z
        "intensity": (0.0, 1e10),  # arbitrary units
    }

    validator.add_validation_rule(SchemaValidationRule(lcms_schema))
    validator.add_validation_rule(RangeValidationRule(lcms_ranges))

    return validator


def create_assay_validator() -> DataValidator:
    """Create a validator specialized for assay data."""
    validator = DataValidator()

    # Add assay-specific rules
    assay_schema = {
        "required_fields": ["results"],
        "field_types": {
            "results": "list",
            "plate_id": "string",
            "assay_type": "string",
        },
    }

    assay_ranges = {
        "result_value": (-1000.0, 1000.0),  # Flexible range for assay results
        "concentration": (1e-12, 1e-3),  # Molar, typical concentration range
        "ic50": (1e-12, 1e-3),  # Molar, IC50 values
    }

    validator.add_validation_rule(SchemaValidationRule(assay_schema))
    validator.add_validation_rule(RangeValidationRule(assay_ranges))

    # Set more lenient thresholds for assay data
    validator.set_quality_thresholds(
        {"minimum_confidence": 0.6, "error_tolerance": 1, "warning_tolerance": 5}
    )

    return validator

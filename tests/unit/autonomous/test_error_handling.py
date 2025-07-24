"""
Tests for Error Handling and Recovery Mechanisms

This module contains comprehensive tests for the error handling system including
error handlers, recovery strategies, and fault tolerance mechanisms.
"""

import pytest
import asyncio
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from chemi_accel.autonomous.error_handling import (
    ErrorSeverity,
    RecoveryAction,
    ErrorRecord,
    RecoveryStrategy,
    ConnectionErrorHandler,
    DataValidationErrorHandler,
    FileSystemErrorHandler,
    InstrumentErrorHandler,
    CircuitBreakerErrorHandler,
    ErrorRecoveryManager,
    create_instrument_error_manager,
    create_validation_focused_manager
)
from chemi_accel.autonomous.data_ingestion import InstrumentConnection


class TestErrorSeverity:
    """Test cases for ErrorSeverity enum."""
    
    def test_error_severity_values(self):
        """Test error severity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestRecoveryAction:
    """Test cases for RecoveryAction enum."""
    
    def test_recovery_action_values(self):
        """Test recovery action enum values."""
        assert RecoveryAction.RETRY.value == "retry"
        assert RecoveryAction.SKIP.value == "skip"
        assert RecoveryAction.FALLBACK.value == "fallback"
        assert RecoveryAction.ALERT_ONLY.value == "alert_only"
        assert RecoveryAction.SHUTDOWN.value == "shutdown"


class TestErrorRecord:
    """Test cases for ErrorRecord dataclass."""
    
    def test_error_record_creation(self):
        """Test basic error record creation."""
        timestamp = datetime.now(timezone.utc)
        record = ErrorRecord(
            error_id="error_123",
            timestamp=timestamp,
            error_type="connection_error",
            severity=ErrorSeverity.HIGH,
            message="Connection failed to instrument",
            source="nmr_400",
            context={"host": "192.168.1.100", "port": 8080},
            stack_trace="Traceback...",
            recovery_action=RecoveryAction.RETRY,
            recovery_successful=True,
            retry_count=2
        )
        
        assert record.error_id == "error_123"
        assert record.timestamp == timestamp
        assert record.error_type == "connection_error"
        assert record.severity == ErrorSeverity.HIGH
        assert record.message == "Connection failed to instrument"
        assert record.source == "nmr_400"
        assert record.context["host"] == "192.168.1.100"
        assert record.recovery_action == RecoveryAction.RETRY
        assert record.recovery_successful is True
        assert record.retry_count == 2
    
    def test_error_record_defaults(self):
        """Test error record with default values."""
        record = ErrorRecord(
            error_id="error_456",
            timestamp=datetime.now(timezone.utc),
            error_type="generic_error",
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            source="test_source"
        )
        
        assert len(record.context) == 0
        assert record.stack_trace is None
        assert record.recovery_action is None
        assert record.recovery_successful is None
        assert record.retry_count == 0


class TestRecoveryStrategy:
    """Test cases for RecoveryStrategy dataclass."""
    
    def test_recovery_strategy_creation(self):
        """Test recovery strategy creation."""
        strategy = RecoveryStrategy(
            error_patterns=["connection", "timeout"],
            max_retries=3,
            retry_delay_seconds=5.0,
            backoff_multiplier=2.0,
            recovery_actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK],
            escalation_threshold=5,
            timeout_seconds=30.0
        )
        
        assert strategy.error_patterns == ["connection", "timeout"]
        assert strategy.max_retries == 3
        assert strategy.retry_delay_seconds == 5.0
        assert strategy.backoff_multiplier == 2.0
        assert len(strategy.recovery_actions) == 2
        assert strategy.escalation_threshold == 5
        assert strategy.timeout_seconds == 30.0


class TestConnectionErrorHandler:
    """Test cases for ConnectionErrorHandler."""
    
    @pytest.fixture
    def connection_handler(self) -> ConnectionErrorHandler:
        """Create connection error handler."""
        return ConnectionErrorHandler(max_retries=3, retry_delay=5.0)
    
    @pytest.mark.asyncio
    async def test_can_handle_connection_errors(self, connection_handler):
        """Test connection error detection."""
        # Test connection-related errors
        connection_error = ConnectionError("Connection refused")
        timeout_error = TimeoutError("Request timeout")
        refused_error = ConnectionRefusedError("Connection refused")
        
        context = {"source": "test_instrument"}
        
        assert await connection_handler.can_handle(connection_error, context) is True
        assert await connection_handler.can_handle(timeout_error, context) is True
        assert await connection_handler.can_handle(refused_error, context) is True
        
        # Test non-connection errors
        value_error = ValueError("Invalid value")
        assert await connection_handler.can_handle(value_error, context) is False
    
    @pytest.mark.asyncio
    async def test_handle_first_connection_error(self, connection_handler):
        """Test handling of first connection error."""
        error = ConnectionError("Connection failed")
        context = {"source": "test_instrument", "operation": "data_fetch"}
        
        record = await connection_handler.handle_error(error, context)
        
        assert record.error_type == "connection_error"
        assert record.severity == ErrorSeverity.MEDIUM  # First failure
        assert record.source == "test_instrument"
        assert record.recovery_action == RecoveryAction.RETRY
        assert "Connection error" in record.message
        assert record.context["consecutive_failures"] == 1
        assert record.context["total_failures"] == 1
    
    @pytest.mark.asyncio
    async def test_handle_repeated_connection_errors(self, connection_handler):
        """Test handling of repeated connection errors."""
        error = ConnectionError("Connection failed")
        context = {"source": "problematic_instrument"}
        
        # Generate multiple consecutive failures
        records = []
        for i in range(6):  # Exceed the critical threshold
            record = await connection_handler.handle_error(error, context)
            records.append(record)
        
        # First few should be medium severity
        assert records[0].severity == ErrorSeverity.MEDIUM
        assert records[1].severity == ErrorSeverity.MEDIUM
        
        # Later ones should escalate to high severity
        assert records[2].severity == ErrorSeverity.HIGH
        
        # After max retries, should be critical
        assert records[5].severity == ErrorSeverity.CRITICAL
        assert records[5].recovery_action == RecoveryAction.ALERT_ONLY
    
    @pytest.mark.asyncio
    async def test_reset_failure_count(self, connection_handler):
        """Test resetting failure count after successful recovery."""
        error = ConnectionError("Connection failed")
        context = {"source": "recovered_instrument"}
        
        # Generate failures
        await connection_handler.handle_error(error, context)
        await connection_handler.handle_error(error, context)
        
        # Reset failure count
        await connection_handler.reset_failure_count("recovered_instrument")
        
        # Next error should be treated as first failure
        record = await connection_handler.handle_error(error, context)
        assert record.context["consecutive_failures"] == 1
        assert record.severity == ErrorSeverity.MEDIUM
    
    def test_get_handler_info(self, connection_handler):
        """Test connection handler information."""
        info = connection_handler.get_handler_info()
        
        assert info["handler_type"] == "connection_error"
        assert info["max_retries"] == 3
        assert info["retry_delay"] == 5.0
        assert "tracked_sources" in info


class TestDataValidationErrorHandler:
    """Test cases for DataValidationErrorHandler."""
    
    @pytest.fixture
    def validation_handler(self) -> DataValidationErrorHandler:
        """Create data validation error handler."""
        return DataValidationErrorHandler(quarantine_threshold=0.3)
    
    @pytest.mark.asyncio
    async def test_can_handle_validation_errors(self, validation_handler):
        """Test validation error detection."""
        # Test validation-related errors
        validation_error = Exception("Validation failed for field X")
        schema_error = ValueError("Schema validation error")
        
        context = {"operation": "data_validation", "source": "test_source"}
        
        assert await validation_handler.can_handle(validation_error, context) is True
        assert await validation_handler.can_handle(schema_error, context) is True
        
        # Test non-validation errors
        connection_error = ConnectionError("Connection failed")
        context_no_data = {"operation": "connection", "source": "test_source"}
        assert await validation_handler.can_handle(connection_error, context_no_data) is False
    
    @pytest.mark.asyncio
    async def test_handle_validation_error(self, validation_handler):
        """Test handling of validation errors."""
        error = ValueError("Schema validation failed")
        context = {
            "source": "test_instrument",
            "data_type": "nmr",
            "operation": "data_validation"
        }
        
        record = await validation_handler.handle_error(error, context)
        
        assert record.error_type == "data_validation_error"
        assert record.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
        assert record.recovery_action == RecoveryAction.SKIP
        assert "Data validation failed" in record.message
        assert record.context["data_type"] == "nmr"
    
    @pytest.mark.asyncio
    async def test_quarantine_logic(self, validation_handler):
        """Test quarantine logic for high error rates."""
        error = ValueError("Repeated validation failure")
        context = {
            "source": "problematic_source",
            "data_type": "test_data",
            "operation": "data_validation"
        }
        
        # Generate many validation errors quickly
        records = []
        for i in range(20):  # Generate many errors
            record = await validation_handler.handle_error(error, context)
            records.append(record)
        
        # Later records should show quarantine
        final_record = records[-1]
        assert final_record.context["quarantined"] is True
        assert final_record.severity == ErrorSeverity.HIGH
        assert final_record.recovery_action == RecoveryAction.FALLBACK
    
    @pytest.mark.asyncio
    async def test_lift_quarantine(self, validation_handler):
        """Test lifting quarantine conditions."""
        source = "quarantine_test"
        data_type = "test_data"
        
        # Force quarantine
        validation_handler.validation_stats[f"{source}_{data_type}"] = {
            "total_errors": 10,
            "recent_errors": [],  # No recent errors
            "quarantined": True
        }
        
        # Attempt to lift quarantine
        result = await validation_handler.lift_quarantine(source, data_type)
        
        assert result is True
        assert validation_handler.validation_stats[f"{source}_{data_type}"]["quarantined"] is False
    
    def test_get_handler_info(self, validation_handler):
        """Test validation handler information."""
        info = validation_handler.get_handler_info()
        
        assert info["handler_type"] == "data_validation_error"
        assert info["quarantine_threshold"] == 0.3
        assert "tracked_sources" in info
        assert "quarantined_sources" in info


class TestFileSystemErrorHandler:
    """Test cases for FileSystemErrorHandler."""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def fs_handler(self, temp_backup_dir) -> FileSystemErrorHandler:
        """Create file system error handler with backup."""
        return FileSystemErrorHandler(backup_directory=temp_backup_dir)
    
    @pytest.mark.asyncio
    async def test_can_handle_fs_errors(self, fs_handler):
        """Test file system error detection."""
        # Test FS-related errors
        file_not_found = FileNotFoundError("File not found")
        permission_error = PermissionError("Permission denied")
        os_error = OSError("OS error")
        
        context = {"source": "file_watcher"}
        
        assert await fs_handler.can_handle(file_not_found, context) is True
        assert await fs_handler.can_handle(permission_error, context) is True
        assert await fs_handler.can_handle(os_error, context) is True
        
        # Test non-FS errors
        connection_error = ConnectionError("Connection failed")
        assert await fs_handler.can_handle(connection_error, context) is False
    
    @pytest.mark.asyncio
    async def test_handle_file_not_found(self, fs_handler):
        """Test handling of file not found errors."""
        error = FileNotFoundError("File '/path/to/file.txt' not found")
        context = {
            "source": "file_watcher",
            "file_path": "/path/to/file.txt",
            "operation": "file_read"
        }
        
        record = await fs_handler.handle_error(error, context)
        
        assert record.error_type == "file_system_error"
        assert record.severity == ErrorSeverity.MEDIUM
        assert record.recovery_action == RecoveryAction.SKIP
        assert "File system error" in record.message
        assert record.context["file_path"] == "/path/to/file.txt"
        assert record.context["error_class"] == "FileNotFoundError"
    
    @pytest.mark.asyncio
    async def test_handle_permission_error(self, fs_handler):
        """Test handling of permission errors."""
        error = PermissionError("Permission denied")
        context = {
            "source": "file_writer",
            "file_path": "/protected/file.txt"
        }
        
        record = await fs_handler.handle_error(error, context)
        
        assert record.severity == ErrorSeverity.HIGH
        assert record.recovery_action == RecoveryAction.ALERT_ONLY
        assert record.context["error_class"] == "PermissionError"
    
    @pytest.mark.asyncio
    async def test_backup_creation(self, fs_handler, temp_backup_dir):
        """Test backup file creation on error."""
        error = OSError("Disk full")
        test_data = {"experiment_data": [1, 2, 3, 4, 5], "metadata": {"test": True}}
        context = {
            "source": "data_writer",
            "file_path": "/full/disk/file.txt",
            "data": test_data
        }
        
        record = await fs_handler.handle_error(error, context)
        
        assert record.context["backup_created"] is True
        
        # Check backup file was created
        backup_dir = Path(temp_backup_dir)
        backup_files = list(backup_dir.glob("error_backup_*.json"))
        assert len(backup_files) == 1
        
        # Verify backup content
        import json
        with open(backup_files[0], 'r') as f:
            backup_data = json.load(f)
        assert backup_data["experiment_data"] == [1, 2, 3, 4, 5]
    
    def test_get_handler_info(self, fs_handler, temp_backup_dir):
        """Test file system handler information."""
        info = fs_handler.get_handler_info()
        
        assert info["handler_type"] == "file_system_error"
        assert info["backup_directory"] == temp_backup_dir
        assert info["backup_enabled"] is True


class TestInstrumentErrorHandler:
    """Test cases for InstrumentErrorHandler."""
    
    @pytest.fixture
    def instrument_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create instrument configurations."""
        return {
            "nmr_400": {
                "error_threshold": 3,
                "critical_error_patterns": ["hardware_failure", "calibration_error"],
                "maintenance_patterns": ["temperature_drift"]
            },
            "lcms_qtof": {
                "error_threshold": 5,
                "critical_error_patterns": ["pump_failure", "detector_error"],
                "maintenance_patterns": ["pressure_drift"]
            }
        }
    
    @pytest.fixture
    def instrument_handler(self, instrument_configs) -> InstrumentErrorHandler:
        """Create instrument error handler."""
        return InstrumentErrorHandler(instrument_configs)
    
    @pytest.mark.asyncio
    async def test_can_handle_instrument_errors(self, instrument_handler):
        """Test instrument error detection."""
        # Test with instrument ID in context
        context_with_instrument = {"instrument_id": "nmr_400", "source": "instrument"}
        context_without_instrument = {"source": "generic"}
        context_unknown_instrument = {"instrument_id": "unknown_device", "source": "instrument"}
        
        error = Exception("Test error")
        
        assert await instrument_handler.can_handle(error, context_with_instrument) is True
        assert await instrument_handler.can_handle(error, context_without_instrument) is False
        assert await instrument_handler.can_handle(error, context_unknown_instrument) is False
    
    @pytest.mark.asyncio
    async def test_handle_normal_instrument_error(self, instrument_handler):
        """Test handling of normal instrument errors."""
        error = Exception("Temperature sensor malfunction")
        context = {
            "instrument_id": "nmr_400",
            "source": "instrument_nmr_400",
            "operation": "measurement"
        }
        
        record = await instrument_handler.handle_error(error, context)
        
        assert record.error_type == "instrument_error"
        assert record.severity == ErrorSeverity.MEDIUM
        assert record.recovery_action == RecoveryAction.RETRY
        assert record.source == "instrument_nmr_400"
        assert record.context["instrument_id"] == "nmr_400"
        assert record.context["instrument_status"] == "degraded"
        assert record.context["error_count"] == 1
    
    @pytest.mark.asyncio
    async def test_handle_critical_instrument_error(self, instrument_handler):
        """Test handling of critical instrument errors."""
        error = Exception("Hardware failure detected in main board")
        context = {
            "instrument_id": "nmr_400",
            "source": "instrument_nmr_400"
        }
        
        record = await instrument_handler.handle_error(error, context)
        
        assert record.severity == ErrorSeverity.CRITICAL
        assert record.recovery_action == RecoveryAction.SHUTDOWN
        assert record.context["instrument_status"] == "critical"
        assert record.context["maintenance_required"] is True
        assert record.context["is_critical"] is True
    
    @pytest.mark.asyncio
    async def test_error_threshold_escalation(self, instrument_handler):
        """Test error threshold escalation."""
        error = Exception("Minor sensor error")
        context = {
            "instrument_id": "nmr_400",
            "source": "instrument_nmr_400"
        }
        
        # Generate errors up to threshold
        records = []
        for i in range(4):  # Threshold is 3, so 4th should escalate
            record = await instrument_handler.handle_error(error, context)
            records.append(record)
        
        # First few should be medium severity
        assert records[0].severity == ErrorSeverity.MEDIUM
        assert records[1].severity == ErrorSeverity.MEDIUM
        assert records[2].severity == ErrorSeverity.MEDIUM
        
        # Fourth should escalate to high
        assert records[3].severity == ErrorSeverity.HIGH
        assert records[3].recovery_action == RecoveryAction.FALLBACK
        assert records[3].context["instrument_status"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_reset_instrument_health(self, instrument_handler):
        """Test resetting instrument health status."""
        # Generate some errors first
        error = Exception("Test error")
        context = {"instrument_id": "lcms_qtof", "source": "instrument_lcms_qtof"}
        
        await instrument_handler.handle_error(error, context)
        await instrument_handler.handle_error(error, context)
        
        assert instrument_handler.instrument_health["lcms_qtof"]["error_count"] == 2
        
        # Reset health
        await instrument_handler.reset_instrument_health("lcms_qtof")
        
        health = instrument_handler.instrument_health["lcms_qtof"]
        assert health["status"] == "healthy"
        assert health["error_count"] == 0
        assert health["maintenance_required"] is False
    
    def test_get_handler_info(self, instrument_handler):
        """Test instrument handler information."""
        info = instrument_handler.get_handler_info()
        
        assert info["handler_type"] == "instrument_error"
        assert "nmr_400" in info["tracked_instruments"]
        assert "lcms_qtof" in info["tracked_instruments"]
        assert "instrument_health" in info


class TestCircuitBreakerErrorHandler:
    """Test cases for CircuitBreakerErrorHandler."""
    
    @pytest.fixture
    def circuit_breaker_handler(self) -> CircuitBreakerErrorHandler:
        """Create circuit breaker error handler."""
        return CircuitBreakerErrorHandler(failure_threshold=3, recovery_timeout=1.0)
    
    @pytest.mark.asyncio
    async def test_can_handle_all_errors(self, circuit_breaker_handler):
        """Test that circuit breaker can handle any error."""
        any_error = Exception("Any error type")
        context = {"source": "any_source"}
        
        assert await circuit_breaker_handler.can_handle(any_error, context) is True
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, circuit_breaker_handler):
        """Test circuit breaker in closed state."""
        error = Exception("Normal error")
        context = {"source": "test_source"}
        
        record = await circuit_breaker_handler.handle_error(error, context)
        
        assert record.error_type == "circuit_breaker_error"
        assert record.severity == ErrorSeverity.MEDIUM
        assert record.recovery_action == RecoveryAction.RETRY
        assert record.context["circuit_state"] == "closed"
        assert record.context["failure_count"] == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self, circuit_breaker_handler):
        """Test circuit breaker opening after threshold."""
        error = Exception("Repeated error")
        context = {"source": "failing_source"}
        
        # Generate failures up to threshold
        records = []
        for i in range(4):  # Threshold is 3
            record = await circuit_breaker_handler.handle_error(error, context)
            records.append(record)
        
        # Circuit should open on 3rd failure
        assert records[2].context["circuit_state"] == "open"
        assert records[2].severity == ErrorSeverity.HIGH
        assert records[2].recovery_action == RecoveryAction.FALLBACK
        assert records[2].context["next_attempt"] is not None
    
    @pytest.mark.asyncio
    async def test_can_attempt_when_closed(self, circuit_breaker_handler):
        """Test can_attempt when circuit is closed."""
        source = "test_source"
        
        # Initially should allow attempts
        assert await circuit_breaker_handler.can_attempt(source) is True
    
    @pytest.mark.asyncio
    async def test_can_attempt_when_open(self, circuit_breaker_handler):
        """Test can_attempt when circuit is open."""
        error = Exception("Failure")
        context = {"source": "failing_source"}
        
        # Force circuit open
        for i in range(3):
            await circuit_breaker_handler.handle_error(error, context)
        
        # Should not allow attempts when open
        assert await circuit_breaker_handler.can_attempt("failing_source") is False
    
    @pytest.mark.asyncio
    async def test_circuit_half_open_transition(self, circuit_breaker_handler):
        """Test circuit transition to half-open state."""
        error = Exception("Failure")
        context = {"source": "recovery_source"}
        
        # Force circuit open
        for i in range(3):
            await circuit_breaker_handler.handle_error(error, context)
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should transition to half-open and allow attempt
        assert await circuit_breaker_handler.can_attempt("recovery_source") is True
        
        # Check state is half-open
        breaker = circuit_breaker_handler.circuit_breakers["recovery_source"]
        assert breaker["state"] == "half_open"
    
    @pytest.mark.asyncio
    async def test_record_success_closes_circuit(self, circuit_breaker_handler):
        """Test that recording success closes the circuit."""
        error = Exception("Failure")
        context = {"source": "recovery_test"}
        
        # Force circuit open
        for i in range(3):
            await circuit_breaker_handler.handle_error(error, context)
        
        # Wait for half-open
        await asyncio.sleep(1.1)
        await circuit_breaker_handler.can_attempt("recovery_test")  # Transition to half-open
        
        # Record success
        await circuit_breaker_handler.record_success("recovery_test")
        
        # Circuit should be closed
        breaker = circuit_breaker_handler.circuit_breakers["recovery_test"]
        assert breaker["state"] == "closed"
        assert breaker["failure_count"] == 0
    
    def test_get_handler_info(self, circuit_breaker_handler):
        """Test circuit breaker handler information."""
        info = circuit_breaker_handler.get_handler_info()
        
        assert info["handler_type"] == "circuit_breaker"
        assert info["failure_threshold"] == 3
        assert info["recovery_timeout"] == 1.0
        assert "circuit_breakers" in info


class TestErrorRecoveryManager:
    """Test cases for ErrorRecoveryManager."""
    
    @pytest.fixture
    def recovery_manager(self) -> ErrorRecoveryManager:
        """Create error recovery manager."""
        return ErrorRecoveryManager()
    
    def test_initialization(self, recovery_manager):
        """Test error recovery manager initialization."""
        assert len(recovery_manager.error_handlers) > 0
        assert len(recovery_manager.recovery_strategies) > 0
        assert len(recovery_manager.error_history) == 0
        assert len(recovery_manager.alert_callbacks) == 0
        assert recovery_manager.max_history_size == 1000
    
    def test_add_error_handler(self, recovery_manager):
        """Test adding custom error handler."""
        initial_count = len(recovery_manager.error_handlers)
        
        custom_handler = Mock()
        custom_handler.get_handler_info.return_value = {"handler_type": "custom"}
        
        recovery_manager.add_error_handler(custom_handler)
        
        assert len(recovery_manager.error_handlers) == initial_count + 1
    
    def test_add_recovery_strategy(self, recovery_manager):
        """Test adding custom recovery strategy."""
        strategy = RecoveryStrategy(
            error_patterns=["custom_error"],
            max_retries=2,
            retry_delay_seconds=1.0,
            backoff_multiplier=1.5,
            recovery_actions=[RecoveryAction.RETRY],
            escalation_threshold=3
        )
        
        recovery_manager.add_recovery_strategy("custom_strategy", strategy)
        
        assert "custom_strategy" in recovery_manager.recovery_strategies
        assert recovery_manager.recovery_strategies["custom_strategy"] == strategy
    
    def test_add_alert_callback(self, recovery_manager):
        """Test adding alert callback."""
        mock_callback = Mock()
        
        recovery_manager.add_alert_callback(mock_callback)
        
        assert len(recovery_manager.alert_callbacks) == 1
        assert recovery_manager.alert_callbacks[0] == mock_callback
    
    @pytest.mark.asyncio
    async def test_handle_error_with_handler(self, recovery_manager):
        """Test error handling with appropriate handler."""
        # Mock handler
        mock_handler = Mock()
        mock_handler.can_handle = AsyncMock(return_value=True)
        mock_handler.handle_error = AsyncMock(return_value=ErrorRecord(
            error_id="handled_123",
            timestamp=datetime.now(timezone.utc),
            error_type="mock_error",
            severity=ErrorSeverity.MEDIUM,
            message="Mock error handled",
            source="test_source",
            recovery_action=RecoveryAction.RETRY
        ))
        
        recovery_manager.add_error_handler(mock_handler)
        
        # Handle error
        error = Exception("Test error")
        context = {"source": "test_source", "operation": "test"}
        
        record = await recovery_manager.handle_error(error, context, retry_count=1)
        
        assert record.error_id == "handled_123"
        assert record.error_type == "mock_error"
        assert record.retry_count == 1
        assert len(recovery_manager.error_history) == 1
        
        # Check handler was called
        mock_handler.can_handle.assert_called_once_with(error, context)
        mock_handler.handle_error.assert_called_once_with(error, context)
    
    @pytest.mark.asyncio
    async def test_handle_error_without_handler(self, recovery_manager):
        """Test error handling when no handler matches."""
        # Clear existing handlers
        recovery_manager.error_handlers = []
        
        error = Exception("Unhandled error")
        context = {"source": "unknown_source"}
        
        record = await recovery_manager.handle_error(error, context)
        
        assert record.error_type == "generic_error"
        assert record.severity == ErrorSeverity.MEDIUM
        assert record.recovery_action == RecoveryAction.ALERT_ONLY
        assert "Unhandled error" in record.message
    
    @pytest.mark.asyncio
    async def test_attempt_recovery_success(self, recovery_manager):
        """Test successful recovery attempt."""
        # Create error record
        record = ErrorRecord(
            error_id="recovery_test",
            timestamp=datetime.now(timezone.utc),
            error_type="test_error",
            severity=ErrorSeverity.MEDIUM,
            message="Test error for recovery",
            source="test_source",
            recovery_action=RecoveryAction.RETRY,
            retry_count=1
        )
        
        # Mock successful recovery function
        recovery_function = AsyncMock(return_value="success")
        
        result = await recovery_manager.attempt_recovery(record, recovery_function)
        
        assert result is True
        assert record.recovery_successful is True
        recovery_function.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_attempt_recovery_failure(self, recovery_manager):
        """Test failed recovery attempt."""
        record = ErrorRecord(
            error_id="recovery_fail_test",
            timestamp=datetime.now(timezone.utc),
            error_type="test_error",
            severity=ErrorSeverity.MEDIUM,
            message="Test error for failed recovery",
            source="test_source",
            recovery_action=RecoveryAction.RETRY,
            retry_count=0
        )
        
        # Mock failing recovery function
        recovery_function = AsyncMock(side_effect=Exception("Recovery failed"))
        
        result = await recovery_manager.attempt_recovery(record, recovery_function)
        
        assert result is False
        assert record.recovery_successful is False
        recovery_function.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_attempt_recovery_skip(self, recovery_manager):
        """Test recovery attempt with skip action."""
        record = ErrorRecord(
            error_id="recovery_skip_test",
            timestamp=datetime.now(timezone.utc),
            error_type="test_error",
            severity=ErrorSeverity.LOW,
            message="Test error for skip recovery",
            source="test_source",
            recovery_action=RecoveryAction.SKIP
        )
        
        recovery_function = AsyncMock()
        
        result = await recovery_manager.attempt_recovery(record, recovery_function)
        
        assert result is False
        recovery_function.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_attempt_recovery_max_retries(self, recovery_manager):
        """Test recovery attempt exceeding max retries."""
        record = ErrorRecord(
            error_id="max_retry_test",
            timestamp=datetime.now(timezone.utc),
            error_type="connection_error",  # Matches connection strategy
            severity=ErrorSeverity.MEDIUM,
            message="Connection failed repeatedly",
            source="test_source",
            recovery_action=RecoveryAction.RETRY,
            retry_count=5  # Exceeds max retries (3 for connection errors)
        )
        
        recovery_function = AsyncMock()
        
        result = await recovery_manager.attempt_recovery(record, recovery_function)
        
        assert result is False
        recovery_function.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_alert_sending(self, recovery_manager):
        """Test alert sending for high-severity errors."""
        mock_callback = Mock()
        recovery_manager.add_alert_callback(mock_callback)
        
        # Handle high-severity error
        error = Exception("Critical error")
        context = {"source": "critical_source"}
        
        # Mock handler to return critical error
        mock_handler = Mock()
        mock_handler.can_handle = AsyncMock(return_value=True)
        mock_handler.handle_error = AsyncMock(return_value=ErrorRecord(
            error_id="critical_123",
            timestamp=datetime.now(timezone.utc),
            error_type="critical_error",
            severity=ErrorSeverity.CRITICAL,
            message="Critical system failure",
            source="critical_source",
            recovery_action=RecoveryAction.SHUTDOWN
        ))
        
        recovery_manager.error_handlers = [mock_handler]
        
        await recovery_manager.handle_error(error, context)
        
        # Check alert callback was called
        mock_callback.assert_called_once()
    
    def test_get_error_statistics(self, recovery_manager):
        """Test error statistics calculation."""
        # Create some test error records
        now = datetime.now(timezone.utc)
        test_errors = [
            ErrorRecord("error_1", now, "type_a", ErrorSeverity.LOW, "Error 1", "source_1"),
            ErrorRecord("error_2", now, "type_a", ErrorSeverity.MEDIUM, "Error 2", "source_1"),
            ErrorRecord("error_3", now, "type_b", ErrorSeverity.HIGH, "Error 3", "source_2"),
        ]
        
        # Add to history
        recovery_manager.error_history.extend(test_errors)
        
        stats = recovery_manager.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert stats["error_types"]["type_a"] == 2
        assert stats["error_types"]["type_b"] == 1
        assert stats["severity_distribution"]["low"] == 1
        assert stats["severity_distribution"]["medium"] == 1
        assert stats["severity_distribution"]["high"] == 1
        assert stats["source_distribution"]["source_1"] == 2
        assert stats["source_distribution"]["source_2"] == 1
    
    def test_get_handler_status(self, recovery_manager):
        """Test handler status retrieval."""
        status = recovery_manager.get_handler_status()
        
        assert len(status) > 0
        assert all("handler_type" in handler_info for handler_info in status)
    
    @pytest.mark.asyncio
    async def test_health_check(self, recovery_manager):
        """Test system health check."""
        health = await recovery_manager.health_check()
        
        assert "status" in health
        assert "timestamp" in health
        assert "statistics" in health
        assert "recommendations" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy", "critical"]


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_instrument_error_manager(self):
        """Test instrument-focused error manager creation."""
        connections = [
            InstrumentConnection(
                instrument_id="nmr_400",
                instrument_type="nmr",
                connection_type="api",
                endpoint="http://nmr:8080"
            ),
            InstrumentConnection(
                instrument_id="lcms_qtof",
                instrument_type="lcms",
                connection_type="api",
                endpoint="http://lcms:8080"
            )
        ]
        
        manager = create_instrument_error_manager(connections)
        
        assert isinstance(manager, ErrorRecoveryManager)
        
        # Check for instrument-specific handler
        handler_types = [h.get_handler_info()["handler_type"] for h in manager.error_handlers]
        assert "instrument_error" in handler_types
    
    def test_create_validation_focused_manager(self):
        """Test validation-focused error manager creation."""
        manager = create_validation_focused_manager()
        
        assert isinstance(manager, ErrorRecoveryManager)
        
        # Check for validation-specific configuration
        handler_types = [h.get_handler_info()["handler_type"] for h in manager.error_handlers]
        assert "data_validation_error" in handler_types
        
        # Check for strict validation strategy
        assert "strict_validation" in manager.recovery_strategies
        strategy = manager.recovery_strategies["strict_validation"]
        assert strategy.max_retries == 2
        assert strategy.escalation_threshold == 5


class TestIntegrationScenarios:
    """Integration test scenarios for error handling system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_error_handling(self):
        """Test complete end-to-end error handling workflow."""
        manager = ErrorRecoveryManager()
        
        # Track alerts
        alerts_received = []
        def alert_callback(record):
            alerts_received.append(record)
        
        manager.add_alert_callback(alert_callback)
        
        # Simulate connection error
        error = ConnectionError("Connection timeout to instrument")
        context = {"source": "nmr_spectrometer", "operation": "data_fetch"}
        
        # Handle error
        record = await manager.handle_error(error, context)
        
        assert record.error_type == "connection_error"
        assert record.source == "nmr_spectrometer"
        assert len(manager.error_history) == 1
        
        # Attempt recovery
        mock_recovery = AsyncMock(return_value="reconnected")
        recovery_success = await manager.attempt_recovery(record, mock_recovery)
        
        assert recovery_success is True
        assert record.recovery_successful is True
    
    @pytest.mark.asyncio
    async def test_cascading_error_handling(self):
        """Test handling of cascading errors."""
        manager = ErrorRecoveryManager()
        
        # Simulate series of related errors
        connection_error = ConnectionError("Initial connection failure")
        timeout_error = TimeoutError("Subsequent timeout")
        validation_error = ValueError("Data validation failed due to incomplete data")
        
        context = {"source": "problematic_instrument", "experiment_id": "exp_123"}
        
        # Handle errors in sequence
        record1 = await manager.handle_error(connection_error, context)
        record2 = await manager.handle_error(timeout_error, context)
        record3 = await manager.handle_error(validation_error, context)
        
        # All should be handled appropriately
        assert record1.error_type == "connection_error"
        assert record2.error_type == "connection_error"  # Timeout is connection-related
        assert record3.error_type == "generic_error"  # Validation not in main context
        
        # Check error escalation
        assert len(manager.error_history) == 3
        
        # Later connection errors should have higher severity
        assert record2.context["consecutive_failures"] > record1.context["consecutive_failures"]
    
    @pytest.mark.asyncio
    async def test_recovery_strategy_selection(self):
        """Test selection of appropriate recovery strategies."""
        manager = ErrorRecoveryManager()
        
        # Test connection error recovery
        connection_error = ConnectionError("Network unreachable")
        context = {"source": "remote_instrument"}
        
        record = await manager.handle_error(connection_error, context)
        
        # Should use connection strategy
        recovery_function = AsyncMock(return_value="recovered")
        
        # First attempt should succeed quickly
        start_time = datetime.now()
        result = await manager.attempt_recovery(record, recovery_function)
        end_time = datetime.now()
        
        assert result is True
        # Should have some delay due to retry strategy
        assert (end_time - start_time).total_seconds() >= 0  # At least some time passed
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with other handlers."""
        manager = ErrorRecoveryManager()
        
        # Generate repeated failures for same source
        error = Exception("Persistent system failure")
        context = {"source": "unreliable_service"}
        
        records = []
        for i in range(6):  # Exceed circuit breaker threshold
            record = await manager.handle_error(error, context, retry_count=i)
            records.append(record)
        
        # Later records should show circuit breaker effects
        final_record = records[-1]
        assert final_record.error_type in ["circuit_breaker_error", "generic_error"]
        
        # Check circuit breaker state
        cb_handler = None
        for handler in manager.error_handlers:
            if hasattr(handler, 'circuit_breakers'):
                cb_handler = handler
                break
        
        if cb_handler:
            assert "unreliable_service" in cb_handler.circuit_breakers
            breaker = cb_handler.circuit_breakers["unreliable_service"]
            assert breaker["failure_count"] >= 3  # Should have accumulated failures
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self):
        """Test health monitoring with error patterns."""
        manager = ErrorRecoveryManager()
        
        # Generate various types of errors
        errors_and_contexts = [
            (ConnectionError("Connection 1"), {"source": "instrument_1"}),
            (ConnectionError("Connection 2"), {"source": "instrument_1"}),
            (ValueError("Validation 1"), {"source": "validator", "operation": "data_validation"}),
            (FileNotFoundError("File 1"), {"source": "file_watcher"}),
            (Exception("Critical failure"), {"source": "instrument_1"}),
        ]
        
        for error, context in errors_and_contexts:
            await manager.handle_error(error, context)
        
        # Check health status
        health = await manager.health_check()
        
        assert health["status"] in ["healthy", "degraded", "unhealthy", "critical"]
        assert health["statistics"]["total_errors"] == 5
        assert len(health["recommendations"]) >= 0
        
        # Should have recommendations for high error sources
        if health["statistics"]["recent_errors_1h"] > 10:
            assert any("error rate" in rec.lower() for rec in health["recommendations"])
    
    @pytest.mark.asyncio
    async def test_recovery_timeout_handling(self):
        """Test handling of recovery timeouts."""
        manager = ErrorRecoveryManager()
        
        # Create slow recovery function
        async def slow_recovery():
            await asyncio.sleep(2.0)  # 2 second delay
            return "slow_success"
        
        # Handle error with short timeout strategy
        error = Exception("Timeout test error")
        context = {"source": "slow_service"}
        
        record = await manager.handle_error(error, context)
        
        # Add strategy with short timeout
        manager.add_recovery_strategy("timeout_test", RecoveryStrategy(
            error_patterns=["timeout"],
            max_retries=1,
            retry_delay_seconds=0.1,
            backoff_multiplier=1.0,
            recovery_actions=[RecoveryAction.RETRY],
            escalation_threshold=1,
            timeout_seconds=0.5  # Short timeout
        ))
        
        # Recovery should timeout
        result = await manager.attempt_recovery(record, slow_recovery)
        
        # Should fail due to timeout
        assert result is False
        assert record.recovery_successful is False
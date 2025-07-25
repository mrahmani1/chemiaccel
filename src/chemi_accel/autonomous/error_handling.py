"""
Error Handling and Recovery Mechanisms

This module implements comprehensive error handling, fault tolerance, and
recovery mechanisms for the autonomous laboratory data ingestion system.
"""

import asyncio
import json
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .data_ingestion import InstrumentConnection

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Recovery action types."""

    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ALERT_ONLY = "alert_only"
    SHUTDOWN = "shutdown"


@dataclass
class ErrorRecord:
    """Represents an error occurrence."""

    error_id: str
    timestamp: datetime
    error_type: str
    severity: ErrorSeverity
    message: str
    source: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_action: Optional[RecoveryAction] = None
    recovery_successful: Optional[bool] = None
    retry_count: int = 0


@dataclass
class RecoveryStrategy:
    """Configuration for error recovery strategy."""

    error_patterns: List[str]
    max_retries: int
    retry_delay_seconds: float
    backoff_multiplier: float
    recovery_actions: List[RecoveryAction]
    escalation_threshold: int
    timeout_seconds: Optional[float] = None


class ErrorHandler(ABC):
    """Abstract base class for error handlers."""

    @abstractmethod
    async def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this handler can handle the given error."""
        pass

    @abstractmethod
    async def handle_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> ErrorRecord:
        """Handle the error and return error record."""
        pass

    @abstractmethod
    def get_handler_info(self) -> Dict[str, Any]:
        """Get information about this error handler."""
        pass


class ConnectionErrorHandler(ErrorHandler):
    """Handles connection-related errors."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 5.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_failures = {}

    async def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this is a connection error."""
        connection_error_types = [
            "ConnectionError",
            "ConnectionRefusedError",
            "TimeoutError",
            "aiohttp.ClientError",
            "websockets.exceptions.ConnectionClosed",
        ]

        error_type = type(error).__name__
        return any(err_type in error_type for err_type in connection_error_types)

    async def handle_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> ErrorRecord:
        """Handle connection errors with retry logic."""
        error_id = f"conn_error_{datetime.now().timestamp()}"
        source = context.get("source", "unknown")

        # Track connection failures for this source
        if source not in self.connection_failures:
            self.connection_failures[source] = {
                "count": 0,
                "last_failure": None,
                "consecutive_failures": 0,
            }

        failure_info = self.connection_failures[source]
        failure_info["count"] += 1
        failure_info["last_failure"] = datetime.now(timezone.utc)
        failure_info["consecutive_failures"] += 1

        # Determine severity based on failure frequency
        if failure_info["consecutive_failures"] > 5:
            severity = ErrorSeverity.CRITICAL
        elif failure_info["consecutive_failures"] > 2:
            severity = ErrorSeverity.HIGH
        else:
            severity = ErrorSeverity.MEDIUM

        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc),
            error_type="connection_error",
            severity=severity,
            message=f"Connection error: {str(error)}",
            source=source,
            context={
                "error_class": type(error).__name__,
                "consecutive_failures": failure_info["consecutive_failures"],
                "total_failures": failure_info["count"],
                **context,
            },
            stack_trace=traceback.format_exc(),
            recovery_action=RecoveryAction.RETRY
            if failure_info["consecutive_failures"] <= self.max_retries
            else RecoveryAction.ALERT_ONLY,
        )

        logger.error(
            f"Connection error in {source}: {error} (consecutive: {failure_info['consecutive_failures']})"
        )

        return error_record

    async def reset_failure_count(self, source: str) -> None:
        """Reset failure count for successful reconnection."""
        if source in self.connection_failures:
            self.connection_failures[source]["consecutive_failures"] = 0
            logger.info(f"Reset connection failure count for {source}")

    def get_handler_info(self) -> Dict[str, Any]:
        """Get handler information."""
        return {
            "handler_type": "connection_error",
            "description": "Handles connection-related errors with retry logic",
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "tracked_sources": list(self.connection_failures.keys()),
        }


class DataValidationErrorHandler(ErrorHandler):
    """Handles data validation errors."""

    def __init__(self, quarantine_threshold: float = 0.3):
        self.quarantine_threshold = quarantine_threshold
        self.validation_stats = {}

    async def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this is a validation error."""
        return (
            "validation" in str(error).lower()
            or type(error).__name__ in ["ValidationError", "SchemaError", "ValueError"]
            and "data" in context.get("operation", "").lower()
        )

    async def handle_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> ErrorRecord:
        """Handle validation errors."""
        error_id = f"validation_error_{datetime.now().timestamp()}"
        source = context.get("source", "unknown")
        data_type = context.get("data_type", "unknown")

        # Track validation errors by source and data type
        key = f"{source}_{data_type}"
        if key not in self.validation_stats:
            self.validation_stats[key] = {
                "total_errors": 0,
                "recent_errors": [],
                "quarantined": False,
            }

        stats = self.validation_stats[key]
        stats["total_errors"] += 1
        stats["recent_errors"].append(datetime.now(timezone.utc))

        # Keep only recent errors (last hour)
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        stats["recent_errors"] = [t for t in stats["recent_errors"] if t > one_hour_ago]

        # Calculate error rate
        recent_error_count = len(stats["recent_errors"])
        error_rate = (
            recent_error_count / 60.0
        )  # errors per minute (assuming steady data flow)

        # Determine severity and recovery action
        if error_rate > self.quarantine_threshold:
            severity = ErrorSeverity.HIGH
            recovery_action = RecoveryAction.FALLBACK
            stats["quarantined"] = True
            logger.warning(
                f"Quarantining data source {key} due to high validation error rate: {error_rate:.2f}/min"
            )
        else:
            severity = (
                ErrorSeverity.MEDIUM if recent_error_count > 5 else ErrorSeverity.LOW
            )
            recovery_action = RecoveryAction.SKIP

        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc),
            error_type="data_validation_error",
            severity=severity,
            message=f"Data validation failed: {str(error)}",
            source=source,
            context={
                "data_type": data_type,
                "error_rate": error_rate,
                "recent_errors": recent_error_count,
                "quarantined": stats["quarantined"],
                **context,
            },
            stack_trace=traceback.format_exc()
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
            else None,
            recovery_action=recovery_action,
        )

        return error_record

    async def lift_quarantine(self, source: str, data_type: str) -> bool:
        """Lift quarantine for a data source if conditions are met."""
        key = f"{source}_{data_type}"
        if key in self.validation_stats and self.validation_stats[key]["quarantined"]:
            # Check if error rate has decreased
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            recent_errors = [
                t
                for t in self.validation_stats[key]["recent_errors"]
                if t > one_hour_ago
            ]

            if len(recent_errors) < 5:  # Arbitrary threshold for recovery
                self.validation_stats[key]["quarantined"] = False
                logger.info(f"Lifted quarantine for {key}")
                return True

        return False

    def get_handler_info(self) -> Dict[str, Any]:
        """Get handler information."""
        return {
            "handler_type": "data_validation_error",
            "description": "Handles data validation errors with quarantine logic",
            "quarantine_threshold": self.quarantine_threshold,
            "tracked_sources": list(self.validation_stats.keys()),
            "quarantined_sources": [
                k for k, v in self.validation_stats.items() if v["quarantined"]
            ],
        }


class FileSystemErrorHandler(ErrorHandler):
    """Handles file system related errors."""

    def __init__(self, backup_directory: Optional[str] = None):
        self.backup_directory = Path(backup_directory) if backup_directory else None
        if self.backup_directory:
            self.backup_directory.mkdir(parents=True, exist_ok=True)

    async def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this is a file system error."""
        fs_error_types = [
            "FileNotFoundError",
            "PermissionError",
            "OSError",
            "IOError",
            "IsADirectoryError",
            "NotADirectoryError",
        ]

        return type(error).__name__ in fs_error_types

    async def handle_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> ErrorRecord:
        """Handle file system errors."""
        error_id = f"fs_error_{datetime.now().timestamp()}"
        source = context.get("source", "unknown")
        file_path = context.get("file_path", "unknown")

        # Determine severity based on error type
        error_type = type(error).__name__
        if error_type in ["PermissionError", "OSError"]:
            severity = ErrorSeverity.HIGH
            recovery_action = RecoveryAction.ALERT_ONLY
        elif error_type == "FileNotFoundError":
            severity = ErrorSeverity.MEDIUM
            recovery_action = RecoveryAction.SKIP
        else:
            severity = ErrorSeverity.LOW
            recovery_action = RecoveryAction.RETRY

        # Attempt to create backup if possible
        backup_created = False
        if self.backup_directory and "data" in context:
            try:
                backup_file = self.backup_directory / f"error_backup_{error_id}.json"
                with open(backup_file, "w") as f:
                    json.dump(context["data"], f, indent=2, default=str)
                backup_created = True
                logger.info(f"Created error backup: {backup_file}")
            except Exception as backup_error:
                logger.error(f"Failed to create backup: {backup_error}")

        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc),
            error_type="file_system_error",
            severity=severity,
            message=f"File system error: {str(error)}",
            source=source,
            context={
                "file_path": file_path,
                "error_class": error_type,
                "backup_created": backup_created,
                **context,
            },
            stack_trace=traceback.format_exc(),
            recovery_action=recovery_action,
        )

        return error_record

    def get_handler_info(self) -> Dict[str, Any]:
        """Get handler information."""
        return {
            "handler_type": "file_system_error",
            "description": "Handles file system related errors with backup capability",
            "backup_directory": str(self.backup_directory)
            if self.backup_directory
            else None,
            "backup_enabled": self.backup_directory is not None,
        }


class InstrumentErrorHandler(ErrorHandler):
    """Handles instrument-specific errors."""

    def __init__(self, instrument_configs: Dict[str, Dict[str, Any]]):
        self.instrument_configs = instrument_configs
        self.instrument_health = {}

    async def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this is an instrument-specific error."""
        instrument_id = context.get("instrument_id")
        return instrument_id is not None and instrument_id in self.instrument_configs

    async def handle_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> ErrorRecord:
        """Handle instrument-specific errors."""
        error_id = f"instrument_error_{datetime.now().timestamp()}"
        instrument_id = context.get("instrument_id", "unknown")

        # Initialize health tracking for this instrument
        if instrument_id not in self.instrument_health:
            self.instrument_health[instrument_id] = {
                "status": "healthy",
                "error_count": 0,
                "last_error": None,
                "maintenance_required": False,
            }

        health = self.instrument_health[instrument_id]
        health["error_count"] += 1
        health["last_error"] = datetime.now(timezone.utc)

        # Get instrument-specific configuration
        config = self.instrument_configs.get(instrument_id, {})
        error_threshold = config.get("error_threshold", 5)
        critical_errors = config.get("critical_error_patterns", [])

        # Check if this is a critical error pattern
        is_critical = any(pattern in str(error).lower() for pattern in critical_errors)

        # Determine severity and health status
        if is_critical:
            severity = ErrorSeverity.CRITICAL
            health["status"] = "critical"
            health["maintenance_required"] = True
            recovery_action = RecoveryAction.SHUTDOWN
        elif health["error_count"] >= error_threshold:
            severity = ErrorSeverity.HIGH
            health["status"] = "unhealthy"
            recovery_action = RecoveryAction.FALLBACK
        else:
            severity = ErrorSeverity.MEDIUM
            health["status"] = "degraded"
            recovery_action = RecoveryAction.RETRY

        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc),
            error_type="instrument_error",
            severity=severity,
            message=f"Instrument {instrument_id} error: {str(error)}",
            source=f"instrument_{instrument_id}",
            context={
                "instrument_id": instrument_id,
                "instrument_status": health["status"],
                "error_count": health["error_count"],
                "maintenance_required": health["maintenance_required"],
                "is_critical": is_critical,
                **context,
            },
            stack_trace=traceback.format_exc()
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
            else None,
            recovery_action=recovery_action,
        )

        logger.error(
            f"Instrument {instrument_id} error (count: {health['error_count']}): {error}"
        )

        return error_record

    async def reset_instrument_health(self, instrument_id: str) -> None:
        """Reset health status after successful maintenance."""
        if instrument_id in self.instrument_health:
            self.instrument_health[instrument_id].update(
                {"status": "healthy", "error_count": 0, "maintenance_required": False}
            )
            logger.info(f"Reset health status for instrument {instrument_id}")

    def get_handler_info(self) -> Dict[str, Any]:
        """Get handler information."""
        return {
            "handler_type": "instrument_error",
            "description": "Handles instrument-specific errors with health tracking",
            "tracked_instruments": list(self.instrument_configs.keys()),
            "instrument_health": self.instrument_health,
        }


class CircuitBreakerErrorHandler(ErrorHandler):
    """Implements circuit breaker pattern for error handling."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.circuit_breakers = {}

    async def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Always return True as circuit breaker is a general pattern."""
        return True

    async def handle_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> ErrorRecord:
        """Handle errors using circuit breaker pattern."""
        error_id = f"circuit_breaker_{datetime.now().timestamp()}"
        source = context.get("source", "unknown")

        # Initialize circuit breaker for this source
        if source not in self.circuit_breakers:
            self.circuit_breakers[source] = {
                "state": "closed",  # closed, open, half_open
                "failure_count": 0,
                "last_failure": None,
                "last_success": None,
                "next_attempt": None,
            }

        breaker = self.circuit_breakers[source]
        breaker["failure_count"] += 1
        breaker["last_failure"] = datetime.now(timezone.utc)

        # Determine circuit breaker state
        if breaker["failure_count"] >= self.failure_threshold:
            breaker["state"] = "open"
            breaker["next_attempt"] = datetime.now(timezone.utc) + timedelta(
                seconds=self.recovery_timeout
            )
            severity = ErrorSeverity.HIGH
            recovery_action = RecoveryAction.FALLBACK
            logger.warning(
                f"Circuit breaker opened for {source} (failures: {breaker['failure_count']})"
            )
        else:
            severity = ErrorSeverity.MEDIUM
            recovery_action = RecoveryAction.RETRY

        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc),
            error_type="circuit_breaker_error",
            severity=severity,
            message=f"Circuit breaker handling error: {str(error)}",
            source=source,
            context={
                "circuit_state": breaker["state"],
                "failure_count": breaker["failure_count"],
                "next_attempt": breaker["next_attempt"].isoformat()
                if breaker["next_attempt"]
                else None,
                **context,
            },
            recovery_action=recovery_action,
        )

        return error_record

    async def record_success(self, source: str) -> None:
        """Record successful operation for circuit breaker."""
        if source in self.circuit_breakers:
            breaker = self.circuit_breakers[source]
            breaker["failure_count"] = 0
            breaker["last_success"] = datetime.now(timezone.utc)

            if breaker["state"] == "half_open":
                breaker["state"] = "closed"
                logger.info(f"Circuit breaker closed for {source}")
            elif breaker["state"] == "open":
                breaker["state"] = "half_open"
                logger.info(f"Circuit breaker half-open for {source}")

    async def can_attempt(self, source: str) -> bool:
        """Check if operation can be attempted based on circuit breaker state."""
        if source not in self.circuit_breakers:
            return True

        breaker = self.circuit_breakers[source]

        if breaker["state"] == "closed":
            return True
        elif breaker["state"] == "open":
            if (
                breaker["next_attempt"]
                and datetime.now(timezone.utc) >= breaker["next_attempt"]
            ):
                breaker["state"] = "half_open"
                logger.info(f"Circuit breaker transitioning to half-open for {source}")
                return True
            return False
        elif breaker["state"] == "half_open":
            return True

        return False

    def get_handler_info(self) -> Dict[str, Any]:
        """Get handler information."""
        return {
            "handler_type": "circuit_breaker",
            "description": "Implements circuit breaker pattern for error handling",
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "circuit_breakers": {
                k: {
                    **v,
                    "next_attempt": v["next_attempt"].isoformat()
                    if v["next_attempt"]
                    else None,
                }
                for k, v in self.circuit_breakers.items()
            },
        }


class ErrorRecoveryManager:
    """
    Main manager for error handling and recovery operations.

    Coordinates multiple error handlers and implements recovery strategies.
    """

    def __init__(self):
        self.error_handlers: List[ErrorHandler] = []
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.error_history: List[ErrorRecord] = []
        self.alert_callbacks: List[Callable[[ErrorRecord], None]] = []
        self.max_history_size = 1000

        # Initialize default handlers
        self._initialize_default_handlers()
        self._initialize_default_strategies()

    def _initialize_default_handlers(self) -> None:
        """Initialize default error handlers."""
        self.add_error_handler(ConnectionErrorHandler())
        self.add_error_handler(DataValidationErrorHandler())
        self.add_error_handler(FileSystemErrorHandler())
        self.add_error_handler(CircuitBreakerErrorHandler())

    def _initialize_default_strategies(self) -> None:
        """Initialize default recovery strategies."""
        # Connection error strategy
        self.recovery_strategies["connection_error"] = RecoveryStrategy(
            error_patterns=["connection", "timeout", "refused"],
            max_retries=3,
            retry_delay_seconds=5.0,
            backoff_multiplier=2.0,
            recovery_actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK],
            escalation_threshold=5,
            timeout_seconds=30.0,
        )

        # Validation error strategy
        self.recovery_strategies["validation_error"] = RecoveryStrategy(
            error_patterns=["validation", "schema", "format"],
            max_retries=1,
            retry_delay_seconds=1.0,
            backoff_multiplier=1.0,
            recovery_actions=[RecoveryAction.SKIP, RecoveryAction.ALERT_ONLY],
            escalation_threshold=10,
            timeout_seconds=10.0,
        )

        # Critical error strategy
        self.recovery_strategies["critical_error"] = RecoveryStrategy(
            error_patterns=["critical", "fatal", "shutdown"],
            max_retries=0,
            retry_delay_seconds=0.0,
            backoff_multiplier=1.0,
            recovery_actions=[RecoveryAction.ALERT_ONLY, RecoveryAction.SHUTDOWN],
            escalation_threshold=1,
            timeout_seconds=5.0,
        )

    def add_error_handler(self, handler: ErrorHandler) -> None:
        """Add an error handler."""
        self.error_handlers.append(handler)
        logger.info(
            f"Added error handler: {handler.get_handler_info()['handler_type']}"
        )

    def add_recovery_strategy(self, name: str, strategy: RecoveryStrategy) -> None:
        """Add a recovery strategy."""
        self.recovery_strategies[name] = strategy
        logger.info(f"Added recovery strategy: {name}")

    def add_alert_callback(self, callback: Callable[[ErrorRecord], None]) -> None:
        """Add callback for error alerts."""
        self.alert_callbacks.append(callback)

    async def handle_error(
        self, error: Exception, context: Dict[str, Any], retry_count: int = 0
    ) -> ErrorRecord:
        """
        Handle an error using appropriate handlers and recovery strategies.

        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            retry_count: Current retry attempt count

        Returns:
            ErrorRecord with handling results
        """
        error_record = None

        # Find appropriate handler
        for handler in self.error_handlers:
            try:
                if await handler.can_handle(error, context):
                    error_record = await handler.handle_error(error, context)
                    error_record.retry_count = retry_count
                    break
            except Exception as handler_error:
                logger.error(
                    f"Error in handler {handler.__class__.__name__}: {handler_error}"
                )

        # Create generic error record if no handler found
        if error_record is None:
            error_record = ErrorRecord(
                error_id=f"generic_error_{datetime.now().timestamp()}",
                timestamp=datetime.now(timezone.utc),
                error_type="generic_error",
                severity=ErrorSeverity.MEDIUM,
                message=f"Unhandled error: {str(error)}",
                source=context.get("source", "unknown"),
                context=context,
                stack_trace=traceback.format_exc(),
                recovery_action=RecoveryAction.ALERT_ONLY,
                retry_count=retry_count,
            )

        # Add to error history
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size :]

        # Send alerts if necessary
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await self._send_alerts(error_record)

        logger.error(f"Handled error {error_record.error_id}: {error_record.message}")

        return error_record

    async def attempt_recovery(
        self, error_record: ErrorRecord, recovery_function: Callable[[], Any]
    ) -> bool:
        """
        Attempt recovery based on error record and recovery strategy.

        Args:
            error_record: The error record to recover from
            recovery_function: Function to retry

        Returns:
            True if recovery was successful, False otherwise
        """
        if error_record.recovery_action == RecoveryAction.SKIP:
            logger.info(f"Skipping recovery for error {error_record.error_id}")
            return False

        if error_record.recovery_action == RecoveryAction.ALERT_ONLY:
            logger.info(f"Alert-only recovery for error {error_record.error_id}")
            return False

        if error_record.recovery_action == RecoveryAction.SHUTDOWN:
            logger.critical(f"Shutdown required for error {error_record.error_id}")
            return False

        # Find appropriate recovery strategy
        strategy = None
        for strategy_name, strat in self.recovery_strategies.items():
            if any(
                pattern in error_record.message.lower()
                for pattern in strat.error_patterns
            ):
                strategy = strat
                break

        if strategy is None:
            strategy = self.recovery_strategies.get(
                "generic",
                RecoveryStrategy(
                    error_patterns=[],
                    max_retries=1,
                    retry_delay_seconds=1.0,
                    backoff_multiplier=1.0,
                    recovery_actions=[RecoveryAction.RETRY],
                    escalation_threshold=3,
                ),
            )

        # Attempt recovery with retry logic
        if error_record.retry_count >= strategy.max_retries:
            logger.warning(f"Max retries exceeded for error {error_record.error_id}")
            return False

        # Calculate delay with backoff
        delay = strategy.retry_delay_seconds * (
            strategy.backoff_multiplier**error_record.retry_count
        )

        logger.info(
            f"Attempting recovery for {error_record.error_id} after {delay}s delay (attempt {error_record.retry_count + 1})"
        )

        await asyncio.sleep(delay)

        try:
            # Set timeout if specified
            if strategy.timeout_seconds:
                result = await asyncio.wait_for(
                    recovery_function(), timeout=strategy.timeout_seconds
                )
            else:
                result = await recovery_function()

            error_record.recovery_successful = True
            logger.info(f"Recovery successful for error {error_record.error_id}")
            return True

        except Exception as recovery_error:
            error_record.recovery_successful = False
            logger.error(
                f"Recovery failed for error {error_record.error_id}: {recovery_error}"
            )

            # Handle the recovery error
            await self.handle_error(
                recovery_error,
                {**error_record.context, "recovery_attempt": True},
                error_record.retry_count + 1,
            )

            return False

    async def _send_alerts(self, error_record: ErrorRecord) -> None:
        """Send alerts for high-severity errors."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_record)
                else:
                    callback(error_record)
            except Exception as alert_error:
                logger.error(f"Error in alert callback: {alert_error}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and health metrics."""
        if not self.error_history:
            return {"total_errors": 0}

        # Calculate statistics
        total_errors = len(self.error_history)
        error_types = {}
        severity_counts = {}
        source_counts = {}
        recovery_success_rate = 0

        for error in self.error_history:
            # Count by type
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1

            # Count by severity
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Count by source
            source_counts[error.source] = source_counts.get(error.source, 0) + 1

            # Recovery success rate
            if error.recovery_successful is not None:
                recovery_success_rate += 1 if error.recovery_successful else 0

        # Calculate rates
        recent_errors = [
            e
            for e in self.error_history
            if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 3600
        ]

        return {
            "total_errors": total_errors,
            "recent_errors_1h": len(recent_errors),
            "error_rate_per_hour": len(recent_errors),
            "error_types": error_types,
            "severity_distribution": severity_counts,
            "source_distribution": source_counts,
            "recovery_success_rate": recovery_success_rate
            / max(
                1,
                len(
                    [e for e in self.error_history if e.recovery_successful is not None]
                ),
            ),
            "handlers_count": len(self.error_handlers),
            "strategies_count": len(self.recovery_strategies),
        }

    def get_handler_status(self) -> List[Dict[str, Any]]:
        """Get status of all error handlers."""
        return [handler.get_handler_info() for handler in self.error_handlers]

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check based on error patterns."""
        stats = self.get_error_statistics()

        # Determine overall health
        recent_critical = len(
            [
                e
                for e in self.error_history[-100:]
                if e.severity == ErrorSeverity.CRITICAL
            ]
        )

        recent_error_rate = stats["recent_errors_1h"]

        if recent_critical > 0:
            health_status = "critical"
        elif recent_error_rate > 50:
            health_status = "unhealthy"
        elif recent_error_rate > 20:
            health_status = "degraded"
        else:
            health_status = "healthy"

        return {
            "status": health_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": stats,
            "recommendations": await self._generate_recommendations(stats),
        }

    async def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []

        if stats["recent_errors_1h"] > 30:
            recommendations.append(
                "High error rate detected. Consider investigating root causes."
            )

        if stats["recovery_success_rate"] < 0.5:
            recommendations.append(
                "Low recovery success rate. Review recovery strategies."
            )

        # Check for dominant error types
        if stats["error_types"]:
            max_error_type = max(stats["error_types"], key=stats["error_types"].get)
            if stats["error_types"][max_error_type] > stats["total_errors"] * 0.6:
                recommendations.append(
                    f"Dominant error type: {max_error_type}. Focus on resolving this pattern."
                )

        # Check for problematic sources
        if stats["source_distribution"]:
            max_source = max(
                stats["source_distribution"], key=stats["source_distribution"].get
            )
            if stats["source_distribution"][max_source] > stats["total_errors"] * 0.4:
                recommendations.append(
                    f"High error rate from source: {max_source}. Check this component."
                )

        return recommendations


# Factory functions


def create_instrument_error_manager(
    instrument_connections: List[InstrumentConnection],
) -> ErrorRecoveryManager:
    """Create error manager with instrument-specific handlers."""
    manager = ErrorRecoveryManager()

    # Create instrument configurations
    instrument_configs = {}
    for conn in instrument_connections:
        instrument_configs[conn.instrument_id] = {
            "error_threshold": 5,
            "critical_error_patterns": [
                "hardware_failure",
                "calibration_error",
                "sensor_fault",
            ],
            "maintenance_required_patterns": ["temperature_drift", "pressure_anomaly"],
        }

    # Add instrument-specific handler
    manager.add_error_handler(InstrumentErrorHandler(instrument_configs))

    return manager


def create_validation_focused_manager() -> ErrorRecoveryManager:
    """Create error manager focused on data validation."""
    manager = ErrorRecoveryManager()

    # Add specialized validation handler with stricter thresholds
    validation_handler = DataValidationErrorHandler(quarantine_threshold=0.2)
    manager.add_error_handler(validation_handler)

    # Add focused recovery strategy
    manager.add_recovery_strategy(
        "strict_validation",
        RecoveryStrategy(
            error_patterns=["validation", "schema", "quality"],
            max_retries=2,
            retry_delay_seconds=0.5,
            backoff_multiplier=1.5,
            recovery_actions=[RecoveryAction.SKIP, RecoveryAction.FALLBACK],
            escalation_threshold=5,
            timeout_seconds=15.0,
        ),
    )

    return manager

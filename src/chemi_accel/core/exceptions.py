"""
Exception classes for ChemiAccel.

This module defines the exception hierarchy for robust error handling
throughout the ChemiAccel library.
"""

from typing import Any, List, Optional


class ChemiAccelError(Exception):
    """Base exception for all ChemiAccel errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        """Initialize ChemiAccel exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class BackendUnavailableError(ChemiAccelError):
    """Raised when requested backend is not available."""

    def __init__(
        self, backend: str, reason: str, available_backends: Optional[List[str]] = None
    ):
        """Initialize backend unavailable error.

        Args:
            backend: Name of the unavailable backend
            reason: Reason why backend is unavailable
            available_backends: List of available backends
        """
        self.backend = backend
        self.reason = reason
        self.available_backends = available_backends or []

        message = f"Backend '{backend}' is not available: {reason}"
        if self.available_backends:
            message += f". Available backends: {', '.join(self.available_backends)}"

        super().__init__(
            message,
            {
                "backend": backend,
                "reason": reason,
                "available_backends": self.available_backends,
            },
        )


class GPUMemoryError(ChemiAccelError):
    """Raised when GPU memory is insufficient."""

    def __init__(
        self, required_memory: int, available_memory: int, operation: str = ""
    ):
        """Initialize GPU memory error.

        Args:
            required_memory: Required memory in bytes
            available_memory: Available memory in bytes
            operation: Description of the operation that failed
        """
        self.required_memory = required_memory
        self.available_memory = available_memory
        self.operation = operation

        required_mb = required_memory / (1024 * 1024)
        available_mb = available_memory / (1024 * 1024)

        message = f"Insufficient GPU memory: need {required_mb:.1f}MB, have {available_mb:.1f}MB"
        if operation:
            message = f"{operation}: {message}"

        super().__init__(
            message,
            {
                "required_memory": required_memory,
                "available_memory": available_memory,
                "operation": operation,
            },
        )


class InvalidMoleculeError(ChemiAccelError):
    """Raised for invalid molecular structures."""

    def __init__(
        self, molecule_id: Any, reason: str, suggestions: Optional[List[str]] = None
    ):
        """Initialize invalid molecule error.

        Args:
            molecule_id: Identifier for the invalid molecule (SMILES, index, etc.)
            reason: Reason why molecule is invalid
            suggestions: Optional list of suggestions to fix the issue
        """
        self.molecule_id = molecule_id
        self.reason = reason
        self.suggestions = suggestions or []

        message = f"Invalid molecule '{molecule_id}': {reason}"
        if self.suggestions:
            message += f". Suggestions: {'; '.join(self.suggestions)}"

        super().__init__(
            message,
            {
                "molecule_id": molecule_id,
                "reason": reason,
                "suggestions": self.suggestions,
            },
        )


class DescriptorCalculationError(ChemiAccelError):
    """Raised when descriptor calculation fails."""

    def __init__(self, descriptor: str, molecule_id: Any, reason: str):
        """Initialize descriptor calculation error.

        Args:
            descriptor: Name of the descriptor that failed
            molecule_id: Identifier for the molecule
            reason: Reason for calculation failure
        """
        self.descriptor = descriptor
        self.molecule_id = molecule_id
        self.reason = reason

        message = f"Failed to calculate descriptor '{descriptor}' for molecule '{molecule_id}': {reason}"

        super().__init__(
            message,
            {"descriptor": descriptor, "molecule_id": molecule_id, "reason": reason},
        )


class ConfigurationError(ChemiAccelError):
    """Raised for configuration-related errors."""

    def __init__(self, parameter: str, value: Any, reason: str):
        """Initialize configuration error.

        Args:
            parameter: Name of the configuration parameter
            value: Invalid value
            reason: Reason why value is invalid
        """
        self.parameter = parameter
        self.value = value
        self.reason = reason

        message = f"Invalid configuration for '{parameter}' = '{value}': {reason}"

        super().__init__(
            message, {"parameter": parameter, "value": value, "reason": reason}
        )


class ValidationError(ChemiAccelError):
    """Raised for input validation errors."""

    def __init__(
        self, input_name: str, input_value: Any, expected_type: str, reason: str = ""
    ):
        """Initialize validation error.

        Args:
            input_name: Name of the input parameter
            input_value: Invalid input value
            expected_type: Expected type or format
            reason: Additional reason for validation failure
        """
        self.input_name = input_name
        self.input_value = input_value
        self.expected_type = expected_type
        self.reason = reason

        message = f"Invalid input '{input_name}': expected {expected_type}, got {type(input_value).__name__}"
        if reason:
            message += f" ({reason})"

        super().__init__(
            message,
            {
                "input_name": input_name,
                "input_value": input_value,
                "expected_type": expected_type,
                "reason": reason,
            },
        )


class DistributedComputingError(ChemiAccelError):
    """Raised when distributed computing operations fail."""

    def __init__(self, operation: str, reason: str, node_info: Optional[dict] = None):
        """Initialize distributed computing error.

        Args:
            operation: Description of the failed operation
            reason: Reason for failure
            node_info: Optional information about the node/cluster
        """
        self.operation = operation
        self.reason = reason
        self.node_info = node_info or {}

        message = f"Distributed computing error in {operation}: {reason}"
        if node_info:
            message += f" (Node info: {node_info})"

        super().__init__(
            message, {"operation": operation, "reason": reason, "node_info": node_info}
        )


# Convenience function for error handling
def handle_graceful_degradation(error: Exception, fallback_action: str) -> str:
    """Handle graceful degradation with informative logging.

    Args:
        error: The original error that occurred
        fallback_action: Description of the fallback action being taken

    Returns:
        Formatted warning message for logging
    """
    return f"Warning: {error}. Falling back to {fallback_action}."

"""
Backend selection and management for ChemiAccel.

This module handles automatic detection and selection of the optimal
backend based on available hardware and user preferences.
"""

from typing import List, Optional, Dict, Any
from enum import Enum

from ..utils.config import BackendType, get_config
from ..utils.logging import get_logger
from .exceptions import BackendUnavailableError


logger = get_logger()


def get_available_backends() -> List[BackendType]:
    """Detect and return list of available backends.
    
    Returns:
        List of available backend types
    """
    available = [BackendType.CPU]  # CPU is always available
    
    # Check for CUDA availability
    try:
        import cupy
        # Test CUDA device access
        cupy.cuda.Device(0).use()
        available.append(BackendType.CUDA)
        logger.debug("CUDA backend available")
    except (ImportError, Exception):
        logger.debug("CUDA backend not available")
    
    # Check for JAX availability
    try:
        import jax
        # Test JAX device access
        jax.devices()
        available.append(BackendType.JAX)
        logger.debug("JAX backend available")
    except (ImportError, Exception):
        logger.debug("JAX backend not available")
    
    return available


def select_optimal_backend(preferred: Optional[BackendType] = None) -> BackendType:
    """Select the optimal backend based on availability and preferences.
    
    Args:
        preferred: Preferred backend type, if any
        
    Returns:
        Selected backend type
        
    Raises:
        BackendUnavailableError: If preferred backend is not available
    """
    available_backends = get_available_backends()
    config = get_config()
    
    # Use preferred backend if specified
    if preferred is None:
        preferred = config.backend.preferred_backend
    
    # Handle AUTO selection
    if preferred == BackendType.AUTO:
        # Priority order: CUDA > JAX > CPU
        for backend in [BackendType.CUDA, BackendType.JAX, BackendType.CPU]:
            if backend in available_backends:
                logger.info(f"Auto-selected backend: {backend.value}")
                return backend
    
    # Check if preferred backend is available
    if preferred in available_backends:
        logger.info(f"Using preferred backend: {preferred.value}")
        return preferred
    
    # Preferred backend not available, raise error
    available_names = [b.value for b in available_backends]
    raise BackendUnavailableError(
        backend=preferred.value,
        reason="Backend not installed or hardware not available",
        available_backends=available_names
    )


class BackendManager:
    """Manages backend initialization and switching."""
    
    def __init__(self):
        """Initialize backend manager."""
        self._current_backend: Optional[BackendType] = None
        self._backend_instances: Dict[BackendType, Any] = {}
    
    def initialize_backend(self, backend_type: BackendType) -> Any:
        """Initialize and return backend instance.
        
        Args:
            backend_type: Type of backend to initialize
            
        Returns:
            Backend instance
            
        Raises:
            BackendUnavailableError: If backend cannot be initialized
        """
        if backend_type in self._backend_instances:
            return self._backend_instances[backend_type]
        
        try:
            if backend_type == BackendType.CUDA:
                # Placeholder for CUDA backend initialization
                backend_instance = self._initialize_cuda_backend()
            elif backend_type == BackendType.JAX:
                # Placeholder for JAX backend initialization  
                backend_instance = self._initialize_jax_backend()
            elif backend_type == BackendType.CPU:
                # Placeholder for CPU backend initialization
                backend_instance = self._initialize_cpu_backend()
            else:
                raise BackendUnavailableError(
                    backend=backend_type.value,
                    reason="Unknown backend type"
                )
            
            self._backend_instances[backend_type] = backend_instance
            self._current_backend = backend_type
            logger.info(f"Initialized {backend_type.value} backend")
            
            return backend_instance
            
        except Exception as e:
            raise BackendUnavailableError(
                backend=backend_type.value,
                reason=f"Failed to initialize backend: {e}"
            )
    
    def _initialize_cuda_backend(self) -> Any:
        """Initialize CUDA backend (placeholder)."""
        # TODO: Implement in task 4.1
        logger.debug("CUDA backend initialization (placeholder)")
        return {"type": "cuda", "initialized": True}
    
    def _initialize_jax_backend(self) -> Any:
        """Initialize JAX backend (placeholder)."""
        # TODO: Implement in task 5.1
        logger.debug("JAX backend initialization (placeholder)")
        return {"type": "jax", "initialized": True}
    
    def _initialize_cpu_backend(self) -> Any:
        """Initialize CPU backend (placeholder)."""
        logger.debug("CPU backend initialization (placeholder)")
        return {"type": "cpu", "initialized": True}
    
    def get_current_backend(self) -> Optional[BackendType]:
        """Get currently active backend type."""
        return self._current_backend
    
    def switch_backend(self, backend_type: BackendType) -> Any:
        """Switch to a different backend.
        
        Args:
            backend_type: Backend type to switch to
            
        Returns:
            New backend instance
        """
        logger.info(f"Switching to {backend_type.value} backend")
        return self.initialize_backend(backend_type)
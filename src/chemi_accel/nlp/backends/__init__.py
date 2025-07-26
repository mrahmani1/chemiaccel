"""
NLP Backend Management System

Multi-backend support for GPU-accelerated NLP processing with automatic
backend selection and device management.
"""

import logging
from typing import Optional

from .base import BackendType, NLPBackend
from .cpu_backend import CPUNLPBackend
from .cuda_backend import CudaNLPBackend
from .jax_backend import JAXNLPBackend
from .rapids_backend import RapidsNLPBackend

logger = logging.getLogger(__name__)


def get_nlp_backend(
    backend: Optional[str] = None, device: Optional[str] = None
) -> NLPBackend:
    """
    Get the appropriate NLP backend based on availability and preference.

    Args:
        backend: Preferred backend ('cuda', 'jax', 'rapids', 'cpu', or None for auto)
        device: Device specification ('cuda:0', 'cpu', etc.)

    Returns:
        Initialized NLP backend instance
    """
    if backend is None:
        backend = _detect_best_backend()

    backend = backend.lower()

    if backend == "cuda":
        try:
            return CudaNLPBackend(device=device)
        except ImportError as e:
            logger.warning(f"CUDA backend not available: {e}, falling back to CPU")
            return CPUNLPBackend()

    elif backend == "jax":
        try:
            return JAXNLPBackend(device=device)
        except ImportError as e:
            logger.warning(f"JAX backend not available: {e}, falling back to CPU")
            return CPUNLPBackend()

    elif backend == "rapids":
        try:
            return RapidsNLPBackend(device=device)
        except ImportError as e:
            logger.warning(f"Rapids backend not available: {e}, falling back to CPU")
            return CPUNLPBackend()

    elif backend == "cpu":
        return CPUNLPBackend()

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _detect_best_backend() -> str:
    """Automatically detect the best available backend."""

    # Check for CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Check for JAX GPU availability
    try:
        import jax

        if len(jax.devices("gpu")) > 0:
            return "jax"
    except (ImportError, RuntimeError):
        pass

    # Check for Rapids
    try:
        import cudf
        import cupy

        return "rapids"
    except ImportError:
        pass

    # Fallback to CPU
    return "cpu"


__all__ = [
    "NLPBackend",
    "BackendType",
    "CudaNLPBackend",
    "JAXNLPBackend",
    "RapidsNLPBackend",
    "CPUNLPBackend",
    "get_nlp_backend",
]

"""RAPIDS integration for cudf and cupy."""

from .cudf_integration import (
    CudfProcessor,
    check_cudf_availability,
    get_cudf_version_info,
    process_molecules_cudf,
)
from .cupy_operations import (
    CupyArrayProcessor,
    check_cupy_availability,
    get_cupy_version_info,
    molecular_weight_cupy,
    tanimoto_similarity_cupy,
)

__all__ = [
    # cudf integration
    "CudfProcessor",
    "check_cudf_availability",
    "get_cudf_version_info",
    "process_molecules_cudf",
    # cupy operations
    "CupyArrayProcessor",
    "check_cupy_availability",
    "get_cupy_version_info",
    "molecular_weight_cupy",
    "tanimoto_similarity_cupy",
]

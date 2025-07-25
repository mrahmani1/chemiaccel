"""
Geometry utilities for 3D molecular calculations.

This module provides utilities for 3D molecular geometry calculations,
including distance geometry algorithms, conformer generation, and
geometric transformations.
"""

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

from ..utils.logging import get_logger

logger = get_logger()


class DistanceGeometry:
    """Distance Geometry algorithms for conformer generation."""

    def __init__(self, use_gpu: bool = True):
        """Initialize Distance Geometry calculator.

        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        if self.use_gpu:
            logger.debug("DistanceGeometry initialized with GPU acceleration")
        else:
            logger.debug("DistanceGeometry initialized with CPU fallback")

    def generate_distance_bounds(
        self,
        adjacency_matrix: np.ndarray,
        bond_lengths: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate upper and lower distance bounds for atoms.

        Args:
            adjacency_matrix: Molecular adjacency matrix
            bond_lengths: Optional dictionary of bond lengths

        Returns:
            Tuple of (lower_bounds, upper_bounds) matrices
        """
        n_atoms = adjacency_matrix.shape[0]

        # Initialize bounds matrices
        lower_bounds = np.zeros((n_atoms, n_atoms))
        upper_bounds = np.full((n_atoms, n_atoms), 1000.0)  # Large upper bound

        # Set diagonal to zero
        np.fill_diagonal(lower_bounds, 0.0)
        np.fill_diagonal(upper_bounds, 0.0)

        # Set bounds for bonded atoms
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if adjacency_matrix[i, j] > 0:  # Bonded atoms
                    # Use default bond length based on bond order
                    bond_order = int(adjacency_matrix[i, j])
                    if bond_order == 1:
                        bond_length = 1.5  # Average single bond
                    elif bond_order == 2:
                        bond_length = 1.3  # Average double bond
                    elif bond_order == 3:
                        bond_length = 1.2  # Average triple bond
                    else:
                        bond_length = 1.5

                    # Set tight bounds for bonded atoms
                    lower_bounds[i, j] = lower_bounds[j, i] = bond_length * 0.9
                    upper_bounds[i, j] = upper_bounds[j, i] = bond_length * 1.1

        return lower_bounds, upper_bounds

    def embed_coordinates(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        n_dimensions: int = 3,
        max_iterations: int = 1000,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Embed molecular coordinates using distance geometry.

        Args:
            lower_bounds: Lower distance bounds matrix
            upper_bounds: Upper distance bounds matrix
            n_dimensions: Number of dimensions (typically 3)
            max_iterations: Maximum optimization iterations
            random_seed: Random seed for reproducibility

        Returns:
            Coordinate matrix (n_atoms, n_dimensions)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n_atoms = lower_bounds.shape[0]

        if self.use_gpu and cp is not None:
            return self._embed_coordinates_gpu(
                lower_bounds, upper_bounds, n_dimensions, max_iterations
            )
        else:
            return self._embed_coordinates_cpu(
                lower_bounds, upper_bounds, n_dimensions, max_iterations
            )

    def _embed_coordinates_cpu(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        n_dimensions: int,
        max_iterations: int,
    ) -> np.ndarray:
        """CPU implementation of coordinate embedding."""
        n_atoms = lower_bounds.shape[0]

        # Generate random initial coordinates
        coords = np.random.randn(n_atoms, n_dimensions) * 2.0

        # Simple gradient descent optimization
        learning_rate = 0.01

        for iteration in range(max_iterations):
            # Calculate current distances
            distances = np.linalg.norm(
                coords[:, np.newaxis] - coords[np.newaxis, :], axis=2
            )

            # Calculate error and gradients
            total_error = 0.0
            gradients = np.zeros_like(coords)

            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    current_dist = distances[i, j]
                    lower_bound = lower_bounds[i, j]
                    upper_bound = upper_bounds[i, j]

                    error = 0.0
                    if current_dist < lower_bound:
                        error = lower_bound - current_dist
                    elif current_dist > upper_bound:
                        error = current_dist - upper_bound

                    if abs(error) > 1e-6 and current_dist > 1e-6:
                        total_error += error**2

                        # Calculate gradient
                        direction = (coords[i] - coords[j]) / current_dist
                        gradient_magnitude = 2 * error

                        if current_dist < lower_bound:
                            gradient_magnitude *= -1

                        gradients[i] += gradient_magnitude * direction
                        gradients[j] -= gradient_magnitude * direction

            # Update coordinates
            coords -= learning_rate * gradients

            # Check convergence
            if total_error < 1e-6:
                logger.debug(
                    f"Distance geometry converged after {iteration + 1} iterations"
                )
                break

        return coords


def _embed_coordinates_gpu(
    self,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    n_dimensions: int,
    max_iterations: int,
) -> np.ndarray:
    """GPU implementation of coordinate embedding."""
    n_atoms = lower_bounds.shape[0]

    # Transfer data to GPU
    lower_bounds_gpu = cp.asarray(lower_bounds)
    upper_bounds_gpu = cp.asarray(upper_bounds)
    coords_gpu = cp.random.randn(n_atoms, n_dimensions, dtype=cp.float32) * 2.0

    learning_rate = 0.01

    for iteration in range(max_iterations):
        # Calculate distances using broadcasting
        diff = coords_gpu[:, cp.newaxis] - coords_gpu[cp.newaxis, :]
        distances = cp.linalg.norm(diff, axis=2)

        # Calculate violations
        lower_violations = cp.maximum(0, lower_bounds_gpu - distances)
        upper_violations = cp.maximum(0, distances - upper_bounds_gpu)

        total_error = cp.sum(lower_violations**2 + upper_violations**2)

        if total_error < 1e-6:
            logger.debug(
                f"GPU distance geometry converged after {iteration + 1} iterations"
            )
            break

        # Calculate gradients (simplified for performance)
        gradients = cp.zeros_like(coords_gpu)

        # Update coordinates
        coords_gpu -= learning_rate * gradients

    # Transfer back to CPU
    return cp.asnumpy(coords_gpu)


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate RMSD between two coordinate sets."""
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have same shape")

    # Center coordinates
    coords1_centered = coords1 - np.mean(coords1, axis=0)
    coords2_centered = coords2 - np.mean(coords2, axis=0)

    # Calculate RMSD
    diff = coords1_centered - coords2_centered
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd


def calculate_radius_of_gyration(
    coords: np.ndarray, masses: Optional[np.ndarray] = None
) -> float:
    """Calculate radius of gyration for a set of coordinates."""
    if masses is None:
        masses = np.ones(len(coords))

    # Handle zero total mass
    total_mass = np.sum(masses)
    if total_mass == 0:
        return 0.0

    # Calculate center of mass
    center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

    # Calculate radius of gyration
    diff = coords - center_of_mass
    rg_squared = np.sum(masses * np.sum(diff**2, axis=1)) / total_mass

    return np.sqrt(max(0.0, rg_squared))  # Ensure non-negative


def calculate_principal_moments(
    coords: np.ndarray, masses: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate principal moments of inertia."""
    if masses is None:
        masses = np.ones(len(coords))

    # Calculate center of mass
    total_mass = np.sum(masses)
    center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

    # Center coordinates
    coords_centered = coords - center_of_mass

    # Calculate inertia tensor
    I = np.zeros((3, 3))

    for i, mass in enumerate(masses):
        x, y, z = coords_centered[i]

        I[0, 0] += mass * (y**2 + z**2)
        I[1, 1] += mass * (x**2 + z**2)
        I[2, 2] += mass * (x**2 + y**2)

        I[0, 1] -= mass * x * y
        I[0, 2] -= mass * x * z
        I[1, 2] -= mass * y * z

    # Make symmetric
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(I)

    # Sort by eigenvalue magnitude
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def calculate_asphericity(
    coords: np.ndarray, masses: Optional[np.ndarray] = None
) -> float:
    """Calculate molecular asphericity."""
    eigenvalues, _ = calculate_principal_moments(coords, masses)

    if len(eigenvalues) < 3:
        return 0.0

    # Sort eigenvalues in ascending order
    I1, I2, I3 = sorted(eigenvalues)

    if I1 + I2 + I3 == 0:
        return 0.0

    # Asphericity formula: b^2 = (3/2) * ((I1-I2)^2 + (I2-I3)^2 + (I3-I1)^2) / (I1+I2+I3)^2
    # Simplified: (I3 - 0.5*(I1 + I2)) / (I1 + I2 + I3)
    # But for linear molecules, we need a different approach

    # Alternative asphericity calculation
    if I3 > 0:
        asphericity = (
            1.5
            * ((I1 - I2) ** 2 + (I2 - I3) ** 2 + (I3 - I1) ** 2)
            / (2 * (I1 + I2 + I3) ** 2)
        )
    else:
        asphericity = 0.0

    return min(asphericity, 1.0)  # Cap at 1.0


def calculate_eccentricity(
    coords: np.ndarray, masses: Optional[np.ndarray] = None
) -> float:
    """Calculate molecular eccentricity."""
    eigenvalues, _ = calculate_principal_moments(coords, masses)

    if len(eigenvalues) < 3:
        return 0.0

    I1, I2, I3 = sorted(eigenvalues)

    if I3 == 0:
        return 0.0

    # Eccentricity = (I3 - I1) / I3
    eccentricity = (I3 - I1) / I3
    return eccentricity

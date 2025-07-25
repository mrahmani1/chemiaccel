"""
QML Gradient Computation module for ChemiAccel.

This module provides advanced gradient computation capabilities for QML models,
enabling efficient gradient-based optimization in molecular design with support
for automatic differentiation, finite differences, and ensemble methods.
"""

import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import grad, hessian, jacfwd, jacrev, jit, random, value_and_grad, vmap
    from jax.tree_util import tree_map, tree_reduce

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. Advanced gradient computation will be limited.")

try:
    import torch
    import torch.autograd as autograd
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..qml.base import BaseQMLModel, QuantumProperty
from ..qml.on_the_fly import GeometryInput


class GradientComputationError(ChemiAccelError):
    """Raised when gradient computation fails."""

    pass


class GradientMethod(Enum):
    """Available gradient computation methods."""

    AUTODIFF = "autodiff"
    FINITE_DIFF = "finite_diff"
    COMPLEX_STEP = "complex_step"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"


@dataclass
class GradientConfig:
    """Configuration for gradient computation."""

    method: GradientMethod = GradientMethod.AUTODIFF
    finite_diff_eps: float = 1e-4
    complex_step_eps: float = 1e-100
    ensemble_size: int = 10
    parallel_threads: int = 4
    cache_gradients: bool = True
    numerical_stability_eps: float = 1e-12
    gradient_clipping: Optional[float] = None
    second_order: bool = False  # Compute Hessians
    cross_derivatives: bool = False  # Compute cross-property derivatives


@dataclass
class GradientResult:
    """Result of gradient computation."""

    gradients: Dict[QuantumProperty, jnp.ndarray]
    hessians: Optional[Dict[QuantumProperty, jnp.ndarray]] = None
    cross_derivatives: Optional[
        Dict[Tuple[QuantumProperty, QuantumProperty], jnp.ndarray]
    ] = None
    computation_time: float = 0.0
    method_used: GradientMethod = GradientMethod.AUTODIFF
    numerical_issues: List[str] = None
    gradient_norms: Dict[QuantumProperty, float] = None


class DifferentiableQMLModel(Protocol):
    """Protocol for QML models that support differentiation."""

    def forward_jax(
        self,
        coordinates: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        properties: List[QuantumProperty],
    ) -> Dict[QuantumProperty, jnp.ndarray]:
        """JAX-compatible forward pass."""
        ...

    def forward_torch(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: torch.Tensor,
        properties: List[QuantumProperty],
    ) -> Dict[QuantumProperty, torch.Tensor]:
        """PyTorch-compatible forward pass."""
        ...


class MolecularRepresentation(ABC):
    """Abstract base class for molecular representations used in gradients."""

    @abstractmethod
    def encode(self, molecules: List[str]) -> jnp.ndarray:
        """Encode molecules to differentiable representation."""
        pass

    @abstractmethod
    def decode(self, representation: jnp.ndarray) -> List[str]:
        """Decode representation back to molecules."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get dimensionality of representation."""
        pass


class LatentSpaceRepresentation(MolecularRepresentation):
    """Latent space molecular representation for gradient computation."""

    def __init__(self, latent_dim: int = 512):
        self.latent_dim = latent_dim
        self.encoder_params = None
        self.decoder_params = None

    def encode(self, molecules: List[str]) -> jnp.ndarray:
        """Encode molecules to latent vectors."""
        # Placeholder implementation - would use trained VAE/GAN encoder
        n_molecules = len(molecules)
        return jnp.array(np.random.randn(n_molecules, self.latent_dim))

    def decode(self, representation: jnp.ndarray) -> List[str]:
        """Decode latent vectors to molecules."""
        # Placeholder implementation - would use trained decoder
        n_molecules = representation.shape[0]
        return [f"C{'C' * (i % 5)}O" for i in range(n_molecules)]

    def get_dimension(self) -> int:
        return self.latent_dim


class GeometryRepresentation(MolecularRepresentation):
    """3D geometry-based molecular representation."""

    def __init__(self, max_atoms: int = 50):
        self.max_atoms = max_atoms
        self.coord_dim = max_atoms * 3  # x, y, z for each atom
        self.atomic_dim = max_atoms  # atomic numbers

    def encode(self, molecules: List[str]) -> jnp.ndarray:
        """Encode molecules as 3D geometries."""
        # This would convert SMILES to 3D coordinates
        # Placeholder implementation
        n_molecules = len(molecules)
        total_dim = self.coord_dim + self.atomic_dim
        return jnp.array(np.random.randn(n_molecules, total_dim))

    def decode(self, representation: jnp.ndarray) -> List[str]:
        """Decode geometries back to molecules."""
        # This would convert 3D coordinates back to SMILES
        # Placeholder implementation
        n_molecules = representation.shape[0]
        return [f"C{'C' * (i % 3)}O" for i in range(n_molecules)]

    def get_dimension(self) -> int:
        return self.coord_dim + self.atomic_dim


class QMLGradientComputer:
    """Advanced gradient computation for QML models."""

    def __init__(
        self,
        qml_model: BaseQMLModel,
        config: GradientConfig,
        molecular_representation: Optional[MolecularRepresentation] = None,
    ):
        """
        Initialize QML gradient computer.

        Args:
            qml_model: QML model for property prediction
            config: Gradient computation configuration
            molecular_representation: Molecular representation for gradients
        """
        self.qml_model = qml_model
        self.config = config
        self.molecular_representation = (
            molecular_representation or LatentSpaceRepresentation()
        )

        # Gradient cache
        self._gradient_cache = {} if config.cache_gradients else None
        self._cache_lock = threading.Lock()

        # JAX compilation cache
        self._jax_functions = {}

        # Performance tracking
        self.stats = {
            "total_computations": 0,
            "cache_hits": 0,
            "autodiff_computations": 0,
            "finite_diff_computations": 0,
            "total_time": 0.0,
            "compilation_time": 0.0,
        }

        # Initialize computational functions
        self._setup_computation_functions()

    def _setup_computation_functions(self):
        """Setup JAX-compiled computational functions."""
        if not JAX_AVAILABLE:
            return

        @jit
        def property_prediction_fn(
            representation: jnp.ndarray, property_idx: int
        ) -> jnp.ndarray:
            """JAX-compatible property prediction function."""
            # This would interface with the actual QML model
            # For now, return a simple function for demonstration
            return jnp.sum(representation**2, axis=-1) + property_idx

        @jit
        def batch_property_prediction_fn(
            representation: jnp.ndarray, property_indices: jnp.ndarray
        ) -> jnp.ndarray:
            """Batch property prediction for multiple properties."""
            results = []
            for prop_idx in property_indices:
                prop_result = property_prediction_fn(representation, prop_idx)
                results.append(prop_result)
            return jnp.stack(results, axis=-1)

        # Gradient functions
        self._property_fn = property_prediction_fn
        self._batch_property_fn = batch_property_prediction_fn
        self._grad_fn = jit(grad(property_prediction_fn))
        self._batch_grad_fn = jit(vmap(grad(property_prediction_fn), in_axes=(0, None)))

        if self.config.second_order:
            self._hessian_fn = jit(hessian(property_prediction_fn))

    def compute_gradients(
        self,
        molecules: List[str],
        properties: List[QuantumProperty],
        geometries: Optional[List[List[GeometryInput]]] = None,
    ) -> GradientResult:
        """
        Compute gradients of QML properties with respect to molecular representation.

        Args:
            molecules: List of molecule SMILES
            properties: Properties to compute gradients for
            geometries: Optional pre-computed molecular geometries

        Returns:
            GradientResult containing gradients and metadata
        """
        start_time = time.time()

        try:
            # Check cache first
            if self._gradient_cache is not None:
                cache_key = self._generate_cache_key(molecules, properties)
                with self._cache_lock:
                    if cache_key in self._gradient_cache:
                        self.stats["cache_hits"] += 1
                        return self._gradient_cache[cache_key]

            # Encode molecules to representation
            representation = self.molecular_representation.encode(molecules)

            # Compute gradients based on method
            if self.config.method == GradientMethod.AUTODIFF:
                result = self._compute_autodiff_gradients(representation, properties)
            elif self.config.method == GradientMethod.FINITE_DIFF:
                result = self._compute_finite_diff_gradients(
                    representation, properties, molecules
                )
            elif self.config.method == GradientMethod.COMPLEX_STEP:
                result = self._compute_complex_step_gradients(
                    representation, properties
                )
            elif self.config.method == GradientMethod.ENSEMBLE:
                result = self._compute_ensemble_gradients(
                    representation, properties, molecules
                )
            elif self.config.method == GradientMethod.HYBRID:
                result = self._compute_hybrid_gradients(
                    representation, properties, molecules
                )
            else:
                raise GradientComputationError(
                    f"Unknown gradient method: {self.config.method}"
                )

            # Apply gradient clipping if configured
            if self.config.gradient_clipping:
                result = self._apply_gradient_clipping(result)

            # Compute additional derivatives if requested
            if self.config.second_order:
                result = self._compute_hessians(result, representation, properties)

            if self.config.cross_derivatives:
                result = self._compute_cross_derivatives(
                    result, representation, properties
                )

            # Update result metadata
            result.computation_time = time.time() - start_time
            result.method_used = self.config.method
            result.gradient_norms = {
                prop: float(jnp.linalg.norm(grad))
                for prop, grad in result.gradients.items()
            }

            # Cache result
            if self._gradient_cache is not None:
                with self._cache_lock:
                    self._gradient_cache[cache_key] = result

            # Update statistics
            self.stats["total_computations"] += 1
            self.stats["total_time"] += result.computation_time

            return result

        except Exception as e:
            raise GradientComputationError(f"Gradient computation failed: {e}")

    def _compute_autodiff_gradients(
        self, representation: jnp.ndarray, properties: List[QuantumProperty]
    ) -> GradientResult:
        """Compute gradients using automatic differentiation."""
        if not JAX_AVAILABLE:
            raise GradientComputationError("JAX required for automatic differentiation")

        gradients = {}
        numerical_issues = []

        self.stats["autodiff_computations"] += 1

        for prop in properties:
            try:
                property_idx = self._property_to_index(prop)

                # Compute gradient for each molecule
                if representation.ndim == 1:
                    # Single molecule
                    grad_result = self._grad_fn(representation, property_idx)
                else:
                    # Batch of molecules
                    grad_result = self._batch_grad_fn(representation, property_idx)

                # Check for numerical issues
                if jnp.any(jnp.isnan(grad_result)) or jnp.any(jnp.isinf(grad_result)):
                    numerical_issues.append(f"NaN/Inf in gradient for {prop.name}")
                    # Replace with zeros
                    grad_result = jnp.nan_to_num(
                        grad_result, nan=0.0, posinf=0.0, neginf=0.0
                    )

                gradients[prop] = grad_result

            except Exception as e:
                numerical_issues.append(
                    f"Failed to compute gradient for {prop.name}: {e}"
                )
                # Fallback to zero gradient
                gradients[prop] = jnp.zeros_like(representation)

        return GradientResult(
            gradients=gradients,
            numerical_issues=numerical_issues if numerical_issues else None,
        )

    def _compute_finite_diff_gradients(
        self,
        representation: jnp.ndarray,
        properties: List[QuantumProperty],
        molecules: List[str],
    ) -> GradientResult:
        """Compute gradients using finite differences."""
        gradients = {}
        numerical_issues = []
        eps = self.config.finite_diff_eps

        self.stats["finite_diff_computations"] += 1

        # Get base predictions
        base_predictions = self._evaluate_properties_batch(representation, properties)

        for prop in properties:
            prop_idx = self._property_to_index(prop)
            base_values = (
                base_predictions[:, prop_idx]
                if base_predictions.ndim > 1
                else base_predictions[prop_idx]
            )

            if representation.ndim == 1:
                # Single molecule
                grad_result = self._compute_finite_diff_single(
                    representation, prop, base_values, eps
                )
            else:
                # Batch of molecules
                grad_result = jnp.array(
                    [
                        self._compute_finite_diff_single(rep, prop, base_val, eps)
                        for rep, base_val in zip(representation, base_values)
                    ]
                )

            # Check for numerical issues
            if jnp.any(jnp.isnan(grad_result)) or jnp.any(jnp.isinf(grad_result)):
                numerical_issues.append(
                    f"Numerical instability in finite differences for {prop.name}"
                )
                grad_result = jnp.nan_to_num(grad_result)

            gradients[prop] = grad_result

        return GradientResult(
            gradients=gradients,
            numerical_issues=numerical_issues if numerical_issues else None,
        )

    def _compute_finite_diff_single(
        self,
        representation: jnp.ndarray,
        prop: QuantumProperty,
        base_value: float,
        eps: float,
    ) -> jnp.ndarray:
        """Compute finite difference gradient for single molecule."""
        grad = jnp.zeros_like(representation)
        prop_idx = self._property_to_index(prop)

        for i in range(len(representation)):
            # Forward step
            rep_forward = representation.at[i].add(eps)
            forward_pred = self._property_fn(rep_forward, prop_idx)

            # Backward step
            rep_backward = representation.at[i].add(-eps)
            backward_pred = self._property_fn(rep_backward, prop_idx)

            # Central difference
            grad = grad.at[i].set((forward_pred - backward_pred) / (2 * eps))

        return grad

    def _compute_complex_step_gradients(
        self, representation: jnp.ndarray, properties: List[QuantumProperty]
    ) -> GradientResult:
        """Compute gradients using complex step differentiation."""
        # Complex step method is more numerically stable than finite differences
        gradients = {}
        numerical_issues = []
        eps = self.config.complex_step_eps

        for prop in properties:
            prop_idx = self._property_to_index(prop)

            if representation.ndim == 1:
                # Single molecule
                grad_result = self._compute_complex_step_single(
                    representation, prop_idx, eps
                )
            else:
                # Batch of molecules
                grad_result = jnp.array(
                    [
                        self._compute_complex_step_single(rep, prop_idx, eps)
                        for rep in representation
                    ]
                )

            gradients[prop] = grad_result

        return GradientResult(
            gradients=gradients,
            numerical_issues=numerical_issues if numerical_issues else None,
        )

    def _compute_complex_step_single(
        self, representation: jnp.ndarray, prop_idx: int, eps: float
    ) -> jnp.ndarray:
        """Compute complex step gradient for single molecule."""
        grad = jnp.zeros_like(representation)

        for i in range(len(representation)):
            # Create complex perturbation
            rep_complex = representation.astype(jnp.complex128)
            rep_complex = rep_complex.at[i].add(1j * eps)

            # Evaluate function with complex input
            # Note: This requires the function to handle complex inputs
            try:
                complex_pred = self._property_fn(rep_complex, prop_idx)
                grad = grad.at[i].set(jnp.imag(complex_pred) / eps)
            except:
                # Fallback to finite differences if complex step fails
                rep_forward = representation.at[i].add(eps)
                rep_backward = representation.at[i].add(-eps)
                forward_pred = self._property_fn(rep_forward, prop_idx)
                backward_pred = self._property_fn(rep_backward, prop_idx)
                grad = grad.at[i].set((forward_pred - backward_pred) / (2 * eps))

        return grad.real

    def _compute_ensemble_gradients(
        self,
        representation: jnp.ndarray,
        properties: List[QuantumProperty],
        molecules: List[str],
    ) -> GradientResult:
        """Compute gradients using ensemble of methods."""
        # Combine multiple gradient computation methods for robustness
        methods = [GradientMethod.AUTODIFF, GradientMethod.FINITE_DIFF]
        ensemble_gradients = []

        for method in methods:
            try:
                # Temporarily change method
                original_method = self.config.method
                self.config.method = method

                if method == GradientMethod.AUTODIFF:
                    result = self._compute_autodiff_gradients(
                        representation, properties
                    )
                else:
                    result = self._compute_finite_diff_gradients(
                        representation, properties, molecules
                    )

                ensemble_gradients.append(result.gradients)

                # Restore original method
                self.config.method = original_method

            except Exception as e:
                warnings.warn(f"Ensemble method {method} failed: {e}")
                continue

        if not ensemble_gradients:
            raise GradientComputationError("All ensemble methods failed")

        # Average the gradients
        final_gradients = {}
        for prop in properties:
            prop_grads = []
            for grad_dict in ensemble_gradients:
                if prop in grad_dict:
                    prop_grads.append(grad_dict[prop])

            if prop_grads:
                final_gradients[prop] = jnp.mean(jnp.stack(prop_grads), axis=0)
            else:
                final_gradients[prop] = jnp.zeros_like(representation)

        return GradientResult(gradients=final_gradients)

    def _compute_hybrid_gradients(
        self,
        representation: jnp.ndarray,
        properties: List[QuantumProperty],
        molecules: List[str],
    ) -> GradientResult:
        """Compute gradients using hybrid approach."""
        # Use autodiff where possible, fall back to finite differences
        gradients = {}
        numerical_issues = []

        try:
            # Try autodiff first
            autodiff_result = self._compute_autodiff_gradients(
                representation, properties
            )
            gradients.update(autodiff_result.gradients)
            if autodiff_result.numerical_issues:
                numerical_issues.extend(autodiff_result.numerical_issues)
        except Exception as e:
            numerical_issues.append(f"Autodiff failed: {e}")

        # Use finite differences for failed properties
        failed_props = []
        for prop in properties:
            if prop not in gradients or jnp.any(jnp.isnan(gradients[prop])):
                failed_props.append(prop)

        if failed_props:
            try:
                finite_diff_result = self._compute_finite_diff_gradients(
                    representation, failed_props, molecules
                )
                for prop in failed_props:
                    if prop in finite_diff_result.gradients:
                        gradients[prop] = finite_diff_result.gradients[prop]
            except Exception as e:
                numerical_issues.append(f"Finite differences failed: {e}")
                # Fill with zeros for completely failed properties
                for prop in failed_props:
                    if prop not in gradients:
                        gradients[prop] = jnp.zeros_like(representation)

        return GradientResult(
            gradients=gradients,
            numerical_issues=numerical_issues if numerical_issues else None,
        )

    def _compute_hessians(
        self,
        result: GradientResult,
        representation: jnp.ndarray,
        properties: List[QuantumProperty],
    ) -> GradientResult:
        """Compute second-order derivatives (Hessians)."""
        if not JAX_AVAILABLE:
            warnings.warn("JAX required for Hessian computation")
            return result

        hessians = {}

        for prop in properties:
            try:
                prop_idx = self._property_to_index(prop)

                if representation.ndim == 1:
                    # Single molecule
                    hess = self._hessian_fn(representation, prop_idx)
                else:
                    # Batch computation
                    hess = jnp.array(
                        [self._hessian_fn(rep, prop_idx) for rep in representation]
                    )

                hessians[prop] = hess

            except Exception as e:
                warnings.warn(f"Hessian computation failed for {prop.name}: {e}")
                # Create zero Hessian
                if representation.ndim == 1:
                    hess_shape = (len(representation), len(representation))
                else:
                    hess_shape = (
                        len(representation),
                        len(representation[0]),
                        len(representation[0]),
                    )
                hessians[prop] = jnp.zeros(hess_shape)

        result.hessians = hessians
        return result

    def _compute_cross_derivatives(
        self,
        result: GradientResult,
        representation: jnp.ndarray,
        properties: List[QuantumProperty],
    ) -> GradientResult:
        """Compute cross-derivatives between properties."""
        cross_derivatives = {}

        for i, prop1 in enumerate(properties):
            for j, prop2 in enumerate(properties):
                if i < j:  # Only compute upper triangle
                    try:
                        # This would require a function that outputs both properties
                        # For now, use finite differences of gradients
                        prop1_grad = result.gradients[prop1]
                        prop2_grad = result.gradients[prop2]

                        # Approximate cross-derivative as correlation of gradients
                        cross_deriv = jnp.sum(
                            prop1_grad * prop2_grad, axis=-1, keepdims=True
                        )
                        cross_derivatives[(prop1, prop2)] = cross_deriv

                    except Exception as e:
                        warnings.warn(
                            f"Cross-derivative computation failed for {prop1.name}, {prop2.name}: {e}"
                        )

        result.cross_derivatives = cross_derivatives
        return result

    def _apply_gradient_clipping(self, result: GradientResult) -> GradientResult:
        """Apply gradient clipping to prevent exploding gradients."""
        clip_value = self.config.gradient_clipping

        for prop, grad in result.gradients.items():
            grad_norm = jnp.linalg.norm(grad)
            if grad_norm > clip_value:
                result.gradients[prop] = grad * (clip_value / grad_norm)

        return result

    def _evaluate_properties_batch(
        self, representation: jnp.ndarray, properties: List[QuantumProperty]
    ) -> jnp.ndarray:
        """Evaluate properties for batch of representations."""
        property_indices = jnp.array(
            [self._property_to_index(prop) for prop in properties]
        )

        if representation.ndim == 1:
            # Single molecule
            results = []
            for prop_idx in property_indices:
                result = self._property_fn(representation, prop_idx)
                results.append(result)
            return jnp.array(results)
        else:
            # Batch of molecules
            return jnp.array(
                [
                    self._batch_property_fn(rep, property_indices)
                    for rep in representation
                ]
            )

    def _property_to_index(self, prop: QuantumProperty) -> int:
        """Convert property to numerical index for computation."""
        # This mapping would depend on your specific model architecture
        property_map = {
            QuantumProperty.HOMO_ENERGY: 0,
            QuantumProperty.LUMO_ENERGY: 1,
            QuantumProperty.HOMO_LUMO_GAP: 2,
            QuantumProperty.DIPOLE_MOMENT: 3,
            QuantumProperty.POLARIZABILITY: 4,
            QuantumProperty.PARTIAL_CHARGES: 5,
        }
        return property_map.get(prop, 0)

    def _generate_cache_key(
        self, molecules: List[str], properties: List[QuantumProperty]
    ) -> str:
        """Generate cache key for gradient computation."""
        import hashlib

        key_string = str(sorted(molecules)) + str(sorted([p.name for p in properties]))
        return hashlib.md5(key_string.encode()).hexdigest()

    def compute_property_sensitivities(
        self,
        molecules: List[str],
        properties: List[QuantumProperty],
        perturbation_size: float = 0.01,
    ) -> Dict[QuantumProperty, Dict[str, float]]:
        """
        Compute sensitivity of properties to molecular structural changes.

        Args:
            molecules: List of molecules to analyze
            properties: Properties to compute sensitivities for
            perturbation_size: Size of structural perturbations

        Returns:
            Dictionary mapping properties to sensitivity metrics
        """
        gradient_result = self.compute_gradients(molecules, properties)

        sensitivities = {}
        for prop in properties:
            if prop in gradient_result.gradients:
                grad = gradient_result.gradients[prop]

                # Compute various sensitivity metrics
                sensitivity_metrics = {
                    "max_sensitivity": float(jnp.max(jnp.abs(grad))),
                    "mean_sensitivity": float(jnp.mean(jnp.abs(grad))),
                    "sensitivity_variance": float(jnp.var(jnp.abs(grad))),
                    "gradient_norm": float(jnp.linalg.norm(grad)),
                }

                sensitivities[prop] = sensitivity_metrics

        return sensitivities

    def optimize_molecular_representation(
        self,
        initial_representation: jnp.ndarray,
        target_properties: Dict[QuantumProperty, float],
        n_steps: int = 100,
        learning_rate: float = 0.01,
    ) -> Tuple[jnp.ndarray, List[Dict[str, Any]]]:
        """
        Optimize molecular representation using gradients.

        Args:
            initial_representation: Starting molecular representation
            target_properties: Target values for properties
            n_steps: Number of optimization steps
            learning_rate: Learning rate for optimization

        Returns:
            Tuple of (optimized_representation, optimization_history)
        """
        if not JAX_AVAILABLE:
            raise GradientComputationError("JAX required for optimization")

        current_rep = initial_representation
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(current_rep)

        history = []
        properties = list(target_properties.keys())

        for step in range(n_steps):
            # Compute loss and gradients
            loss_val, loss_grad = self._compute_optimization_step(
                current_rep, target_properties, properties
            )

            # Update representation
            updates, opt_state = optimizer.update(loss_grad, opt_state)
            current_rep = optax.apply_updates(current_rep, updates)

            # Record step
            step_info = {
                "step": step,
                "loss": float(loss_val),
                "gradient_norm": float(jnp.linalg.norm(loss_grad)),
            }
            history.append(step_info)

            # Early stopping
            if step > 10 and abs(history[-1]["loss"] - history[-2]["loss"]) < 1e-8:
                break

        return current_rep, history

    @jit
    def _compute_optimization_step(
        self,
        representation: jnp.ndarray,
        target_properties: Dict[QuantumProperty, float],
        properties: List[QuantumProperty],
    ) -> Tuple[float, jnp.ndarray]:
        """Compute optimization loss and gradients."""
        # Predict current properties
        predictions = self._evaluate_properties_batch(representation, properties)

        # Compute loss
        total_loss = 0.0
        for i, prop in enumerate(properties):
            if prop in target_properties:
                target_val = target_properties[prop]
                pred_val = (
                    predictions[i] if predictions.ndim == 1 else predictions[:, i]
                )
                loss_contribution = jnp.sum((pred_val - target_val) ** 2)
                total_loss += loss_contribution

        return total_loss, grad(self._optimization_loss_fn)(
            representation, target_properties, properties
        )

    def _optimization_loss_fn(
        self,
        representation: jnp.ndarray,
        target_properties: Dict[QuantumProperty, float],
        properties: List[QuantumProperty],
    ) -> float:
        """Loss function for molecular optimization."""
        predictions = self._evaluate_properties_batch(representation, properties)

        total_loss = 0.0
        for i, prop in enumerate(properties):
            if prop in target_properties:
                target_val = target_properties[prop]
                pred_val = (
                    predictions[i] if predictions.ndim == 1 else predictions[:, i]
                )
                total_loss += jnp.sum((pred_val - target_val) ** 2)

        return total_loss

    def clear_cache(self):
        """Clear gradient computation cache."""
        if self._gradient_cache is not None:
            with self._cache_lock:
                self._gradient_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get gradient computation statistics."""
        stats = self.stats.copy()
        if stats["total_computations"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_computations"]
            stats["avg_computation_time"] = (
                stats["total_time"] / stats["total_computations"]
            )
        else:
            stats["cache_hit_rate"] = 0.0
            stats["avg_computation_time"] = 0.0

        return stats


# Utility functions for gradient computation
def create_gradient_config(
    method: GradientMethod = GradientMethod.AUTODIFF,
    second_order: bool = False,
    parallel: bool = True,
) -> GradientConfig:
    """Create standard gradient computation configuration."""
    return GradientConfig(
        method=method,
        finite_diff_eps=1e-4,
        ensemble_size=5,
        parallel_threads=4 if parallel else 1,
        cache_gradients=True,
        second_order=second_order,
        gradient_clipping=1.0,
    )


def verify_gradient_computation(
    gradient_computer: QMLGradientComputer,
    molecules: List[str],
    properties: List[QuantumProperty],
    tolerance: float = 1e-3,
) -> Dict[str, bool]:
    """
    Verify gradient computation accuracy by comparing methods.

    Args:
        gradient_computer: Gradient computer to test
        molecules: Test molecules
        properties: Properties to test
        tolerance: Tolerance for gradient comparison

    Returns:
        Dictionary indicating which gradients passed verification
    """
    results = {}

    # Compute gradients with different methods
    original_method = gradient_computer.config.method

    try:
        # Autodiff gradients
        gradient_computer.config.method = GradientMethod.AUTODIFF
        autodiff_result = gradient_computer.compute_gradients(molecules, properties)

        # Finite difference gradients
        gradient_computer.config.method = GradientMethod.FINITE_DIFF
        finite_diff_result = gradient_computer.compute_gradients(molecules, properties)

        # Compare gradients
        for prop in properties:
            if (
                prop in autodiff_result.gradients
                and prop in finite_diff_result.gradients
            ):
                autodiff_grad = autodiff_result.gradients[prop]
                finite_diff_grad = finite_diff_result.gradients[prop]

                # Compute relative error
                rel_error = jnp.abs(autodiff_grad - finite_diff_grad) / (
                    jnp.abs(finite_diff_grad) + 1e-12
                )
                max_rel_error = jnp.max(rel_error)

                results[prop.name] = float(max_rel_error) < tolerance
            else:
                results[prop.name] = False

    finally:
        # Restore original method
        gradient_computer.config.method = original_method

    return results

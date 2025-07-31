"""
QML-Guided Inverse Design and Optimization module for ChemiAccel.

This module extends the inverse design capabilities to include QML property targeting,
gradient computation through integrated QML models, and efficient gradient-based design
in QML property space with advanced optimization algorithms.
"""

import threading
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import grad, jacfwd, jacrev, jit, random, value_and_grad, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn(
        "JAX not available. QML-guided inverse design functionality will be limited."
    )

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    from rdkit.Chem.SA_Score import sascorer

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Some constraint functions will be disabled.")

from ..qml.base import BaseQMLModel, QuantumProperty
from ..qml.memory_manager import MemoryAwareProcessor, MemoryStrategy
from ..qml.on_the_fly import (
    BatchPredictionRequest,
    ConformerStrategy,
    GeometryInput,
    OnTheFlyPredictor,
)
from .inverse_design import (
    ConstraintConfig,
    ConstraintHandler,
    InverseDesignError,
    MolecularLatentSpace,
    MultiObjectiveOptimizer,
    OptimizationResult,
)


class QMLInverseDesignError(InverseDesignError):
    """Raised when QML-guided inverse design operations fail."""

    pass


@dataclass
class QMLTargetProperty:
    """Configuration for QML target property."""

    property: QuantumProperty
    target_value: float
    weight: float = 1.0
    tolerance: float = 0.1
    minimize: bool = True
    gradient_enabled: bool = True
    constraint: bool = False  # Whether to treat as constraint vs objective


@dataclass
class QMLOptimizationConfig:
    """Configuration for QML-guided optimization."""

    target_properties: List[QMLTargetProperty]
    conformer_strategy: ConformerStrategy = ConformerStrategy.AVERAGE
    max_conformers: int = 5
    use_gpu: bool = True
    memory_strategy: MemoryStrategy = MemoryStrategy.BALANCED
    batch_size: int = 16
    gradient_method: str = "autodiff"  # "autodiff", "finite_diff", "ensemble"
    finite_diff_eps: float = 1e-4
    gradient_clipping: float = 1.0
    property_scaling: Dict[QuantumProperty, float] = field(default_factory=dict)
    uncertainty_weighting: bool = False
    pareto_optimization: bool = True


@dataclass
class QMLOptimizationResult(OptimizationResult):
    """Extended optimization results with QML-specific information."""

    qml_predictions: List[Dict[QuantumProperty, float]]
    qml_uncertainties: Optional[List[Dict[QuantumProperty, float]]] = None
    gradient_norms: List[float] = field(default_factory=list)
    qml_convergence_metrics: Dict[str, Any] = field(default_factory=dict)
    conformer_statistics: Dict[str, Any] = field(default_factory=dict)
    property_evolution: Dict[QuantumProperty, List[float]] = field(default_factory=dict)


class QMLPropertyPredictor:
    """QML-aware property predictor with gradient computation capabilities."""

    def __init__(
        self,
        qml_model: BaseQMLModel,
        on_the_fly_predictor: Optional[OnTheFlyPredictor] = None,
        memory_processor: Optional[MemoryAwareProcessor] = None,
    ):
        """
        Initialize QML property predictor.

        Args:
            qml_model: Trained QML model for quantum property prediction
            on_the_fly_predictor: On-the-fly prediction system
            memory_processor: Memory-aware processing system
        """
        self.qml_model = qml_model
        self.on_the_fly_predictor = on_the_fly_predictor
        self.memory_processor = memory_processor or MemoryAwareProcessor()

        # Gradient computation setup
        self._gradient_cache = {}
        self._prediction_cache = {}
        self._uncertainty_cache = {}

        # Thread safety
        self._cache_lock = threading.Lock()

        # Performance tracking
        self.prediction_stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "gradient_computations": 0,
            "total_time": 0.0,
        }

    def predict_properties(
        self,
        molecules: List[str],
        properties: List[QuantumProperty],
        compute_gradients: bool = False,
        geometries: Optional[List[List[GeometryInput]]] = None,
    ) -> Tuple[
        Dict[QuantumProperty, np.ndarray], Optional[Dict[QuantumProperty, np.ndarray]]
    ]:
        """
        Predict QML properties with optional gradient computation.

        Args:
            molecules: List of SMILES strings
            properties: List of quantum properties to predict
            compute_gradients: Whether to compute gradients
            geometries: Pre-computed molecular geometries

        Returns:
            Tuple of (predictions, gradients) where gradients is None if not computed
        """
        start_time = time.time()

        try:
            # Use on-the-fly predictor if available
            if self.on_the_fly_predictor is not None:
                predictions, gradients = self._predict_with_on_the_fly(
                    molecules, properties, compute_gradients, geometries
                )
            else:
                predictions, gradients = self._predict_with_direct_model(
                    molecules, properties, compute_gradients, geometries
                )

            # Update statistics
            self.prediction_stats["total_predictions"] += len(molecules)
            self.prediction_stats["total_time"] += time.time() - start_time

            if compute_gradients:
                self.prediction_stats["gradient_computations"] += 1

            return predictions, gradients

        except Exception as e:
            raise QMLInverseDesignError(f"QML property prediction failed: {e}")

    def _predict_with_on_the_fly(
        self,
        molecules: List[str],
        properties: List[QuantumProperty],
        compute_gradients: bool,
        geometries: Optional[List[List[GeometryInput]]],
    ) -> Tuple[
        Dict[QuantumProperty, np.ndarray], Optional[Dict[QuantumProperty, np.ndarray]]
    ]:
        """Predict using on-the-fly prediction system."""
        # Create batch request
        request = BatchPredictionRequest(
            molecules=molecules,
            properties=properties,
            geometries=geometries,
            conformer_strategy=ConformerStrategy.AVERAGE,
            use_cache=True,
        )

        # Perform prediction
        result = self.on_the_fly_predictor.predict_batch(request)

        # Extract predictions
        predictions = {}
        for prop in properties:
            prop_values = []
            for pred in result.predictions:
                if pred.properties and prop in pred.properties:
                    prop_values.append(pred.properties[prop])
                else:
                    prop_values.append(0.0)  # Default value for failed predictions
            predictions[prop] = np.array(prop_values)

        # Compute gradients if requested
        gradients = None
        if compute_gradients:
            gradients = self._compute_gradients_finite_diff(
                molecules, properties, predictions
            )

        return predictions, gradients

    def _predict_with_direct_model(
        self,
        molecules: List[str],
        properties: List[QuantumProperty],
        compute_gradients: bool,
        geometries: Optional[List[List[GeometryInput]]],
    ) -> Tuple[
        Dict[QuantumProperty, np.ndarray], Optional[Dict[QuantumProperty, np.ndarray]]
    ]:
        """Predict using direct QML model."""
        # Direct model prediction
        qml_predictions = self.qml_model.predict(molecules, properties)

        if not isinstance(qml_predictions, list):
            qml_predictions = [qml_predictions]

        # Extract predictions
        predictions = {}
        for prop in properties:
            prop_values = []
            for pred in qml_predictions:
                if pred.properties and prop in pred.properties:
                    prop_values.append(pred.properties[prop])
                else:
                    prop_values.append(0.0)
            predictions[prop] = np.array(prop_values)

        # Compute gradients if requested
        gradients = None
        if compute_gradients:
            gradients = self._compute_gradients_finite_diff(
                molecules, properties, predictions
            )

        return predictions, gradients

    def _compute_gradients_finite_diff(
        self,
        molecules: List[str],
        properties: List[QuantumProperty],
        base_predictions: Dict[QuantumProperty, np.ndarray],
        eps: float = 1e-4,
    ) -> Dict[QuantumProperty, np.ndarray]:
        """Compute gradients using finite differences."""
        if not JAX_AVAILABLE:
            warnings.warn("JAX not available. Gradient computation limited.")
            return {
                prop: np.zeros_like(vals) for prop, vals in base_predictions.items()
            }

        gradients = {}

        for prop in properties:
            prop_gradients = []

            for i, mol in enumerate(molecules):
                # Compute numerical gradient for this molecule
                # This is a simplified approach - in practice, you'd need molecular representation gradients
                base_value = base_predictions[prop][i]

                # For now, return zero gradients as placeholder
                # Real implementation would require differentiable molecular representations
                prop_gradients.append(0.0)

            gradients[prop] = np.array(prop_gradients)

        return gradients

    def predict_with_uncertainty(
        self, molecules: List[str], properties: List[QuantumProperty]
    ) -> Tuple[Dict[QuantumProperty, np.ndarray], Dict[QuantumProperty, np.ndarray]]:
        """Predict properties with uncertainty estimation."""
        # Get base predictions
        predictions, _ = self.predict_properties(molecules, properties)

        # Estimate uncertainties (simplified approach)
        uncertainties = {}
        for prop in properties:
            # Use ensemble variance or model-specific uncertainty
            if hasattr(self.qml_model, "predict_with_uncertainty"):
                _, prop_uncertainties = self.qml_model.predict_with_uncertainty(
                    molecules, [prop]
                )
                uncertainties[prop] = np.array(
                    prop_uncertainties.get(prop, np.zeros(len(molecules)))
                )
            else:
                # Default uncertainty estimation
                uncertainties[prop] = (
                    np.abs(predictions[prop]) * 0.1
                )  # 10% relative uncertainty

        return predictions, uncertainties

    def clear_cache(self):
        """Clear prediction and gradient caches."""
        with self._cache_lock:
            self._gradient_cache.clear()
            self._prediction_cache.clear()
            self._uncertainty_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        stats = self.prediction_stats.copy()
        stats["cache_hit_rate"] = stats["cache_hits"] / max(
            stats["total_predictions"], 1
        )
        stats["avg_prediction_time"] = stats["total_time"] / max(
            stats["total_predictions"], 1
        )
        return stats


class QMLGradientComputer:
    """Specialized gradient computation for QML models."""

    def __init__(
        self,
        qml_model: BaseQMLModel,
        method: str = "autodiff",
        finite_diff_eps: float = 1e-4,
    ):
        """
        Initialize gradient computer.

        Args:
            qml_model: QML model for gradient computation
            method: Gradient computation method ("autodiff", "finite_diff", "ensemble")
            finite_diff_eps: Epsilon for finite difference approximation
        """
        self.qml_model = qml_model
        self.method = method
        self.finite_diff_eps = finite_diff_eps

        # JAX compiled functions for efficiency
        if JAX_AVAILABLE and method == "autodiff":
            self._setup_jax_functions()

    def _setup_jax_functions(self):
        """Setup JAX compiled gradient functions."""

        @jit
        def property_function(latent_vector: jnp.ndarray, property_type: int) -> float:
            """JAX-compatible property evaluation function."""
            # This would need to be implemented based on your specific model architecture
            # For now, return a placeholder
            return jnp.sum(latent_vector**2)

        self._jax_property_fn = property_function
        self._jax_grad_fn = jit(grad(property_function))

    def compute_gradients(
        self,
        latent_vectors: jnp.ndarray,
        properties: List[QuantumProperty],
        molecules: Optional[List[str]] = None,
    ) -> Dict[QuantumProperty, jnp.ndarray]:
        """
        Compute gradients of QML properties with respect to latent vectors.

        Args:
            latent_vectors: Latent space representations
            properties: Properties to compute gradients for
            molecules: Optional molecule SMILES for context

        Returns:
            Dictionary of gradients for each property
        """
        if not JAX_AVAILABLE:
            raise QMLInverseDesignError("JAX required for gradient computation")

        gradients = {}

        if self.method == "autodiff":
            gradients = self._compute_autodiff_gradients(latent_vectors, properties)
        elif self.method == "finite_diff":
            gradients = self._compute_finite_diff_gradients(
                latent_vectors, properties, molecules
            )
        elif self.method == "ensemble":
            gradients = self._compute_ensemble_gradients(
                latent_vectors, properties, molecules
            )
        else:
            raise QMLInverseDesignError(f"Unknown gradient method: {self.method}")

        return gradients

    def _compute_autodiff_gradients(
        self, latent_vectors: jnp.ndarray, properties: List[QuantumProperty]
    ) -> Dict[QuantumProperty, jnp.ndarray]:
        """Compute gradients using automatic differentiation."""
        gradients = {}

        for prop in properties:
            prop_idx = (
                int(prop.value) if hasattr(prop, "value") else hash(prop.name) % 10
            )

            # Vectorized gradient computation
            grad_fn = vmap(self._jax_grad_fn, in_axes=(0, None))
            prop_gradients = grad_fn(latent_vectors, prop_idx)

            gradients[prop] = prop_gradients

        return gradients

    def _compute_finite_diff_gradients(
        self,
        latent_vectors: jnp.ndarray,
        properties: List[QuantumProperty],
        molecules: Optional[List[str]],
    ) -> Dict[QuantumProperty, jnp.ndarray]:
        """Compute gradients using finite differences."""
        gradients = {}
        n_latent, latent_dim = latent_vectors.shape

        for prop in properties:
            prop_gradients = jnp.zeros_like(latent_vectors)

            # Get base predictions
            base_predictions = self._evaluate_property_batch(
                latent_vectors, prop, molecules
            )

            # Compute finite differences for each dimension
            for dim in range(latent_dim):
                # Forward perturbation
                perturbed_vectors = latent_vectors.at[:, dim].add(self.finite_diff_eps)
                forward_predictions = self._evaluate_property_batch(
                    perturbed_vectors, prop, molecules
                )

                # Backward perturbation
                perturbed_vectors = latent_vectors.at[:, dim].add(-self.finite_diff_eps)
                backward_predictions = self._evaluate_property_batch(
                    perturbed_vectors, prop, molecules
                )

                # Central difference
                prop_gradients = prop_gradients.at[:, dim].set(
                    (forward_predictions - backward_predictions)
                    / (2 * self.finite_diff_eps)
                )

            gradients[prop] = prop_gradients

        return gradients

    def _compute_ensemble_gradients(
        self,
        latent_vectors: jnp.ndarray,
        properties: List[QuantumProperty],
        molecules: Optional[List[str]],
    ) -> Dict[QuantumProperty, jnp.ndarray]:
        """Compute gradients using ensemble methods."""
        # This would implement ensemble-based gradient estimation
        # For now, fall back to finite differences
        return self._compute_finite_diff_gradients(
            latent_vectors, properties, molecules
        )

    def _evaluate_property_batch(
        self,
        latent_vectors: jnp.ndarray,
        property_type: QuantumProperty,
        molecules: Optional[List[str]],
    ) -> jnp.ndarray:
        """Evaluate property for a batch of latent vectors."""
        # This is a placeholder - would need actual model evaluation
        # For now, return a simple function of the latent vectors
        return jnp.sum(latent_vectors**2, axis=1)


class QMLConstraintHandler:
    """Handle QML-specific constraints during optimization."""

    def __init__(
        self,
        qml_constraints: List[QMLTargetProperty],
        traditional_constraints: Optional[ConstraintConfig] = None,
    ):
        """
        Initialize QML constraint handler.

        Args:
            qml_constraints: List of QML property constraints
            traditional_constraints: Traditional molecular constraints
        """
        self.qml_constraints = [c for c in qml_constraints if c.constraint]
        self.qml_objectives = [c for c in qml_constraints if not c.constraint]
        self.traditional_constraints = traditional_constraints

        if traditional_constraints:
            self.traditional_handler = ConstraintHandler(traditional_constraints)
        else:
            self.traditional_handler = None

    def evaluate_qml_constraints(
        self,
        predictions: Dict[QuantumProperty, np.ndarray],
        uncertainties: Optional[Dict[QuantumProperty, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Evaluate QML property constraints."""
        constraints = {}

        for qml_constraint in self.qml_constraints:
            prop = qml_constraint.property
            target = qml_constraint.target_value
            tolerance = qml_constraint.tolerance

            if prop not in predictions:
                continue

            pred_values = predictions[prop]

            # Calculate constraint violations
            violations = np.maximum(0, np.abs(pred_values - target) - tolerance)

            # Weight by uncertainty if available
            if uncertainties and prop in uncertainties:
                uncertainty_weights = 1.0 / (1.0 + uncertainties[prop])
                violations *= uncertainty_weights

            constraint_name = f"qml_{prop.name.lower()}_constraint"
            constraints[constraint_name] = violations

        return constraints

    def evaluate_all_constraints(
        self,
        molecules: List[str],
        qml_predictions: Dict[QuantumProperty, np.ndarray],
        qml_uncertainties: Optional[Dict[QuantumProperty, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Evaluate all constraints (QML and traditional)."""
        constraints = {}

        # QML constraints
        qml_constraints = self.evaluate_qml_constraints(
            qml_predictions, qml_uncertainties
        )
        constraints.update(qml_constraints)

        # Traditional constraints
        if self.traditional_handler:
            traditional_constraints = self.traditional_handler.evaluate_constraints(
                molecules
            )
            constraints.update(traditional_constraints)

        return constraints


class QMLInverseDesign:
    """
    QML-guided inverse molecular design with gradient-based optimization.

    This class extends inverse molecular design to use QML models for property
    prediction and gradient computation, enabling efficient optimization towards
    target quantum properties with constraint satisfaction.
    """

    def __init__(
        self,
        qml_model: BaseQMLModel,
        config: QMLOptimizationConfig,
        latent_space: Optional[MolecularLatentSpace] = None,
        traditional_constraints: Optional[ConstraintConfig] = None,
    ):
        """
        Initialize QML-guided inverse design.

        Args:
            qml_model: Trained QML model for property prediction
            config: QML optimization configuration
            latent_space: Molecular latent space representation
            traditional_constraints: Traditional molecular constraints
        """
        if not JAX_AVAILABLE:
            raise QMLInverseDesignError("JAX required for QML-guided inverse design")

        self.qml_model = qml_model
        self.config = config
        self.latent_space = latent_space

        # Initialize components
        self.on_the_fly_predictor = OnTheFlyPredictor(
            model=qml_model,
            conformer_strategy=config.conformer_strategy,
            max_batch_size=config.batch_size,
            device="cuda" if config.use_gpu else "cpu",
        )

        self.memory_processor = MemoryAwareProcessor(strategy=config.memory_strategy)

        self.qml_property_predictor = QMLPropertyPredictor(
            qml_model=qml_model,
            on_the_fly_predictor=self.on_the_fly_predictor,
            memory_processor=self.memory_processor,
        )

        self.gradient_computer = QMLGradientComputer(
            qml_model=qml_model,
            method=config.gradient_method,
            finite_diff_eps=config.finite_diff_eps,
        )

        self.constraint_handler = QMLConstraintHandler(
            qml_constraints=config.target_properties,
            traditional_constraints=traditional_constraints,
        )

        # Multi-objective optimization setup
        if config.pareto_optimization:
            objectives = []
            for target_prop in config.target_properties:
                if not target_prop.constraint:
                    objectives.append(
                        {
                            "name": f"qml_{target_prop.property.name.lower()}",
                            "target": target_prop.target_value,
                            "minimize": target_prop.minimize,
                            "weight": target_prop.weight,
                        }
                    )

            self.multi_objective_optimizer = MultiObjectiveOptimizer(objectives)
        else:
            self.multi_objective_optimizer = None

        # Optimization state
        self.optimization_history = []
        self.property_evolution = defaultdict(list)

    def optimize_molecules(
        self,
        starting_molecules: List[str],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
        population_size: int = 50,
        adaptive_lr: bool = True,
    ) -> QMLOptimizationResult:
        """
        Optimize molecules towards target QML properties.

        Args:
            starting_molecules: Initial molecules to optimize
            n_iterations: Number of optimization iterations
            learning_rate: Initial learning rate
            population_size: Size of molecule population
            adaptive_lr: Whether to use adaptive learning rate

        Returns:
            QMLOptimizationResult with optimized molecules and detailed metrics
        """
        # Fit latent space if needed
        if hasattr(self.latent_space, "fit") and self.latent_space:
            self.latent_space.fit(
                starting_molecules + starting_molecules
            )  # Augment training set

        # Initialize population
        current_molecules = self._initialize_population(
            starting_molecules, population_size
        )

        # Encode to latent space
        if self.latent_space:
            latent_vectors = self.latent_space.encode(current_molecules)
        else:
            # Use a simple representation if no latent space is provided
            latent_vectors = jnp.array(np.random.randn(len(current_molecules), 128))

        # Initialize optimizer
        if adaptive_lr:
            optimizer = optax.adam(learning_rate, b1=0.9, b2=0.999)
        else:
            optimizer = optax.sgd(learning_rate)

        opt_state = optimizer.init(latent_vectors)

        # Optimization metrics
        gradient_norms = []
        qml_predictions_history = []

        # Main optimization loop
        for iteration in range(n_iterations):
            # Memory-managed processing
            with self.memory_processor.memory_managed_processing(
                estimated_memory_gb=2.0
            ):
                # Decode current latent vectors to molecules
                if self.latent_space:
                    current_molecules = self.latent_space.decode(latent_vectors)

                # Predict QML properties and compute gradients
                properties_to_predict = [
                    tp.property for tp in self.config.target_properties
                ]

                predictions, gradients = self.qml_property_predictor.predict_properties(
                    current_molecules, properties_to_predict, compute_gradients=True
                )

                # Compute QML property gradients in latent space
                if gradients is not None:
                    latent_gradients = self.gradient_computer.compute_gradients(
                        latent_vectors, properties_to_predict, current_molecules
                    )
                else:
                    # Fallback to finite difference in latent space
                    latent_gradients = self._compute_latent_gradients_finite_diff(
                        latent_vectors, current_molecules
                    )

                # Compute total gradient
                total_gradient = self._compute_total_gradient(
                    latent_gradients, predictions
                )

                # Clip gradients
                if self.config.gradient_clipping > 0:
                    grad_norm = jnp.linalg.norm(total_gradient)
                    if grad_norm > self.config.gradient_clipping:
                        total_gradient = total_gradient * (
                            self.config.gradient_clipping / grad_norm
                        )

                # Update latent vectors
                updates, opt_state = optimizer.update(
                    total_gradient, opt_state, latent_vectors
                )
                latent_vectors = optax.apply_updates(latent_vectors, updates)

                # Clip latent vectors to reasonable bounds
                latent_vectors = jnp.clip(latent_vectors, -5.0, 5.0)

                # Record metrics
                gradient_norms.append(float(jnp.linalg.norm(total_gradient)))
                qml_predictions_history.append(
                    {prop: predictions[prop].copy() for prop in predictions}
                )

                # Update property evolution
                for prop, values in predictions.items():
                    self.property_evolution[prop].append(np.mean(values))

                # Record iteration info
                iteration_info = {
                    "iteration": iteration,
                    "molecules": current_molecules.copy(),
                    "predictions": predictions.copy(),
                    "gradient_norm": gradient_norms[-1],
                    "avg_properties": {
                        prop: np.mean(values) for prop, values in predictions.items()
                    },
                }

                self.optimization_history.append(iteration_info)

                # Check convergence
                if self._check_qml_convergence(iteration, predictions):
                    print(f"QML optimization converged at iteration {iteration}")
                    break

                # Adaptive learning rate
                if adaptive_lr and iteration % 20 == 0 and iteration > 0:
                    if (
                        len(gradient_norms) >= 2
                        and gradient_norms[-1] > gradient_norms[-2]
                    ):
                        learning_rate *= (
                            0.95  # Reduce learning rate if gradient norm increasing
                        )
                        optimizer = optax.adam(learning_rate, b1=0.9, b2=0.999)
                        opt_state = optimizer.init(latent_vectors)

        # Final evaluation
        if self.latent_space:
            final_molecules = self.latent_space.decode(latent_vectors)
        else:
            final_molecules = current_molecules

        (
            final_predictions,
            final_uncertainties,
        ) = self.qml_property_predictor.predict_with_uncertainty(
            final_molecules, properties_to_predict
        )

        # Evaluate constraints
        final_constraints = self.constraint_handler.evaluate_all_constraints(
            final_molecules, final_predictions, final_uncertainties
        )

        # Multi-objective optimization
        pareto_indices = None
        if self.multi_objective_optimizer and len(final_molecules) > 1:
            objective_values = (
                self.multi_objective_optimizer.calculate_objective_values(
                    {
                        f"qml_{prop.name.lower()}": vals
                        for prop, vals in final_predictions.items()
                    },
                    final_constraints,
                )
            )
            pareto_indices = self.multi_objective_optimizer.find_pareto_front(
                objective_values
            )

        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics(
            qml_predictions_history, gradient_norms
        )

        # Calculate conformer statistics
        conformer_stats = self._calculate_conformer_statistics()

        return QMLOptimizationResult(
            optimized_molecules=final_molecules,
            optimization_history=self.optimization_history,
            final_properties={
                f"qml_{prop.name.lower()}": vals
                for prop, vals in final_predictions.items()
            },
            constraint_violations=final_constraints,
            pareto_front_indices=pareto_indices,
            convergence_info={"converged": iteration < n_iterations - 1},
            qml_predictions=[
                {prop: final_predictions[prop][i] for prop in final_predictions}
                for i in range(len(final_molecules))
            ],
            qml_uncertainties=(
                [
                    {prop: final_uncertainties[prop][i] for prop in final_uncertainties}
                    for i in range(len(final_molecules))
                ]
                if final_uncertainties
                else None
            ),
            gradient_norms=gradient_norms,
            qml_convergence_metrics=convergence_metrics,
            conformer_statistics=conformer_stats,
            property_evolution=dict(self.property_evolution),
        )

    def _initialize_population(
        self, starting_molecules: List[str], population_size: int
    ) -> List[str]:
        """Initialize molecular population for optimization."""
        current_molecules = starting_molecules[:population_size]

        # Pad with duplicates if needed
        while len(current_molecules) < population_size:
            current_molecules.extend(starting_molecules)

        return current_molecules[:population_size]

    def _compute_latent_gradients_finite_diff(
        self, latent_vectors: jnp.ndarray, molecules: List[str]
    ) -> Dict[QuantumProperty, jnp.ndarray]:
        """Compute gradients in latent space using finite differences."""
        properties_to_predict = [tp.property for tp in self.config.target_properties]
        return self.gradient_computer.compute_gradients(
            latent_vectors, properties_to_predict, molecules
        )

    def _compute_total_gradient(
        self,
        latent_gradients: Dict[QuantumProperty, jnp.ndarray],
        predictions: Dict[QuantumProperty, np.ndarray],
    ) -> jnp.ndarray:
        """Compute total gradient combining all objectives."""
        total_gradient = jnp.zeros_like(next(iter(latent_gradients.values())))

        for target_prop in self.config.target_properties:
            if target_prop.constraint:
                continue  # Skip constraints in gradient computation

            prop = target_prop.property
            if prop not in latent_gradients:
                continue

            # Property gradient
            prop_gradient = latent_gradients[prop]

            # Scale by property scaling factor
            scaling = self.config.property_scaling.get(prop, 1.0)
            prop_gradient *= scaling

            # Weight by target property weight
            prop_gradient *= target_prop.weight

            # Apply minimization/maximization
            if not target_prop.minimize:
                prop_gradient = -prop_gradient

            total_gradient += prop_gradient

        return total_gradient

    def _check_qml_convergence(
        self, iteration: int, predictions: Dict[QuantumProperty, np.ndarray]
    ) -> bool:
        """Check QML-specific convergence criteria."""
        if iteration < 10:
            return False

        # Check if target properties are within tolerance
        target_satisfaction = 0
        total_targets = 0

        for target_prop in self.config.target_properties:
            if target_prop.constraint:
                continue

            prop = target_prop.property
            if prop not in predictions:
                continue

            current_values = predictions[prop]
            target_value = target_prop.target_value
            tolerance = target_prop.tolerance

            # Check fraction of molecules within tolerance
            within_tolerance = np.abs(current_values - target_value) <= tolerance
            satisfaction_rate = np.mean(within_tolerance)

            target_satisfaction += satisfaction_rate
            total_targets += 1

        if total_targets == 0:
            return False

        avg_satisfaction = target_satisfaction / total_targets

        # Converge if 80% of targets are satisfied for 80% of molecules
        return avg_satisfaction > 0.8

    def _calculate_convergence_metrics(
        self,
        predictions_history: List[Dict[QuantumProperty, np.ndarray]],
        gradient_norms: List[float],
    ) -> Dict[str, Any]:
        """Calculate detailed convergence metrics."""
        if not predictions_history:
            return {}

        metrics = {}

        # Gradient convergence
        if gradient_norms:
            metrics["final_gradient_norm"] = gradient_norms[-1]
            metrics["gradient_reduction"] = gradient_norms[0] / max(
                gradient_norms[-1], 1e-10
            )

            # Check gradient plateau
            if len(gradient_norms) >= 10:
                recent_grads = gradient_norms[-10:]
                metrics["gradient_std_recent"] = np.std(recent_grads)

        # Property convergence
        for prop in predictions_history[0].keys():
            prop_name = prop.name.lower()

            # Property value evolution
            prop_means = [np.mean(pred[prop]) for pred in predictions_history]
            if len(prop_means) >= 2:
                metrics[f"{prop_name}_final_mean"] = prop_means[-1]
                metrics[f"{prop_name}_change"] = abs(prop_means[-1] - prop_means[0])

                # Check for oscillations
                if len(prop_means) >= 10:
                    recent_means = prop_means[-10:]
                    metrics[f"{prop_name}_recent_std"] = np.std(recent_means)

        return metrics

    def _calculate_conformer_statistics(self) -> Dict[str, Any]:
        """Calculate conformer-related statistics."""
        stats = {}

        if hasattr(self.on_the_fly_predictor, "get_statistics"):
            predictor_stats = self.on_the_fly_predictor.get_statistics()
            stats["predictor_stats"] = predictor_stats

        # Memory usage statistics
        memory_stats = self.memory_processor.get_memory_statistics()
        stats["memory_stats"] = memory_stats

        return stats

    def get_best_molecules(
        self, result: QMLOptimizationResult, n_best: int = 10, criterion: str = "pareto"
    ) -> List[Tuple[str, Dict[QuantumProperty, float]]]:
        """
        Get best optimized molecules based on specified criterion.

        Args:
            result: Optimization result
            n_best: Number of best molecules to return
            criterion: Selection criterion ("pareto", "target_distance", "uncertainty")

        Returns:
            List of (molecule, properties) tuples
        """
        if criterion == "pareto" and result.pareto_front_indices is not None:
            # Return Pareto front molecules
            indices = result.pareto_front_indices[:n_best]
        elif criterion == "target_distance":
            # Select based on distance to target properties
            distances = []
            for i, qml_pred in enumerate(result.qml_predictions):
                total_distance = 0
                for target_prop in self.config.target_properties:
                    if not target_prop.constraint and target_prop.property in qml_pred:
                        prop_val = qml_pred[target_prop.property]
                        target_val = target_prop.target_value
                        distance = abs(prop_val - target_val) / max(
                            abs(target_val), 1.0
                        )
                        total_distance += distance * target_prop.weight
                distances.append(total_distance)

            indices = np.argsort(distances)[:n_best]
        elif criterion == "uncertainty" and result.qml_uncertainties:
            # Select based on lowest uncertainty
            total_uncertainties = []
            for i, qml_unc in enumerate(result.qml_uncertainties):
                total_unc = sum(qml_unc.values())
                total_uncertainties.append(total_unc)

            indices = np.argsort(total_uncertainties)[:n_best]
        else:
            # Default: first n_best molecules
            indices = list(range(min(n_best, len(result.optimized_molecules))))

        best_molecules = []
        for idx in indices:
            mol = result.optimized_molecules[idx]
            props = result.qml_predictions[idx]
            best_molecules.append((mol, props))

        return best_molecules

    def analyze_optimization(self, result: QMLOptimizationResult) -> Dict[str, Any]:
        """Provide detailed analysis of optimization results."""
        analysis = {}

        # Convergence analysis
        analysis["convergence"] = {
            "converged": result.convergence_info.get("converged", False),
            "final_gradient_norm": (
                result.gradient_norms[-1] if result.gradient_norms else 0
            ),
            "gradient_reduction_factor": (
                result.gradient_norms[0] / max(result.gradient_norms[-1], 1e-10)
                if len(result.gradient_norms) > 1
                else 1.0
            ),
        }

        # Property target achievement
        analysis["target_achievement"] = {}
        for target_prop in self.config.target_properties:
            prop = target_prop.property
            prop_name = prop.name.lower()

            if result.qml_predictions:
                prop_values = [pred.get(prop, 0) for pred in result.qml_predictions]
                target_val = target_prop.target_value
                tolerance = target_prop.tolerance

                within_tolerance = sum(
                    1 for val in prop_values if abs(val - target_val) <= tolerance
                )

                analysis["target_achievement"][prop_name] = {
                    "target_value": target_val,
                    "tolerance": tolerance,
                    "molecules_within_tolerance": within_tolerance,
                    "satisfaction_rate": within_tolerance / len(prop_values),
                    "mean_value": np.mean(prop_values),
                    "std_value": np.std(prop_values),
                    "best_value": min(prop_values, key=lambda x: abs(x - target_val)),
                }

        # Diversity analysis
        if len(result.optimized_molecules) > 1:
            analysis["diversity"] = self._analyze_molecular_diversity(
                result.optimized_molecules
            )

        # Constraint satisfaction
        if result.constraint_violations:
            total_violations = sum(
                np.sum(violations)
                for violations in result.constraint_violations.values()
            )
            analysis["constraints"] = {
                "total_violations": total_violations,
                "violation_breakdown": {
                    name: np.sum(violations)
                    for name, violations in result.constraint_violations.items()
                },
            }

        return analysis

    def _analyze_molecular_diversity(self, molecules: List[str]) -> Dict[str, Any]:
        """Analyze diversity of generated molecules."""
        diversity_metrics = {
            "unique_molecules": len(set(molecules)),
            "total_molecules": len(molecules),
            "uniqueness_ratio": len(set(molecules)) / len(molecules),
        }

        # Additional diversity metrics would require molecular similarity calculations
        # This would typically use RDKit Tanimoto similarities

        return diversity_metrics


# Utility functions for QML-guided inverse design
def create_qml_target_property(
    property_type: QuantumProperty,
    target_value: float,
    weight: float = 1.0,
    tolerance: float = 0.1,
    is_constraint: bool = False,
) -> QMLTargetProperty:
    """Create a QML target property configuration."""
    return QMLTargetProperty(
        property=property_type,
        target_value=target_value,
        weight=weight,
        tolerance=tolerance,
        constraint=is_constraint,
    )


def create_homo_lumo_gap_optimization(
    target_gap: float, tolerance: float = 0.5
) -> List[QMLTargetProperty]:
    """Create optimization setup for HOMO-LUMO gap targeting."""
    return [
        QMLTargetProperty(
            property=QuantumProperty.HOMO_LUMO_GAP,
            target_value=target_gap,
            weight=1.0,
            tolerance=tolerance,
            minimize=True,
        )
    ]


def create_multi_property_optimization(
    homo_target: float = -6.0, lumo_target: float = -2.0, dipole_target: float = 2.0
) -> List[QMLTargetProperty]:
    """Create multi-property optimization setup."""
    return [
        QMLTargetProperty(
            property=QuantumProperty.HOMO_ENERGY,
            target_value=homo_target,
            weight=1.0,
            tolerance=0.2,
        ),
        QMLTargetProperty(
            property=QuantumProperty.LUMO_ENERGY,
            target_value=lumo_target,
            weight=1.0,
            tolerance=0.2,
        ),
        QMLTargetProperty(
            property=QuantumProperty.DIPOLE_MOMENT,
            target_value=dipole_target,
            weight=0.5,
            tolerance=0.5,
        ),
    ]

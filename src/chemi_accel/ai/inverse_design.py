"""
Inverse Design and Molecular Optimization module for ChemiAccel.

This module provides gradient-based molecular optimization in latent space using JAX
automatic differentiation, with constraint handling for synthetic accessibility and
drug-likeness, and multi-objective optimization with Pareto front analysis.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import grad, jit, random, value_and_grad, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. Inverse design functionality will be limited.")

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    from rdkit.Chem.SA_Score import sascorer

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Some constraint functions will be disabled.")

from ..core.calculator import Calculator
from ..core.exceptions import ChemiAccelError

try:
    from .kg_integration import (
        AIContextualPrediction,
        AIKGIntegrationManager,
        AITaskType,
        KGContextType,
        KGUpdateSuggestion,
    )

    KG_INTEGRATION_AVAILABLE = True
except ImportError:
    KG_INTEGRATION_AVAILABLE = False
    warnings.warn("KG Integration not available. Contextual features will be limited.")


class InverseDesignError(ChemiAccelError):
    """Raised when inverse design operations fail."""

    pass


@dataclass
class OptimizationResult:
    """Results from molecular optimization."""

    optimized_molecules: List[str]
    optimization_history: List[Dict[str, Any]]
    final_properties: Dict[str, np.ndarray]
    constraint_violations: Dict[str, np.ndarray]
    pareto_front_indices: Optional[np.ndarray] = None
    convergence_info: Optional[Dict[str, Any]] = None


@dataclass
class ConstraintConfig:
    """Configuration for molecular constraints."""

    sa_threshold: float = 3.0
    drug_like: bool = True
    mw_range: Tuple[float, float] = (150.0, 500.0)
    logp_range: Tuple[float, float] = (-2.0, 5.0)
    tpsa_max: float = 140.0
    hbd_max: int = 5
    hba_max: int = 10
    rotatable_bonds_max: int = 10
    custom_constraints: Optional[List[Callable]] = None


class MolecularLatentSpace(ABC):
    """Abstract base class for molecular latent space representations."""

    @abstractmethod
    def encode(self, molecules: List[str]) -> jnp.ndarray:
        """Encode molecules to latent space."""
        pass

    @abstractmethod
    def decode(self, latent_vectors: jnp.ndarray) -> List[str]:
        """Decode latent vectors to molecules."""
        pass

    @abstractmethod
    def get_latent_dim(self) -> int:
        """Get dimensionality of latent space."""
        pass


class SimpleLatentSpace(MolecularLatentSpace):
    """
    Simple latent space implementation using molecular fingerprints.

    This is a placeholder implementation. In practice, you would use
    more sophisticated methods like VAE latent spaces or graph embeddings.
    """

    def __init__(self, latent_dim: int = 512):
        """Initialize simple latent space."""
        if not RDKIT_AVAILABLE:
            raise InverseDesignError("RDKit required for SimpleLatentSpace")

        self.latent_dim = latent_dim
        self.reference_molecules = []
        self.reference_fingerprints = None

    def fit(self, reference_molecules: List[str]):
        """Fit latent space to reference molecules."""
        self.reference_molecules = reference_molecules
        fingerprints = []

        for smiles in reference_molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = Chem.RDKFingerprint(mol, fpSize=self.latent_dim)
                fp_array = np.array(fp)
                fingerprints.append(fp_array)

        if fingerprints:
            self.reference_fingerprints = np.array(fingerprints)
        else:
            raise InverseDesignError("No valid molecules in reference set")

    def encode(self, molecules: List[str]) -> jnp.ndarray:
        """Encode molecules to latent space using fingerprints."""
        fingerprints = []

        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = Chem.RDKFingerprint(mol, fpSize=self.latent_dim)
                fp_array = np.array(fp, dtype=np.float32)
                fingerprints.append(fp_array)
            else:
                # Use zero vector for invalid molecules
                fingerprints.append(np.zeros(self.latent_dim, dtype=np.float32))

        return jnp.array(fingerprints)

    def decode(self, latent_vectors: jnp.ndarray) -> List[str]:
        """
        Decode latent vectors to molecules by finding nearest neighbors.

        This is a simplified approach. Real implementations would use
        generative models for proper decoding.
        """
        if self.reference_fingerprints is None:
            raise InverseDesignError("Latent space must be fitted before decoding")

        latent_np = np.array(latent_vectors)
        decoded_molecules = []

        for latent_vec in latent_np:
            # Find nearest neighbor in reference set
            distances = np.linalg.norm(self.reference_fingerprints - latent_vec, axis=1)
            nearest_idx = np.argmin(distances)
            decoded_molecules.append(self.reference_molecules[nearest_idx])

        return decoded_molecules

    def get_latent_dim(self) -> int:
        """Get dimensionality of latent space."""
        return self.latent_dim


class PropertyPredictor:
    """Property prediction interface for optimization."""

    def __init__(self, calculator: Calculator):
        """Initialize with ChemiAccel calculator."""
        self.calculator = calculator

    def predict_properties(self, molecules: List[str]) -> Dict[str, np.ndarray]:
        """Predict molecular properties."""
        try:
            # Convert to pandas Series for calculator
            mol_series = pd.Series(molecules)

            # Calculate descriptors
            results = self.calculator.pandas(mol_series)

            # Extract key properties
            properties = {
                "molecular_weight": results.get(
                    "molecular_weight", np.zeros(len(molecules))
                ),
                "logp": results.get("logp", np.zeros(len(molecules))),
                "tpsa": results.get("tpsa", np.zeros(len(molecules))),
                "num_rotatable_bonds": results.get(
                    "num_rotatable_bonds", np.zeros(len(molecules))
                ),
                "num_aromatic_rings": results.get(
                    "num_aromatic_rings", np.zeros(len(molecules))
                ),
            }

            return properties

        except Exception as e:
            # Fallback to RDKit if calculator fails
            if RDKIT_AVAILABLE:
                return self._rdkit_fallback(molecules)
            else:
                raise InverseDesignError(f"Property prediction failed: {e}")

    def _rdkit_fallback(self, molecules: List[str]) -> Dict[str, np.ndarray]:
        """Fallback property calculation using RDKit."""
        properties = {
            "molecular_weight": [],
            "logp": [],
            "tpsa": [],
            "num_rotatable_bonds": [],
            "num_aromatic_rings": [],
        }

        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                properties["molecular_weight"].append(Descriptors.MolWt(mol))
                properties["logp"].append(Crippen.MolLogP(mol))
                properties["tpsa"].append(Descriptors.TPSA(mol))
                properties["num_rotatable_bonds"].append(
                    Descriptors.NumRotatableBonds(mol)
                )
                properties["num_aromatic_rings"].append(
                    Descriptors.NumAromaticRings(mol)
                )
            else:
                # Use default values for invalid molecules
                properties["molecular_weight"].append(0.0)
                properties["logp"].append(0.0)
                properties["tpsa"].append(0.0)
                properties["num_rotatable_bonds"].append(0)
                properties["num_aromatic_rings"].append(0)

        return {k: np.array(v) for k, v in properties.items()}


class ConstraintHandler:
    """Handle molecular constraints during optimization."""

    def __init__(self, config: ConstraintConfig):
        """Initialize constraint handler."""
        self.config = config

    def evaluate_constraints(self, molecules: List[str]) -> Dict[str, np.ndarray]:
        """Evaluate all constraints for given molecules."""
        constraints = {}

        if not RDKIT_AVAILABLE:
            warnings.warn("RDKit not available. Constraint evaluation limited.")
            return constraints

        # Synthetic accessibility constraint
        if hasattr(self.config, "sa_threshold"):
            constraints["sa_score"] = self._evaluate_sa_constraint(molecules)

        # Drug-likeness constraints
        if self.config.drug_like:
            constraints.update(self._evaluate_drug_like_constraints(molecules))

        # Custom constraints
        if self.config.custom_constraints:
            for i, constraint_fn in enumerate(self.config.custom_constraints):
                constraints[f"custom_{i}"] = self._evaluate_custom_constraint(
                    molecules, constraint_fn
                )

        return constraints

    def _evaluate_sa_constraint(self, molecules: List[str]) -> np.ndarray:
        """Evaluate synthetic accessibility constraint."""
        sa_scores = []

        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    sa_score = sascorer.calculateScore(mol)
                    # Constraint violation: positive if SA score > threshold
                    violation = max(0, sa_score - self.config.sa_threshold)
                    sa_scores.append(violation)
                except:
                    sa_scores.append(10.0)  # High penalty for calculation failure
            else:
                sa_scores.append(10.0)  # High penalty for invalid molecules

        return np.array(sa_scores)

    def _evaluate_drug_like_constraints(
        self, molecules: List[str]
    ) -> Dict[str, np.ndarray]:
        """Evaluate drug-likeness constraints (Lipinski's Rule of Five)."""
        constraints = {
            "mw_violation": [],
            "logp_violation": [],
            "tpsa_violation": [],
            "hbd_violation": [],
            "hba_violation": [],
            "rotatable_bonds_violation": [],
        }

        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                rot_bonds = Descriptors.NumRotatableBonds(mol)

                # Calculate constraint violations
                constraints["mw_violation"].append(
                    max(0, mw - self.config.mw_range[1])
                    + max(0, self.config.mw_range[0] - mw)
                )
                constraints["logp_violation"].append(
                    max(0, logp - self.config.logp_range[1])
                    + max(0, self.config.logp_range[0] - logp)
                )
                constraints["tpsa_violation"].append(
                    max(0, tpsa - self.config.tpsa_max)
                )
                constraints["hbd_violation"].append(max(0, hbd - self.config.hbd_max))
                constraints["hba_violation"].append(max(0, hba - self.config.hba_max))
                constraints["rotatable_bonds_violation"].append(
                    max(0, rot_bonds - self.config.rotatable_bonds_max)
                )
            else:
                # High penalty for invalid molecules
                for key in constraints:
                    constraints[key].append(10.0)

        return {k: np.array(v) for k, v in constraints.items()}

    def _evaluate_custom_constraint(
        self, molecules: List[str], constraint_fn: Callable
    ) -> np.ndarray:
        """Evaluate custom constraint function."""
        violations = []

        for smiles in molecules:
            try:
                violation = constraint_fn(smiles)
                violations.append(max(0, violation))
            except:
                violations.append(10.0)  # High penalty for evaluation failure

        return np.array(violations)


class MultiObjectiveOptimizer:
    """Multi-objective optimization with Pareto front analysis."""

    def __init__(self, objectives: List[Dict[str, Any]]):
        """
        Initialize multi-objective optimizer.

        Args:
            objectives: List of objective dictionaries with keys:
                - 'name': objective name
                - 'target': target value (optional)
                - 'minimize': whether to minimize (default True)
                - 'weight': objective weight (default 1.0)
        """
        self.objectives = objectives
        self._validate_objectives()

    def _validate_objectives(self):
        """Validate objective specifications."""
        for obj in self.objectives:
            if "name" not in obj:
                raise InverseDesignError("Each objective must have a 'name' field")

            # Set defaults
            obj.setdefault("minimize", True)
            obj.setdefault("weight", 1.0)

    def calculate_objective_values(
        self, properties: Dict[str, np.ndarray], constraints: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Calculate objective values for multi-objective optimization."""
        n_molecules = len(next(iter(properties.values())))
        n_objectives = len(self.objectives)

        objective_values = np.zeros((n_molecules, n_objectives))

        for i, obj in enumerate(self.objectives):
            obj_name = obj["name"]
            minimize = obj["minimize"]
            weight = obj["weight"]

            if obj_name in properties:
                values = properties[obj_name]
            elif obj_name in constraints:
                values = constraints[obj_name]
            else:
                raise InverseDesignError(
                    f"Objective '{obj_name}' not found in properties or constraints"
                )

            # Apply target if specified
            if "target" in obj:
                target = obj["target"]
                values = np.abs(values - target)

            # Apply minimization/maximization
            if not minimize:
                values = -values

            # Apply weight
            values = weight * values

            objective_values[:, i] = values

        return objective_values

    def find_pareto_front(self, objective_values: np.ndarray) -> np.ndarray:
        """Find Pareto front using non-dominated sorting."""
        n_points = objective_values.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if np.all(objective_values[j] <= objective_values[i]) and np.any(
                        objective_values[j] < objective_values[i]
                    ):
                        is_pareto[i] = False
                        break

        return np.where(is_pareto)[0]

    def calculate_crowding_distance(
        self, objective_values: np.ndarray, pareto_indices: np.ndarray
    ) -> np.ndarray:
        """Calculate crowding distance for diversity preservation."""
        n_objectives = objective_values.shape[1]
        n_pareto = len(pareto_indices)

        if n_pareto <= 2:
            return np.full(n_pareto, float("inf"))

        crowding_distances = np.zeros(n_pareto)
        pareto_objectives = objective_values[pareto_indices]

        for obj_idx in range(n_objectives):
            # Sort by objective value
            sorted_indices = np.argsort(pareto_objectives[:, obj_idx])

            # Boundary points get infinite distance
            crowding_distances[sorted_indices[0]] = float("inf")
            crowding_distances[sorted_indices[-1]] = float("inf")

            # Calculate crowding distance for interior points
            obj_range = (
                pareto_objectives[sorted_indices[-1], obj_idx]
                - pareto_objectives[sorted_indices[0], obj_idx]
            )

            if obj_range > 0:
                for i in range(1, n_pareto - 1):
                    distance = (
                        pareto_objectives[sorted_indices[i + 1], obj_idx]
                        - pareto_objectives[sorted_indices[i - 1], obj_idx]
                    ) / obj_range
                    crowding_distances[sorted_indices[i]] += distance

        return crowding_distances


class KGContextualInverseDesign:
    """
    Knowledge Graph-enhanced Inverse Design for ChemiAccel.

    This class implements inverse molecular design using JAX automatic differentiation
    to optimize molecules in a latent space representation towards target properties
    while leveraging contextual information from the IC-KG for intelligent optimization
    and constraint handling.
    """

    def __init__(
        self,
        calculator: Calculator,
        target_properties: Dict[str, float],
        kg_integration_manager: Optional["AIKGIntegrationManager"] = None,
        latent_space: Optional[MolecularLatentSpace] = None,
        constraint_config: Optional[ConstraintConfig] = None,
        use_kg_constraints: bool = True,
    ):
        """
        Initialize KG-contextual inverse design module.

        Args:
            calculator: ChemiAccel calculator for property prediction
            target_properties: Dictionary of target property values
            kg_integration_manager: Manager for KG integration
            latent_space: Molecular latent space representation
            constraint_config: Configuration for molecular constraints
            use_kg_constraints: Whether to use KG-derived constraints
        """
        if not JAX_AVAILABLE:
            raise InverseDesignError("JAX required for inverse design")

        self.calculator = calculator
        self.target_properties = target_properties
        self.kg_integration_manager = kg_integration_manager
        self.latent_space = latent_space or SimpleLatentSpace()
        self.constraint_config = constraint_config or ConstraintConfig()
        self.use_kg_constraints = use_kg_constraints

        self.property_predictor = PropertyPredictor(calculator)
        self.constraint_handler = ConstraintHandler(self.constraint_config)

        # Multi-objective optimization setup
        objectives = []
        for prop_name, target_value in target_properties.items():
            objectives.append(
                {
                    "name": prop_name,
                    "target": target_value,
                    "minimize": True,  # Minimize distance to target
                    "weight": 1.0,
                }
            )

        self.multi_objective_optimizer = MultiObjectiveOptimizer(objectives)

        # Optimization state
        self.optimization_history = []
        self.best_molecules = []
        self.kg_contexts = {}
        self.kg_update_suggestions = []

    async def optimize_molecules_with_kg_context(
        self,
        starting_molecules: List[str],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
        population_size: int = 50,
        kg_context_types: Optional[List[KGContextType]] = None,
    ) -> OptimizationResult:
        """
        Generate optimized molecules using KG-contextual gradient-based optimization.

        Args:
            starting_molecules: Initial molecules to optimize
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for gradient descent
            population_size: Size of molecule population to maintain
            kg_context_types: Types of KG context to retrieve

        Returns:
            OptimizationResult with optimized molecules and optimization history
        """
        # Set default KG context types
        if kg_context_types is None:
            kg_context_types = [
                KGContextType.MOLECULAR_SIMILARITY,
                KGContextType.STRUCTURE_ACTIVITY,
                KGContextType.DRUG_TARGET,
                KGContextType.PROPERTY_CORRELATION,
            ]

        # Step 1: Get KG context for starting molecules
        if self.kg_integration_manager and self.use_kg_constraints:
            logging.info("Retrieving knowledge graph context...")
            self.kg_contexts = await self.kg_integration_manager.context_provider.get_molecular_context(
                starting_molecules, kg_context_types
            )

            # Update constraints based on KG context
            self._update_constraints_from_kg_context()

        # Step 2: Analyze similar molecules for guidance
        guidance_molecules = self._extract_guidance_molecules_from_kg()

        # Step 3: Perform optimization with KG guidance
        # Fit latent space if needed
        if hasattr(self.latent_space, "fit"):
            # Combine starting molecules with guidance molecules for better latent space
            all_molecules = starting_molecules + guidance_molecules
            self.latent_space.fit(all_molecules)

        # Initialize population in latent space
        current_molecules = starting_molecules[:population_size]
        if len(current_molecules) < population_size:
            # Add guidance molecules to reach population size
            remaining_slots = population_size - len(current_molecules)
            if guidance_molecules:
                guidance_sample = guidance_molecules[:remaining_slots]
                current_molecules.extend(guidance_sample)

            # Fill remaining slots with duplicates if needed
            while len(current_molecules) < population_size:
                current_molecules.extend(starting_molecules)
            current_molecules = current_molecules[:population_size]

        # Encode to latent space
        latent_vectors = self.latent_space.encode(current_molecules)

        # Initialize optimizer with KG-guided learning rate
        adjusted_lr = self._adjust_learning_rate_with_kg_context(learning_rate)
        optimizer = optax.adam(adjusted_lr)
        opt_state = optimizer.init(latent_vectors)

        # Optimization loop with KG guidance
        for iteration in range(n_iterations):
            # Calculate gradients and update with KG guidance
            (
                latent_vectors,
                opt_state,
                iteration_info,
            ) = await self._kg_guided_optimization_step(
                latent_vectors, opt_state, optimizer, iteration
            )

            # Decode to molecules
            current_molecules = self.latent_space.decode(latent_vectors)

            # Filter invalid molecules and replace with KG-guided alternatives
            current_molecules = self._filter_and_replace_molecules(current_molecules)

            # Record iteration with KG context
            iteration_record = {
                "iteration": iteration,
                "molecules": current_molecules.copy(),
                "kg_guidance_applied": len(guidance_molecules) > 0,
                **iteration_info,
            }
            self.optimization_history.append(iteration_record)

            # Early stopping check with KG-informed criteria
            if self._check_kg_informed_convergence(iteration_info, iteration):
                logging.info(f"Converged at iteration {iteration} with KG guidance")
                break

        # Step 4: Final evaluation with KG context
        final_properties = self.property_predictor.predict_properties(current_molecules)
        final_constraints = self.constraint_handler.evaluate_constraints(
            current_molecules
        )

        # Add KG-based constraints evaluation
        if self.kg_integration_manager:
            kg_constraint_violations = await self._evaluate_kg_constraints(
                current_molecules
            )
            final_constraints.update(kg_constraint_violations)

        # Multi-objective optimization
        objective_values = self.multi_objective_optimizer.calculate_objective_values(
            final_properties, final_constraints
        )
        pareto_indices = self.multi_objective_optimizer.find_pareto_front(
            objective_values
        )

        # Step 5: Generate contextual predictions and KG updates
        contextual_predictions = []
        if self.kg_integration_manager:
            for i, mol in enumerate(current_molecules):
                prediction = {
                    "optimized_molecules": [mol],
                    "target_properties": self.target_properties,
                    "optimization_score": (
                        float(np.mean(objective_values[i]))
                        if len(objective_values) > i
                        else 0.0
                    ),
                    "properties": {
                        k: v[i] for k, v in final_properties.items() if len(v) > i
                    },
                }

                contextual_pred = (
                    await self.kg_integration_manager.enrich_ai_prediction(
                        prediction=prediction,
                        molecules=[mol],
                        task_type=AITaskType.INVERSE_DESIGN,
                        context_types=kg_context_types,
                    )
                )
                contextual_predictions.append(contextual_pred)

            # Generate KG update suggestions
            self.kg_update_suggestions = (
                self.kg_integration_manager.feedback_manager.suggest_kg_updates(
                    contextual_predictions, AITaskType.INVERSE_DESIGN
                )
            )

        # Create enhanced optimization result
        result = OptimizationResult(
            optimized_molecules=current_molecules,
            optimization_history=self.optimization_history,
            final_properties=final_properties,
            constraint_violations=final_constraints,
            pareto_front_indices=pareto_indices,
            convergence_info={
                "converged": iteration < n_iterations - 1,
                "kg_guidance_used": len(guidance_molecules) > 0,
                "contextual_predictions": len(contextual_predictions),
            },
        )

        # Add KG-specific metadata
        if hasattr(result, "metadata"):
            result.metadata = {}
        result.metadata.update(
            {
                "kg_contexts_retrieved": len(self.kg_contexts),
                "guidance_molecules_used": len(guidance_molecules),
                "kg_update_suggestions": len(self.kg_update_suggestions),
                "kg_integration_enabled": self.kg_integration_manager is not None,
            }
        )

        return result

    @jit
    def _optimization_step(
        self, latent_vectors: jnp.ndarray, opt_state: Any, optimizer: Any
    ) -> Tuple[jnp.ndarray, Any, Dict]:
        """Single optimization step with JAX JIT compilation."""
        # Calculate loss and gradients
        loss_value, grads = value_and_grad(self._loss_function)(latent_vectors)

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, latent_vectors)
        latent_vectors = optax.apply_updates(latent_vectors, updates)

        # Clip to reasonable bounds
        latent_vectors = jnp.clip(latent_vectors, -5.0, 5.0)

        iteration_info = {
            "loss": float(loss_value),
            "grad_norm": float(jnp.linalg.norm(grads)),
        }

        return latent_vectors, opt_state, iteration_info

    def _loss_function(self, latent_vectors: jnp.ndarray) -> float:
        """
        Loss function for optimization.

        Note: This is a simplified implementation. In practice, you would need
        a differentiable property predictor and constraint evaluator.
        """
        # For now, return a simple regularization loss
        # In practice, this would include:
        # 1. Property prediction loss (distance to targets)
        # 2. Constraint violation penalties
        # 3. Regularization terms

        regularization_loss = jnp.sum(jnp.square(latent_vectors)) * 0.001

        return regularization_loss

    def _check_convergence(self, iteration_info: Dict) -> bool:
        """Check convergence criteria."""
        if len(self.optimization_history) < 10:
            return False

        # Check if loss has plateaued
        recent_losses = [info["loss"] for info in self.optimization_history[-10:]]
        loss_std = np.std(recent_losses)

        return loss_std < 1e-6

    def apply_constraints(
        self, sa_threshold: float = 3.0, drug_like: bool = True
    ) -> None:
        """
        Apply synthetic accessibility and drug-likeness constraints.

        Args:
            sa_threshold: Maximum allowed synthetic accessibility score
            drug_like: Whether to apply drug-likeness constraints
        """
        self.constraint_config.sa_threshold = sa_threshold
        self.constraint_config.drug_like = drug_like

        # Update constraint handler
        self.constraint_handler = ConstraintHandler(self.constraint_config)

    def get_pareto_front_molecules(self, result: OptimizationResult) -> List[str]:
        """Extract molecules on the Pareto front."""
        if result.pareto_front_indices is not None:
            return [result.optimized_molecules[i] for i in result.pareto_front_indices]
        else:
            return result.optimized_molecules

    def visualize_optimization_progress(
        self, result: OptimizationResult
    ) -> Dict[str, Any]:
        """Create visualization data for optimization progress."""
        iterations = [info["iteration"] for info in result.optimization_history]
        losses = [info["loss"] for info in result.optimization_history]
        grad_norms = [info["grad_norm"] for info in result.optimization_history]

        visualization_data = {
            "iterations": iterations,
            "losses": losses,
            "gradient_norms": grad_norms,
            "n_pareto_solutions": (
                len(result.pareto_front_indices)
                if result.pareto_front_indices is not None
                else 0
            ),
        }

        return visualization_data

    def _update_constraints_from_kg_context(self):
        """Update constraints based on knowledge graph context."""
        if not self.kg_contexts:
            return

        # Analyze KG context to derive additional constraints
        all_similar_mols = []
        for context in self.kg_contexts.values():
            similar_mols = [e for e in context.entities if "similarity_score" in e]
            all_similar_mols.extend(similar_mols)

        if all_similar_mols:
            # Adjust molecular weight range based on similar molecules
            mw_values = [
                mol.get("molecular_weight", 0)
                for mol in all_similar_mols
                if mol.get("molecular_weight")
            ]
            if mw_values:
                mw_mean = np.mean(mw_values)
                mw_std = np.std(mw_values)
                # Adjust range to be within 2 std devs of similar molecules
                new_mw_min = max(
                    self.constraint_config.mw_range[0], mw_mean - 2 * mw_std
                )
                new_mw_max = min(
                    self.constraint_config.mw_range[1], mw_mean + 2 * mw_std
                )
                self.constraint_config.mw_range = (new_mw_min, new_mw_max)
                logging.info(
                    f"Updated MW range based on KG: {new_mw_min:.1f}-{new_mw_max:.1f}"
                )

    def _extract_guidance_molecules_from_kg(self) -> List[str]:
        """Extract guidance molecules from KG context."""
        guidance_molecules = []

        for context in self.kg_contexts.values():
            # Extract similar molecules with high scores
            similar_mols = [
                e
                for e in context.entities
                if "similarity_score" in e
                and e["similarity_score"] > 0.7
                and "smiles" in e
            ]
            guidance_molecules.extend([mol["smiles"] for mol in similar_mols[:5]])

            # Extract molecules with known activities
            active_mols = [
                e
                for e in context.entities
                if "activity_type" in e
                and e.get("confidence", 0) > 0.6
                and "smiles" in e
            ]
            guidance_molecules.extend([mol["smiles"] for mol in active_mols[:3]])

        # Remove duplicates while preserving order
        seen = set()
        unique_guidance = []
        for mol in guidance_molecules:
            if mol not in seen:
                seen.add(mol)
                unique_guidance.append(mol)

        logging.info(f"Extracted {len(unique_guidance)} guidance molecules from KG")
        return unique_guidance[:20]  # Limit to top 20

    def _adjust_learning_rate_with_kg_context(self, base_lr: float) -> float:
        """Adjust learning rate based on KG context confidence."""
        if not self.kg_contexts:
            return base_lr

        # Calculate average KG confidence
        all_confidences = []
        for context in self.kg_contexts.values():
            confidences = list(context.confidence_scores.values())
            all_confidences.extend(confidences)

        if all_confidences:
            avg_confidence = np.mean(all_confidences)
            # Higher confidence -> can use higher learning rate
            adjusted_lr = base_lr * (0.5 + avg_confidence)
            logging.info(
                f"Adjusted learning rate from {base_lr} to {adjusted_lr:.4f} based on KG confidence"
            )
            return adjusted_lr

        return base_lr

    async def _kg_guided_optimization_step(
        self,
        latent_vectors: jnp.ndarray,
        opt_state: Any,
        optimizer: Any,
        iteration: int,
    ) -> Tuple[jnp.ndarray, Any, Dict]:
        """Optimization step with KG guidance."""
        # Calculate base loss and gradients
        loss_value, grads = value_and_grad(self._kg_informed_loss_function)(
            latent_vectors, iteration
        )

        # Apply KG-guided gradient modifications
        if self.kg_contexts:
            grads = self._apply_kg_gradient_guidance(grads, latent_vectors)

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, latent_vectors)
        latent_vectors = optax.apply_updates(latent_vectors, updates)

        # Clip to reasonable bounds
        latent_vectors = jnp.clip(latent_vectors, -5.0, 5.0)

        iteration_info = {
            "loss": float(loss_value),
            "grad_norm": float(jnp.linalg.norm(grads)),
            "kg_guidance_strength": self._calculate_kg_guidance_strength(),
        }

        return latent_vectors, opt_state, iteration_info

    def _kg_informed_loss_function(
        self, latent_vectors: jnp.ndarray, iteration: int
    ) -> float:
        """Loss function incorporating KG information."""
        # Base regularization loss
        regularization_loss = jnp.sum(jnp.square(latent_vectors)) * 0.001

        # Add KG-guided loss components
        kg_guidance_loss = 0.0
        if (
            self.kg_contexts and iteration > 10
        ):  # Start KG guidance after initial iterations
            # Encourage similarity to known actives
            kg_guidance_loss = self._calculate_kg_similarity_loss(latent_vectors)

        return regularization_loss + 0.1 * kg_guidance_loss

    def _calculate_kg_similarity_loss(self, latent_vectors: jnp.ndarray) -> float:
        """Calculate loss term to encourage similarity to KG molecules."""
        # This is a simplified implementation
        # In practice, would compare latent representations of current molecules
        # to those of KG-suggested molecules
        return 0.0  # Placeholder

    def _apply_kg_gradient_guidance(
        self, grads: jnp.ndarray, latent_vectors: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply KG-based guidance to gradients."""
        # Simplified gradient modification based on KG context
        # In practice, would analyze KG relationships to bias gradients
        guidance_strength = self._calculate_kg_guidance_strength()

        # Apply gentle guidance (not too strong to avoid local minima)
        guided_grads = grads * (1.0 + 0.1 * guidance_strength)

        return guided_grads

    def _calculate_kg_guidance_strength(self) -> float:
        """Calculate the strength of KG guidance to apply."""
        if not self.kg_contexts:
            return 0.0

        # Base strength on KG confidence and coverage
        total_entities = sum(
            len(context.entities) for context in self.kg_contexts.values()
        )
        avg_confidence = (
            np.mean(
                [
                    conf
                    for context in self.kg_contexts.values()
                    for conf in context.confidence_scores.values()
                ]
            )
            if self.kg_contexts
            else 0.0
        )

        # Normalize to [0, 1]
        coverage_score = min(
            1.0, total_entities / 20.0
        )  # Normalize by expected max entities

        return (coverage_score + avg_confidence) / 2.0

    def _filter_and_replace_molecules(self, molecules: List[str]) -> List[str]:
        """Filter invalid molecules and replace with KG-guided alternatives."""
        filtered = []
        guidance_molecules = self._extract_guidance_molecules_from_kg()

        for mol in molecules:
            if self._is_valid_molecule(mol):
                filtered.append(mol)
            else:
                # Replace with guidance molecule if available
                if guidance_molecules:
                    replacement = np.random.choice(guidance_molecules)
                    filtered.append(replacement)
                else:
                    # Keep original (will be handled by constraint evaluation)
                    filtered.append(mol)

        return filtered

    def _is_valid_molecule(self, smiles: str) -> bool:
        """Check if SMILES represents a valid molecule."""
        if not RDKIT_AVAILABLE:
            return len(smiles) > 0 and smiles.isascii()

        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def _check_kg_informed_convergence(
        self, iteration_info: Dict, iteration: int
    ) -> bool:
        """Check convergence with KG-informed criteria."""
        # Base convergence check
        base_converged = self._check_convergence(iteration_info)

        if base_converged:
            return True

        # Additional KG-based convergence criteria
        if len(self.optimization_history) >= 20:
            # Check if KG guidance strength has stabilized
            recent_guidance = [
                info.get("kg_guidance_strength", 0.0)
                for info in self.optimization_history[-10:]
            ]
            guidance_std = np.std(recent_guidance)

            # Converge if guidance has stabilized and loss is low
            if guidance_std < 0.01 and iteration_info.get("loss", float("inf")) < 0.001:
                return True

        return False

    async def _evaluate_kg_constraints(
        self, molecules: List[str]
    ) -> Dict[str, np.ndarray]:
        """Evaluate KG-based constraints."""
        kg_constraints = {}

        if not self.kg_integration_manager:
            return kg_constraints

        # Get fresh KG context for current molecules
        current_contexts = (
            await self.kg_integration_manager.context_provider.get_molecular_context(
                molecules, [KGContextType.DRUG_TARGET, KGContextType.ADVERSE_EFFECT]
            )
        )

        # Evaluate target selectivity constraints
        target_violations = []
        adverse_effect_violations = []

        for mol in molecules:
            context = current_contexts.get(mol)
            if context:
                # Check for known adverse effects
                adverse_effects = [
                    e
                    for e in context.entities
                    if "effect_type" in e
                    and e.get("severity", "").lower() in ["severe", "serious"]
                ]
                adverse_effect_violations.append(len(adverse_effects))

                # Check target selectivity (prefer selective targets)
                target_rels = [
                    r for r in context.relationships if r["type"] == "TARGETS"
                ]
                # Penalize molecules with too many targets (less selective)
                target_violations.append(max(0, len(target_rels) - 3))
            else:
                adverse_effect_violations.append(0.0)
                target_violations.append(0.0)

        kg_constraints["adverse_effect_risk"] = np.array(adverse_effect_violations)
        kg_constraints["target_selectivity"] = np.array(target_violations)

        return kg_constraints

    async def update_knowledge_graph(
        self, optimization_results: List[OptimizationResult]
    ) -> Dict[str, Any]:
        """Update knowledge graph based on optimization results."""
        if not self.kg_integration_manager:
            return {"error": "KG integration not available"}

        all_suggestions = []
        for result in optimization_results:
            all_suggestions.extend(getattr(result, "kg_update_suggestions", []))

        if all_suggestions:
            update_results = await self.kg_integration_manager.feedback_manager.validate_and_apply_updates(
                all_suggestions
            )
            return update_results

        return {"message": "No KG updates to apply"}


# Keep the original InverseDesign class for backward compatibility
class InverseDesign(KGContextualInverseDesign):
    """
    Backward-compatible Inverse Design class.

    This class maintains the original API while optionally supporting KG integration.
    """

    def __init__(
        self,
        calculator: Calculator,
        target_properties: Dict[str, float],
        latent_space: Optional[MolecularLatentSpace] = None,
        constraint_config: Optional[ConstraintConfig] = None,
    ):
        """Initialize inverse design module (backward compatible)."""
        super().__init__(
            calculator=calculator,
            target_properties=target_properties,
            kg_integration_manager=None,  # No KG integration by default
            latent_space=latent_space,
            constraint_config=constraint_config,
            use_kg_constraints=False,
        )

    def optimize_molecules(
        self,
        starting_molecules: List[str],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
        population_size: int = 50,
    ) -> OptimizationResult:
        """Original optimize_molecules method (synchronous, no KG)."""
        # Use asyncio to run the async method synchronously
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.optimize_molecules_with_kg_context(
                starting_molecules=starting_molecules,
                n_iterations=n_iterations,
                learning_rate=learning_rate,
                population_size=population_size,
                kg_context_types=[],  # No KG context for backward compatibility
            )
        )


# Utility functions for molecular optimization
def create_drug_like_constraints() -> ConstraintConfig:
    """Create standard drug-like constraint configuration."""
    return ConstraintConfig(
        sa_threshold=3.0,
        drug_like=True,
        mw_range=(150.0, 500.0),
        logp_range=(-2.0, 5.0),
        tpsa_max=140.0,
        hbd_max=5,
        hba_max=10,
        rotatable_bonds_max=10,
    )


def create_lead_like_constraints() -> ConstraintConfig:
    """Create lead-like constraint configuration."""
    return ConstraintConfig(
        sa_threshold=2.5,
        drug_like=True,
        mw_range=(200.0, 350.0),
        logp_range=(1.0, 3.0),
        tpsa_max=100.0,
        hbd_max=3,
        hba_max=6,
        rotatable_bonds_max=7,
    )


def create_fragment_like_constraints() -> ConstraintConfig:
    """Create fragment-like constraint configuration."""
    return ConstraintConfig(
        sa_threshold=2.0,
        drug_like=False,
        mw_range=(100.0, 250.0),
        logp_range=(-1.0, 2.5),
        tpsa_max=80.0,
        hbd_max=2,
        hba_max=4,
        rotatable_bonds_max=4,
    )

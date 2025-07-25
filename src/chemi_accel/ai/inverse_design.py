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


class InverseDesign:
    """
    Gradient-based molecular optimization in latent space.

    This class implements inverse molecular design using JAX automatic differentiation
    to optimize molecules in a latent space representation towards target properties
    while satisfying constraints.
    """

    def __init__(
        self,
        calculator: Calculator,
        target_properties: Dict[str, float],
        latent_space: Optional[MolecularLatentSpace] = None,
        constraint_config: Optional[ConstraintConfig] = None,
    ):
        """
        Initialize inverse design module.

        Args:
            calculator: ChemiAccel calculator for property prediction
            target_properties: Dictionary of target property values
            latent_space: Molecular latent space representation
            constraint_config: Configuration for molecular constraints
        """
        if not JAX_AVAILABLE:
            raise InverseDesignError("JAX required for inverse design")

        self.calculator = calculator
        self.target_properties = target_properties
        self.latent_space = latent_space or SimpleLatentSpace()
        self.constraint_config = constraint_config or ConstraintConfig()

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

    def optimize_molecules(
        self,
        starting_molecules: List[str],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
        population_size: int = 50,
    ) -> OptimizationResult:
        """
        Generate optimized molecules using gradient-based optimization.

        Args:
            starting_molecules: Initial molecules to optimize
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for gradient descent
            population_size: Size of molecule population to maintain

        Returns:
            OptimizationResult with optimized molecules and optimization history
        """
        # Fit latent space if needed
        if hasattr(self.latent_space, "fit"):
            self.latent_space.fit(starting_molecules)

        # Initialize population in latent space
        current_molecules = starting_molecules[:population_size]
        if len(current_molecules) < population_size:
            # Duplicate molecules to reach population size
            while len(current_molecules) < population_size:
                current_molecules.extend(starting_molecules)
            current_molecules = current_molecules[:population_size]

        # Encode to latent space
        latent_vectors = self.latent_space.encode(current_molecules)

        # Initialize optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(latent_vectors)

        # Optimization loop
        for iteration in range(n_iterations):
            # Calculate gradients and update
            latent_vectors, opt_state, iteration_info = self._optimization_step(
                latent_vectors, opt_state, optimizer
            )

            # Decode to molecules
            current_molecules = self.latent_space.decode(latent_vectors)

            # Record iteration
            self.optimization_history.append(
                {
                    "iteration": iteration,
                    "molecules": current_molecules.copy(),
                    **iteration_info,
                }
            )

            # Early stopping check
            if self._check_convergence(iteration_info):
                print(f"Converged at iteration {iteration}")
                break

        # Final evaluation
        final_properties = self.property_predictor.predict_properties(current_molecules)
        final_constraints = self.constraint_handler.evaluate_constraints(
            current_molecules
        )

        # Multi-objective optimization
        objective_values = self.multi_objective_optimizer.calculate_objective_values(
            final_properties, final_constraints
        )
        pareto_indices = self.multi_objective_optimizer.find_pareto_front(
            objective_values
        )

        return OptimizationResult(
            optimized_molecules=current_molecules,
            optimization_history=self.optimization_history,
            final_properties=final_properties,
            constraint_violations=final_constraints,
            pareto_front_indices=pareto_indices,
            convergence_info={"converged": iteration < n_iterations - 1},
        )

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
            "n_pareto_solutions": len(result.pareto_front_indices)
            if result.pareto_front_indices is not None
            else 0,
        }

        return visualization_data


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

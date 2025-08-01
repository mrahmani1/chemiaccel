"""
Transport Properties Inverse Design Optimizer

This module extends inverse design capabilities to optimize molecular transport
properties including membrane permeability, diffusion rates, and partitioning
behavior through AI-driven molecular modification.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import cupy as cp
    from cupyx.scipy import optimize as cupy_optimize

    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    cupy_optimize = None
    CUDA_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    jit = None
    vmap = None
    grad = None
    optax = None
    JAX_AVAILABLE = False

from ..ai.inverse_design import InverseDesigner
from ..core.exceptions import ChemiAccelError
from ..core.molecular import MolecularData
from .simulation_engine import MultiPhaseSimulator, TransportResult
from .transport_analyzer import TransportAnalyzer


@dataclass
class TransportTarget:
    """Target transport properties for optimization."""

    permeability_coefficient: Optional[float] = None
    permeability_range: Optional[Tuple[float, float]] = None
    diffusion_coefficients: Optional[Dict[str, float]] = None
    partition_coefficients: Optional[Dict[str, float]] = None
    selectivity_ratio: Optional[float] = None
    transport_mechanism: Optional[str] = None
    bioavailability_score: Optional[float] = None

    # Constraints
    molecular_weight_range: Tuple[float, float] = (100, 800)
    logp_range: Tuple[float, float] = (-2, 6)
    tpsa_range: Tuple[float, float] = (0, 200)

    # Optimization weights
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "permeability": 1.0,
            "diffusion": 0.8,
            "partition": 0.6,
            "selectivity": 0.4,
            "drug_likeness": 0.3,
        }
    )


@dataclass
class OptimizationResult:
    """Results from transport optimization."""

    optimized_molecules: List[str]
    predicted_properties: List[Dict[str, Any]]
    optimization_scores: np.ndarray
    convergence_history: np.ndarray
    modification_strategies: List[str]
    success_rate: float
    computational_time: float
    metadata: Dict[str, Any]


class TransportObjectiveFunction:
    """Objective function for transport property optimization."""

    def __init__(
        self,
        target: TransportTarget,
        simulator: MultiPhaseSimulator,
        analyzer: TransportAnalyzer,
        backend: str = "auto",
    ):
        self.target = target
        self.simulator = simulator
        self.analyzer = analyzer
        self.backend = backend
        self.logger = logging.getLogger(__name__)

        # Precompute normalization factors
        self._setup_normalization()

    def _setup_normalization(self):
        """Setup normalization factors for objective function components."""
        self.normalization = {
            "permeability": 1e-6,  # cm/s
            "diffusion": 1e-9,  # cm²/s
            "partition": 1.0,  # log units
            "selectivity": 1.0,  # ratio
            "molecular_weight": 500.0,  # Da
            "logp": 3.0,
            "tpsa": 100.0,  # Ų
        }

    def evaluate(self, molecules: List[str]) -> np.ndarray:
        """
        Evaluate objective function for list of molecules.

        Args:
            molecules: List of SMILES strings

        Returns:
            Array of objective function values (lower is better)
        """
        scores = []

        for mol_smiles in molecules:
            try:
                score = self._evaluate_single_molecule(mol_smiles)
                scores.append(score)
            except Exception as e:
                self.logger.warning(f"Failed to evaluate molecule {mol_smiles}: {e}")
                scores.append(float("inf"))  # Penalize failed evaluations

        return np.array(scores)

    def _evaluate_single_molecule(self, mol_smiles: str) -> float:
        """Evaluate objective function for a single molecule."""
        # Run transport simulation
        try:
            transport_result = self.simulator.simulate_membrane_transport([mol_smiles])
        except Exception:
            return float("inf")

        # Calculate objective function components
        score_components = {}

        # Permeability component
        if self.target.permeability_coefficient is not None:
            pred_perm = transport_result.permeability_coefficient
            target_perm = self.target.permeability_coefficient

            perm_error = (
                abs(pred_perm - target_perm) / self.normalization["permeability"]
            )
            score_components["permeability"] = (
                perm_error * self.target.weights["permeability"]
            )

        elif self.target.permeability_range is not None:
            pred_perm = transport_result.permeability_coefficient
            min_perm, max_perm = self.target.permeability_range

            if pred_perm < min_perm:
                perm_penalty = (min_perm - pred_perm) / self.normalization[
                    "permeability"
                ]
            elif pred_perm > max_perm:
                perm_penalty = (pred_perm - max_perm) / self.normalization[
                    "permeability"
                ]
            else:
                perm_penalty = 0.0

            score_components["permeability"] = (
                perm_penalty * self.target.weights["permeability"]
            )

        # Diffusion component
        if self.target.diffusion_coefficients:
            diffusion_score = 0.0
            for phase, target_diff in self.target.diffusion_coefficients.items():
                if phase in transport_result.diffusion_coefficients:
                    pred_diff = transport_result.diffusion_coefficients[phase]
                    diff_error = (
                        abs(pred_diff - target_diff) / self.normalization["diffusion"]
                    )
                    diffusion_score += diff_error

            score_components["diffusion"] = (
                diffusion_score * self.target.weights["diffusion"]
            )

        # Partition coefficient component
        if self.target.partition_coefficients:
            partition_score = 0.0
            for (
                partition_type,
                target_part,
            ) in self.target.partition_coefficients.items():
                if partition_type in transport_result.partition_coefficients:
                    pred_part = transport_result.partition_coefficients[partition_type]
                    part_error = (
                        abs(pred_part - target_part) / self.normalization["partition"]
                    )
                    partition_score += part_error

            score_components["partition"] = (
                partition_score * self.target.weights["partition"]
            )

        # Drug-likeness constraints
        drug_likeness_score = self._evaluate_drug_likeness(mol_smiles)
        score_components["drug_likeness"] = (
            drug_likeness_score * self.target.weights["drug_likeness"]
        )

        # Combine components
        total_score = sum(score_components.values())

        return total_score

    def _evaluate_drug_likeness(self, mol_smiles: str) -> float:
        """Evaluate drug-likeness constraints."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Crippen, Descriptors

            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                return float("inf")

            # Calculate molecular properties
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            penalty = 0.0

            # Molecular weight constraint
            mw_min, mw_max = self.target.molecular_weight_range
            if mw < mw_min or mw > mw_max:
                penalty += (
                    abs(mw - np.clip(mw, mw_min, mw_max))
                    / self.normalization["molecular_weight"]
                )

            # LogP constraint
            logp_min, logp_max = self.target.logp_range
            if logp < logp_min or logp > logp_max:
                penalty += (
                    abs(logp - np.clip(logp, logp_min, logp_max))
                    / self.normalization["logp"]
                )

            # TPSA constraint
            tpsa_min, tpsa_max = self.target.tpsa_range
            if tpsa < tpsa_min or tpsa > tpsa_max:
                penalty += (
                    abs(tpsa - np.clip(tpsa, tpsa_min, tpsa_max))
                    / self.normalization["tpsa"]
                )

            return penalty

        except ImportError:
            self.logger.warning("RDKit not available for drug-likeness evaluation")
            return 0.0
        except Exception:
            return float("inf")


class MolecularModificationStrategy(ABC):
    """Abstract base class for molecular modification strategies."""

    @abstractmethod
    def generate_modifications(
        self, mol_smiles: str, n_modifications: int = 10
    ) -> List[str]:
        """Generate molecular modifications."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class FunctionalGroupSubstitution(MolecularModificationStrategy):
    """Strategy for functional group substitution."""

    def __init__(self):
        self.substitution_patterns = [
            # Basic substitutions for transport optimization
            ("[OH]", "[F]"),  # OH -> F (reduce H-bonding)
            ("[NH2]", "[CH3]"),  # NH2 -> CH3 (reduce polarity)
            ("[COOH]", "[COOMe]"),  # COOH -> COOMe (esterification)
            ("[H]", "[CH3]"),  # H -> CH3 (increase lipophilicity)
            ("[CH3]", "[CF3]"),  # CH3 -> CF3 (modulate electronics)
        ]

    def generate_modifications(
        self, mol_smiles: str, n_modifications: int = 10
    ) -> List[str]:
        """Generate modifications through functional group substitution."""
        try:
            from rdkit import Chem
        except ImportError:
            return [
                mol_smiles
            ] * n_modifications  # Return original if RDKit unavailable

        modifications = []

        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            return [mol_smiles] * n_modifications

        # Apply substitution patterns
        for old_group, new_group in self.substitution_patterns:
            try:
                # Simple pattern-based substitution
                new_smiles = mol_smiles.replace(old_group, new_group)
                if (
                    new_smiles != mol_smiles
                    and Chem.MolFromSmiles(new_smiles) is not None
                ):
                    modifications.append(new_smiles)
            except Exception:
                continue

        # Pad with random modifications if needed
        while len(modifications) < n_modifications:
            # Add slight random variations
            modifications.append(mol_smiles)  # Simplified

        return modifications[:n_modifications]

    def get_strategy_name(self) -> str:
        return "functional_group_substitution"


class BioisostereReplacement(MolecularModificationStrategy):
    """Strategy for bioisostere replacement."""

    def __init__(self):
        self.bioisostere_pairs = [
            # Common bioisosteric replacements
            ("c1ccccc1", "c1ccncc1"),  # Benzene -> Pyridine
            ("CC(=O)", "CS(=O)(=O)"),  # Ketone -> Sulfonyl
            ("C(=O)O", "C(=O)N"),  # Carboxylic acid -> Amide
            ("O", "S"),  # Oxygen -> Sulfur
            ("N", "CH"),  # Nitrogen -> CH
        ]

    def generate_modifications(
        self, mol_smiles: str, n_modifications: int = 10
    ) -> List[str]:
        """Generate modifications through bioisostere replacement."""
        try:
            from rdkit import Chem
        except ImportError:
            return [mol_smiles] * n_modifications

        modifications = []

        # Apply bioisosteric replacements
        for old_pattern, new_pattern in self.bioisostere_pairs:
            try:
                new_smiles = mol_smiles.replace(old_pattern, new_pattern)
                if (
                    new_smiles != mol_smiles
                    and Chem.MolFromSmiles(new_smiles) is not None
                ):
                    modifications.append(new_smiles)
            except Exception:
                continue

        # Pad to requested number
        while len(modifications) < n_modifications:
            modifications.append(mol_smiles)

        return modifications[:n_modifications]

    def get_strategy_name(self) -> str:
        return "bioisostere_replacement"


class ConformationalOptimization(MolecularModificationStrategy):
    """Strategy for conformational optimization."""

    def generate_modifications(
        self, mol_smiles: str, n_modifications: int = 10
    ) -> List[str]:
        """Generate conformational variants (simplified)."""
        # In practice, this would generate different conformers
        # For now, return slight variations
        modifications = []

        for i in range(n_modifications):
            # Simplified: just return original molecule
            # In practice: generate different conformers or stereoisomers
            modifications.append(mol_smiles)

        return modifications

    def get_strategy_name(self) -> str:
        return "conformational_optimization"


class TransportOptimizer:
    """
    Inverse design optimizer for transport properties.

    This class extends inverse design capabilities to optimize molecular
    transport properties through AI-driven molecular modification.
    """

    def __init__(self, backend: str = "auto", simulation_method: str = "ai_surrogate"):
        """
        Initialize the transport optimizer.

        Args:
            backend: Computational backend ('cuda', 'jax', 'cpu', 'auto')
            simulation_method: Simulation method for transport evaluation
        """
        self.backend = self._select_backend(backend)
        self.simulation_method = simulation_method
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.simulator = MultiPhaseSimulator(backend, simulation_method)
        self.analyzer = TransportAnalyzer(backend)

        # Initialize modification strategies
        self.modification_strategies = [
            FunctionalGroupSubstitution(),
            BioisostereReplacement(),
            ConformationalOptimization(),
        ]

    def _select_backend(self, backend: str) -> str:
        """Select optimal computational backend."""
        if backend == "auto":
            if CUDA_AVAILABLE:
                return "cuda"
            elif JAX_AVAILABLE:
                return "jax"
            else:
                return "cpu"
        return backend

    def optimize_permeability(
        self,
        initial_molecules: List[str],
        target_permeability: float,
        permeability_range: Optional[Tuple[float, float]] = None,
        n_iterations: int = 100,
        population_size: int = 50,
    ) -> OptimizationResult:
        """
        Optimize molecular permeability through inverse design.

        Args:
            initial_molecules: Starting molecules for optimization
            target_permeability: Target permeability coefficient (cm/s)
            permeability_range: Acceptable permeability range
            n_iterations: Number of optimization iterations
            population_size: Population size for optimization

        Returns:
            OptimizationResult: Comprehensive optimization results
        """
        # Setup target
        target = TransportTarget(
            permeability_coefficient=target_permeability,
            permeability_range=permeability_range,
        )

        self.logger.info(
            f"Starting permeability optimization for {len(initial_molecules)} molecules"
        )

        return self._run_optimization(
            initial_molecules, target, n_iterations, population_size
        )

    def optimize_transport_profile(
        self,
        initial_molecules: List[str],
        transport_target: TransportTarget,
        n_iterations: int = 100,
        population_size: int = 50,
    ) -> OptimizationResult:
        """
        Optimize complete transport profile including permeability, diffusion, and partitioning.

        Args:
            initial_molecules: Starting molecules for optimization
            transport_target: Comprehensive transport target properties
            n_iterations: Number of optimization iterations
            population_size: Population size for optimization

        Returns:
            OptimizationResult: Comprehensive optimization results
        """
        self.logger.info("Starting comprehensive transport profile optimization")

        return self._run_optimization(
            initial_molecules, transport_target, n_iterations, population_size
        )

    def optimize_bioavailability(
        self,
        initial_molecules: List[str],
        target_bioavailability: float,
        constraints: Optional[Dict[str, Any]] = None,
        n_iterations: int = 100,
        population_size: int = 50,
    ) -> OptimizationResult:
        """
        Optimize molecular bioavailability through transport property optimization.

        Args:
            initial_molecules: Starting molecules for optimization
            target_bioavailability: Target bioavailability score (0-1)
            constraints: Additional molecular constraints
            n_iterations: Number of optimization iterations
            population_size: Population size for optimization

        Returns:
            OptimizationResult: Bioavailability optimization results
        """
        # Setup bioavailability-focused target
        target = TransportTarget(
            bioavailability_score=target_bioavailability,
            permeability_range=(1e-7, 1e-5),  # Reasonable permeability range
            partition_coefficients={"water_octanol": 2.0},  # Moderate lipophilicity
        )

        # Apply constraints if provided
        if constraints:
            if "molecular_weight_range" in constraints:
                target.molecular_weight_range = constraints["molecular_weight_range"]
            if "logp_range" in constraints:
                target.logp_range = constraints["logp_range"]
            if "tpsa_range" in constraints:
                target.tpsa_range = constraints["tpsa_range"]

        self.logger.info("Starting bioavailability optimization")

        return self._run_optimization(
            initial_molecules, target, n_iterations, population_size
        )

    def _run_optimization(
        self,
        initial_molecules: List[str],
        target: TransportTarget,
        n_iterations: int,
        population_size: int,
    ) -> OptimizationResult:
        """Run the optimization algorithm."""
        start_time = np.time.time()

        # Initialize objective function
        objective_func = TransportObjectiveFunction(
            target, self.simulator, self.analyzer, self.backend
        )

        # Initialize population
        population = self._initialize_population(initial_molecules, population_size)

        # Optimization history
        convergence_history = []
        best_scores = []

        # Run optimization iterations
        for iteration in range(n_iterations):
            # Evaluate population
            scores = objective_func.evaluate(population)

            # Track progress
            best_score = np.min(scores)
            best_scores.append(best_score)
            convergence_history.append(scores.copy())

            if iteration % 10 == 0:
                self.logger.info(
                    f"Iteration {iteration}: Best score = {best_score:.6f}"
                )

            # Generate next generation
            population = self._evolve_population(population, scores, target)

            # Early stopping if converged
            if len(best_scores) > 20:
                recent_improvement = best_scores[-20] - best_score
                if recent_improvement < 1e-6:
                    self.logger.info(f"Converged after {iteration} iterations")
                    break

        # Final evaluation
        final_scores = objective_func.evaluate(population)

        # Get best molecules
        best_indices = np.argsort(final_scores)[:10]  # Top 10
        optimized_molecules = [population[i] for i in best_indices]

        # Predict properties for best molecules
        predicted_properties = []
        for mol in optimized_molecules:
            try:
                transport_result = self.simulator.simulate_membrane_transport([mol])
                properties = {
                    "permeability_coefficient": transport_result.permeability_coefficient,
                    "diffusion_coefficients": transport_result.diffusion_coefficients,
                    "partition_coefficients": transport_result.partition_coefficients,
                    "smiles": mol,
                }
                predicted_properties.append(properties)
            except Exception as e:
                self.logger.warning(f"Failed to predict properties for {mol}: {e}")
                predicted_properties.append({"smiles": mol, "error": str(e)})

        # Calculate success rate
        valid_scores = final_scores[~np.isinf(final_scores)]
        success_rate = (
            len(valid_scores) / len(final_scores) if len(final_scores) > 0 else 0.0
        )

        computational_time = np.time.time() - start_time

        # Get modification strategies used
        modification_strategies = [
            strategy.get_strategy_name() for strategy in self.modification_strategies
        ]

        metadata = {
            "target": target,
            "n_iterations": n_iterations,
            "population_size": population_size,
            "backend": self.backend,
            "simulation_method": self.simulation_method,
            "initial_molecules": initial_molecules,
        }

        return OptimizationResult(
            optimized_molecules=optimized_molecules,
            predicted_properties=predicted_properties,
            optimization_scores=final_scores[best_indices],
            convergence_history=np.array(convergence_history),
            modification_strategies=modification_strategies,
            success_rate=success_rate,
            computational_time=computational_time,
            metadata=metadata,
        )

    def _initialize_population(
        self, initial_molecules: List[str], population_size: int
    ) -> List[str]:
        """Initialize optimization population."""
        population = []

        # Add initial molecules
        population.extend(initial_molecules)

        # Generate variations to fill population
        while len(population) < population_size:
            # Select random initial molecule
            base_mol = np.random.choice(initial_molecules)

            # Apply random modification strategy
            strategy = np.random.choice(self.modification_strategies)
            modifications = strategy.generate_modifications(base_mol, 1)

            if modifications:
                population.append(modifications[0])
            else:
                population.append(base_mol)

        return population[:population_size]

    def _evolve_population(
        self, population: List[str], scores: np.ndarray, target: TransportTarget
    ) -> List[str]:
        """Evolve population for next generation."""
        population_size = len(population)
        new_population = []

        # Selection: keep best performers
        elite_size = max(1, population_size // 10)
        elite_indices = np.argsort(scores)[:elite_size]
        elite_molecules = [population[i] for i in elite_indices]
        new_population.extend(elite_molecules)

        # Generate rest of population through modifications
        while len(new_population) < population_size:
            # Tournament selection
            parent = self._tournament_selection(population, scores, tournament_size=3)

            # Apply modification strategy
            strategy = np.random.choice(self.modification_strategies)
            modifications = strategy.generate_modifications(parent, 1)

            if modifications:
                new_population.append(modifications[0])
            else:
                new_population.append(parent)

        return new_population[:population_size]

    def _tournament_selection(
        self, population: List[str], scores: np.ndarray, tournament_size: int = 3
    ) -> str:
        """Tournament selection for parent molecules."""
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        tournament_scores = scores[tournament_indices]

        # Select best from tournament (lowest score)
        winner_idx = tournament_indices[np.argmin(tournament_scores)]

        return population[winner_idx]

    def analyze_optimization_results(
        self, result: OptimizationResult
    ) -> Dict[str, Any]:
        """
        Analyze optimization results and provide insights.

        Args:
            result: Optimization result to analyze

        Returns:
            Dictionary containing analysis and insights
        """
        analysis = {}

        # Convergence analysis
        if len(result.convergence_history) > 0:
            best_scores_over_time = [
                np.min(scores) for scores in result.convergence_history
            ]

            analysis["convergence"] = {
                "final_best_score": best_scores_over_time[-1],
                "initial_best_score": best_scores_over_time[0],
                "improvement": best_scores_over_time[0] - best_scores_over_time[-1],
                "converged": len(best_scores_over_time) > 20
                and (best_scores_over_time[-20] - best_scores_over_time[-1]) < 1e-6,
            }

        # Property analysis
        if result.predicted_properties:
            permeabilities = [
                p.get("permeability_coefficient", 0)
                for p in result.predicted_properties
                if "error" not in p
            ]

            if permeabilities:
                analysis["properties"] = {
                    "permeability_range": (min(permeabilities), max(permeabilities)),
                    "mean_permeability": np.mean(permeabilities),
                    "permeability_std": np.std(permeabilities),
                }

        # Success analysis
        analysis["success_metrics"] = {
            "success_rate": result.success_rate,
            "n_valid_molecules": len([m for m in result.optimized_molecules if m]),
            "computational_efficiency": len(result.optimized_molecules)
            / result.computational_time,
        }

        # Strategy effectiveness
        analysis["strategy_analysis"] = {
            "strategies_used": result.modification_strategies,
            "recommended_follow_up": self._recommend_follow_up_strategies(result),
        }

        return analysis

    def _recommend_follow_up_strategies(self, result: OptimizationResult) -> List[str]:
        """Recommend follow-up optimization strategies."""
        recommendations = []

        if result.success_rate < 0.5:
            recommendations.append(
                "Consider loosening constraints or using different modification strategies"
            )

        if result.computational_time > 300:  # 5 minutes
            recommendations.append(
                "Consider using faster surrogate models or reducing population size"
            )

        if len(result.convergence_history) > 0:
            final_scores = result.convergence_history[-1]
            if np.std(final_scores) > np.mean(final_scores):
                recommendations.append(
                    "High score variance - consider diversification strategies"
                )

        return recommendations

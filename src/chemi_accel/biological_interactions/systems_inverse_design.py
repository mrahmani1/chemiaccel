"""
Systems-Level Inverse Design Module for ChemiAccel.

This module provides comprehensive inverse design capabilities for optimizing
molecules based on systems-level biological profiles. It focuses on designing
compounds for desired polypharmacology profiles and minimizing system-level
adverse effects using advanced optimization algorithms and multi-objective
design strategies.

Features:
- Polypharmacology profile optimization with target selectivity control
- Systems-level adverse effects minimization through multi-constraint optimization
- Multi-objective molecular design with Pareto-optimal solutions
- Evolutionary algorithms for molecular space exploration
- Reinforcement learning-based molecular generation
- Bayesian optimization for efficient design space exploration
- Constraint satisfaction for complex biological requirements
- Interactive design workflows with real-time feedback
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pad_sequence

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some features will be limited.")

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Molecular operations will be limited.")

try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern

    _SCIPY_SKLEARN_AVAILABLE = True
except ImportError:
    _SCIPY_SKLEARN_AVAILABLE = False
    warnings.warn(
        "SciPy/scikit-learn not available. Some optimization features will be limited."
    )

from .multi_target_prediction import (
    BiologicalTarget,
    ConfidenceLevel,
    InteractionPrediction,
    MultiTargetInteractionPredictor,
    TargetType,
)
from .toxicology_prediction import (
    RiskLevel,
    SystemsToxicologyEngine,
    SystemsToxicologyProfile,
    ToxicityEndpoint,
)


class OptimizationObjective(Enum):
    """Optimization objectives for systems-level design."""

    MAXIMIZE_EFFICACY = "maximize_efficacy"
    MINIMIZE_TOXICITY = "minimize_toxicity"
    OPTIMIZE_SELECTIVITY = "optimize_selectivity"
    BALANCE_POLYPHARMACOLOGY = "balance_polypharmacology"
    MINIMIZE_OFF_TARGETS = "minimize_off_targets"
    MAXIMIZE_THERAPEUTIC_INDEX = "maximize_therapeutic_index"
    OPTIMIZE_ADMET = "optimize_admet"
    MINIMIZE_DRUG_INTERACTIONS = "minimize_drug_interactions"


class DesignStrategy(Enum):
    """Design strategies for molecular optimization."""

    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    GRADIENT_DESCENT = "gradient_descent"
    MULTI_OBJECTIVE_EVOLUTIONARY = "multi_objective_evolutionary"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"


class MolecularConstraint(Enum):
    """Types of molecular constraints."""

    MOLECULAR_WEIGHT = "molecular_weight"
    LOGP = "logp"
    TPSA = "tpsa"
    HBD_COUNT = "hbd_count"
    HBA_COUNT = "hba_count"
    ROTATABLE_BONDS = "rotatable_bonds"
    AROMATIC_RINGS = "aromatic_rings"
    SYNTHETIC_ACCESSIBILITY = "synthetic_accessibility"
    DRUG_LIKENESS = "drug_likeness"
    CUSTOM_PROPERTY = "custom_property"


@dataclass
class PolypharmacologyProfile:
    """Definition of desired polypharmacology profile."""

    target_engagements: Dict[str, float]  # target_id -> desired engagement (0-1)
    selectivity_constraints: Dict[
        str, Tuple[float, float]
    ]  # target_id -> (min, max) selectivity
    off_target_limits: Dict[str, float]  # target_id -> max allowed engagement
    pathway_modulations: Dict[str, float]  # pathway_id -> desired modulation (-1 to 1)
    therapeutic_window: float = 10.0  # desired therapeutic index
    confidence_threshold: float = 0.7

    def __post_init__(self):
        """Validate profile parameters."""
        for engagement in self.target_engagements.values():
            if not 0 <= engagement <= 1:
                raise ValueError("Target engagements must be between 0 and 1")

        if self.therapeutic_window <= 1.0:
            raise ValueError("Therapeutic window must be > 1.0")


@dataclass
class AdverseEffectsConstraints:
    """Constraints for minimizing adverse effects."""

    toxicity_limits: Dict[
        ToxicityEndpoint, float
    ]  # endpoint -> max allowed probability
    adr_restrictions: Dict[str, float]  # system -> max ADR probability
    off_target_penalties: Dict[TargetType, float]  # target type -> penalty weight
    safety_score_threshold: float = 0.8  # minimum overall safety score
    contraindication_avoidance: List[str] = field(default_factory=list)
    patient_population_factors: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate constraints."""
        for prob in self.toxicity_limits.values():
            if not 0 <= prob <= 1:
                raise ValueError("Toxicity limits must be between 0 and 1")

        if not 0 <= self.safety_score_threshold <= 1:
            raise ValueError("Safety score threshold must be between 0 and 1")


@dataclass
class DesignConstraints:
    """Comprehensive design constraints."""

    molecular_constraints: Dict[
        MolecularConstraint, Tuple[float, float]
    ]  # constraint -> (min, max)
    structural_requirements: List[str] = field(
        default_factory=list
    )  # SMARTS patterns required
    structural_exclusions: List[str] = field(
        default_factory=list
    )  # SMARTS patterns to avoid
    synthetic_feasibility_threshold: float = 0.6
    novelty_requirements: Dict[str, float] = field(
        default_factory=dict
    )  # similarity thresholds
    custom_filters: List[Callable[[str], bool]] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of molecular optimization."""

    optimized_smiles: str
    optimization_score: float
    predicted_profile: Dict[str, Any]
    constraint_satisfaction: Dict[str, bool]
    pareto_rank: Optional[int] = None
    generation_history: List[str] = field(default_factory=list)
    optimization_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    computational_cost: float = 0.0

    def get_profile_match_score(self, target_profile: PolypharmacologyProfile) -> float:
        """Calculate how well the result matches the target profile."""
        if "target_engagements" not in self.predicted_profile:
            return 0.0

        predicted = self.predicted_profile["target_engagements"]
        target = target_profile.target_engagements

        score = 0.0
        count = 0

        for target_id, desired_engagement in target.items():
            if target_id in predicted:
                # Calculate similarity using Gaussian kernel
                diff = abs(predicted[target_id] - desired_engagement)
                similarity = np.exp(-(diff**2) / 0.1)  # sigma = 0.1
                score += similarity
                count += 1

        return score / count if count > 0 else 0.0


class SystemsInverseDesigner:
    """
    Advanced inverse molecular design system for systems-level optimization.

    This class provides comprehensive tools for designing molecules with desired
    polypharmacology profiles while minimizing adverse effects through multi-
    objective optimization and constraint satisfaction.
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize the systems inverse designer.

        Args:
            use_gpu: Whether to use GPU acceleration for computations
        """
        self.use_gpu = use_gpu and _CUPY_AVAILABLE

        # Initialize prediction engines
        self.interaction_predictor = MultiTargetInteractionPredictor(use_gpu=use_gpu)
        self.toxicology_engine = SystemsToxicologyEngine(use_gpu=use_gpu)

        # Initialize optimization components
        self._setup_optimization_engines()
        self._setup_molecular_generators()
        self._setup_constraint_evaluators()

        # Design history and caching
        self.design_history = []
        self.evaluation_cache = {}
        self.pareto_front = []

    def _setup_optimization_engines(self):
        """Setup optimization engines."""
        self.optimization_engines = {
            DesignStrategy.GENETIC_ALGORITHM: self._genetic_algorithm_optimizer,
            DesignStrategy.BAYESIAN_OPTIMIZATION: self._bayesian_optimizer,
            DesignStrategy.REINFORCEMENT_LEARNING: self._rl_optimizer,
            DesignStrategy.SIMULATED_ANNEALING: self._simulated_annealing_optimizer,
            DesignStrategy.MULTI_OBJECTIVE_EVOLUTIONARY: self._moea_optimizer,
        }

        # Initialize Bayesian optimization components
        if _SCIPY_SKLEARN_AVAILABLE:
            self.gp_model = GaussianProcessRegressor(
                kernel=RBF(length_scale=1.0) * Matern(length_scale=1.0, nu=2.5),
                alpha=1e-6,
                normalize_y=True,
            )

        # Initialize RL components if PyTorch is available
        if _TORCH_AVAILABLE:
            self.rl_policy = self._create_rl_policy()
            self.rl_optimizer = optim.Adam(self.rl_policy.parameters(), lr=1e-4)

    def _setup_molecular_generators(self):
        """Setup molecular generation and modification tools."""
        self.generators = {
            "fragment_based": self._fragment_based_generator,
            "scaffold_hopping": self._scaffold_hopping_generator,
            "side_chain_modification": self._side_chain_modifier,
            "ring_replacement": self._ring_replacement_generator,
            "functional_group_addition": self._functional_group_adder,
        }

        # Load molecular fragments and scaffolds
        self.molecular_fragments = self._load_molecular_fragments()
        self.scaffold_library = self._load_scaffold_library()
        self.functional_groups = self._load_functional_groups()

    def _setup_constraint_evaluators(self):
        """Setup constraint evaluation functions."""
        self.constraint_evaluators = {
            MolecularConstraint.MOLECULAR_WEIGHT: self._evaluate_molecular_weight,
            MolecularConstraint.LOGP: self._evaluate_logp,
            MolecularConstraint.TPSA: self._evaluate_tpsa,
            MolecularConstraint.HBD_COUNT: self._evaluate_hbd_count,
            MolecularConstraint.HBA_COUNT: self._evaluate_hba_count,
            MolecularConstraint.ROTATABLE_BONDS: self._evaluate_rotatable_bonds,
            MolecularConstraint.AROMATIC_RINGS: self._evaluate_aromatic_rings,
            MolecularConstraint.SYNTHETIC_ACCESSIBILITY: self._evaluate_synthetic_accessibility,
            MolecularConstraint.DRUG_LIKENESS: self._evaluate_drug_likeness,
        }

    def design_polypharmacology_molecule(
        self,
        target_profile: PolypharmacologyProfile,
        adverse_constraints: AdverseEffectsConstraints,
        design_constraints: DesignConstraints,
        starting_molecules: Optional[List[str]] = None,
        strategy: DesignStrategy = DesignStrategy.MULTI_OBJECTIVE_EVOLUTIONARY,
        max_iterations: int = 1000,
        population_size: int = 100,
        **kwargs,
    ) -> List[OptimizationResult]:
        """
        Design molecules with desired polypharmacology profiles.

        Args:
            target_profile: Desired polypharmacology profile
            adverse_constraints: Constraints for adverse effects
            design_constraints: Molecular design constraints
            starting_molecules: Optional starting molecules for optimization
            strategy: Optimization strategy to use
            max_iterations: Maximum optimization iterations
            population_size: Population size for evolutionary algorithms
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of optimized molecular designs
        """
        start_time = time.time()

        # Validate inputs
        self._validate_design_inputs(
            target_profile, adverse_constraints, design_constraints
        )

        # Initialize population
        if starting_molecules:
            initial_population = starting_molecules[:population_size]
        else:
            initial_population = self._generate_initial_population(
                population_size, design_constraints
            )

        # Run optimization
        optimizer = self.optimization_engines.get(strategy)
        if not optimizer:
            raise ValueError(f"Unsupported optimization strategy: {strategy}")

        results = optimizer(
            initial_population=initial_population,
            target_profile=target_profile,
            adverse_constraints=adverse_constraints,
            design_constraints=design_constraints,
            max_iterations=max_iterations,
            **kwargs,
        )

        # Post-process results
        results = self._post_process_results(
            results, target_profile, adverse_constraints
        )

        # Update design history
        self.design_history.extend(results)

        computation_time = time.time() - start_time
        for result in results:
            result.computational_cost = computation_time / len(results)

        return results

    def optimize_for_minimal_adverse_effects(
        self,
        starting_molecules: List[str],
        adverse_constraints: AdverseEffectsConstraints,
        efficacy_threshold: float = 0.7,
        design_constraints: Optional[DesignConstraints] = None,
        strategy: DesignStrategy = DesignStrategy.BAYESIAN_OPTIMIZATION,
        max_iterations: int = 500,
        **kwargs,
    ) -> List[OptimizationResult]:
        """
        Optimize molecules to minimize adverse effects while maintaining efficacy.

        Args:
            starting_molecules: Starting molecules for optimization
            adverse_constraints: Constraints for adverse effects
            efficacy_threshold: Minimum required efficacy
            design_constraints: Optional molecular design constraints
            strategy: Optimization strategy
            max_iterations: Maximum iterations
            **kwargs: Additional parameters

        Returns:
            List of optimized molecules with minimal adverse effects
        """
        # Create target profile focused on safety
        safety_profile = PolypharmacologyProfile(
            target_engagements={},  # Will be filled based on efficacy requirements
            selectivity_constraints={},
            off_target_limits={},
            therapeutic_window=20.0,  # High therapeutic window
            confidence_threshold=0.8,
        )

        if design_constraints is None:
            design_constraints = DesignConstraints(
                molecular_constraints={
                    MolecularConstraint.MOLECULAR_WEIGHT: (150, 600),
                    MolecularConstraint.LOGP: (-2, 5),
                    MolecularConstraint.TPSA: (20, 140),
                }
            )

        # Evaluate starting molecules to determine efficacy targets
        efficacy_targets = {}
        for smiles in starting_molecules[:10]:  # Sample for efficiency
            try:
                predictions = self.interaction_predictor.predict_interactions_batch(
                    [smiles]
                )
                if smiles in predictions:
                    for prediction in predictions[smiles]:
                        if prediction.interaction_probability >= efficacy_threshold:
                            efficacy_targets[prediction.target.target_id] = max(
                                efficacy_targets.get(prediction.target.target_id, 0),
                                prediction.interaction_probability,
                            )
            except Exception:
                continue

        safety_profile.target_engagements = efficacy_targets

        return self.design_polypharmacology_molecule(
            target_profile=safety_profile,
            adverse_constraints=adverse_constraints,
            design_constraints=design_constraints,
            starting_molecules=starting_molecules,
            strategy=strategy,
            max_iterations=max_iterations,
            population_size=min(50, len(starting_molecules) * 2),
            **kwargs,
        )

    def multi_objective_design(
        self,
        objectives: List[OptimizationObjective],
        constraints: Dict[str, Any],
        starting_molecules: Optional[List[str]] = None,
        pareto_size: int = 50,
        max_iterations: int = 2000,
        **kwargs,
    ) -> List[OptimizationResult]:
        """
        Perform multi-objective molecular design with Pareto optimization.

        Args:
            objectives: List of optimization objectives
            constraints: Constraint specifications
            starting_molecules: Optional starting molecules
            pareto_size: Size of Pareto front to maintain
            max_iterations: Maximum iterations
            **kwargs: Additional parameters

        Returns:
            Pareto-optimal molecular designs
        """
        # Create objective functions
        objective_functions = []
        for obj in objectives:
            obj_func = self._create_objective_function(obj, constraints)
            objective_functions.append(obj_func)

        # Initialize population
        if starting_molecules:
            population = starting_molecules
        else:
            population = self._generate_diverse_population(100, constraints)

        # Run NSGA-II algorithm
        pareto_front = self._nsga_ii_optimization(
            population=population,
            objective_functions=objective_functions,
            constraints=constraints,
            pareto_size=pareto_size,
            max_iterations=max_iterations,
            **kwargs,
        )

        return pareto_front

    def _genetic_algorithm_optimizer(
        self,
        initial_population: List[str],
        target_profile: PolypharmacologyProfile,
        adverse_constraints: AdverseEffectsConstraints,
        design_constraints: DesignConstraints,
        max_iterations: int = 1000,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        **kwargs,
    ) -> List[OptimizationResult]:
        """Genetic algorithm optimization."""
        population = initial_population.copy()
        best_scores = []

        for generation in range(max_iterations):
            # Evaluate population
            fitness_scores = []
            for smiles in population:
                score = self._evaluate_fitness(
                    smiles, target_profile, adverse_constraints, design_constraints
                )
                fitness_scores.append(score)

            best_scores.append(max(fitness_scores))

            # Selection
            selected = self._tournament_selection(
                population, fitness_scores, len(population) // 2
            )

            # Crossover and mutation
            offspring = []
            for i in range(0, len(selected), 2):
                parent1, parent2 = (
                    selected[i],
                    selected[i + 1] if i + 1 < len(selected) else selected[0],
                )

                if np.random.random() < crossover_rate:
                    child1, child2 = self._molecular_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                if np.random.random() < mutation_rate:
                    child1 = self._molecular_mutation(child1, design_constraints)
                if np.random.random() < mutation_rate:
                    child2 = self._molecular_mutation(child2, design_constraints)

                offspring.extend([child1, child2])

            # Replace population
            population = offspring[: len(initial_population)]

        # Return best results
        final_scores = [
            self._evaluate_fitness(
                smiles, target_profile, adverse_constraints, design_constraints
            )
            for smiles in population
        ]

        results = []
        for smiles, score in zip(population, final_scores):
            predicted_profile = self._predict_molecular_profile(smiles)
            constraint_satisfaction = self._evaluate_constraints(
                smiles, design_constraints
            )

            result = OptimizationResult(
                optimized_smiles=smiles,
                optimization_score=score,
                predicted_profile=predicted_profile,
                constraint_satisfaction=constraint_satisfaction,
                optimization_metrics={
                    "final_score": score,
                    "generations": max_iterations,
                },
            )
            results.append(result)

        return sorted(results, key=lambda x: x.optimization_score, reverse=True)[:10]

    def _bayesian_optimizer(
        self,
        initial_population: List[str],
        target_profile: PolypharmacologyProfile,
        adverse_constraints: AdverseEffectsConstraints,
        design_constraints: DesignConstraints,
        max_iterations: int = 500,
        acquisition_function: str = "ei",  # expected improvement
        **kwargs,
    ) -> List[OptimizationResult]:
        """Bayesian optimization for molecular design."""
        if not _SCIPY_SKLEARN_AVAILABLE:
            raise ImportError(
                "SciPy and scikit-learn required for Bayesian optimization"
            )

        # Evaluate initial population
        X_train = []
        y_train = []

        for smiles in initial_population:
            features = self._molecular_to_features(smiles)
            score = self._evaluate_fitness(
                smiles, target_profile, adverse_constraints, design_constraints
            )
            X_train.append(features)
            y_train.append(score)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Fit initial GP model
        self.gp_model.fit(X_train, y_train)

        best_molecules = []

        for iteration in range(max_iterations):
            # Generate candidate molecules
            candidates = self._generate_candidate_molecules(
                design_constraints, n_candidates=100
            )
            candidate_features = np.array(
                [self._molecular_to_features(smiles) for smiles in candidates]
            )

            # Calculate acquisition function
            acquisition_scores = self._calculate_acquisition(
                candidate_features, acquisition_function
            )

            # Select best candidate
            best_idx = np.argmax(acquisition_scores)
            best_candidate = candidates[best_idx]

            # Evaluate and update GP
            true_score = self._evaluate_fitness(
                best_candidate, target_profile, adverse_constraints, design_constraints
            )

            X_train = np.vstack([X_train, candidate_features[best_idx : best_idx + 1]])
            y_train = np.append(y_train, true_score)

            self.gp_model.fit(X_train, y_train)
            best_molecules.append((best_candidate, true_score))

        # Return best results
        best_molecules.sort(key=lambda x: x[1], reverse=True)
        results = []

        for smiles, score in best_molecules[:10]:
            predicted_profile = self._predict_molecular_profile(smiles)
            constraint_satisfaction = self._evaluate_constraints(
                smiles, design_constraints
            )

            result = OptimizationResult(
                optimized_smiles=smiles,
                optimization_score=score,
                predicted_profile=predicted_profile,
                constraint_satisfaction=constraint_satisfaction,
                optimization_metrics={"bayesian_score": score},
            )
            results.append(result)

        return results

    def _rl_optimizer(
        self,
        initial_population: List[str],
        target_profile: PolypharmacologyProfile,
        adverse_constraints: AdverseEffectsConstraints,
        design_constraints: DesignConstraints,
        max_iterations: int = 1000,
        **kwargs,
    ) -> List[OptimizationResult]:
        """Reinforcement learning optimizer."""
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch required for reinforcement learning optimization"
            )

        # Train RL policy
        best_molecules = []

        for episode in range(max_iterations):
            # Generate molecule using policy
            generated_smiles = self._generate_molecule_with_policy(design_constraints)

            # Calculate reward
            reward = self._evaluate_fitness(
                generated_smiles,
                target_profile,
                adverse_constraints,
                design_constraints,
            )

            # Update policy using reward
            self._update_policy(generated_smiles, reward)

            best_molecules.append((generated_smiles, reward))

        # Return best results
        best_molecules.sort(key=lambda x: x[1], reverse=True)
        results = []

        for smiles, reward in best_molecules[:10]:
            predicted_profile = self._predict_molecular_profile(smiles)
            constraint_satisfaction = self._evaluate_constraints(
                smiles, design_constraints
            )

            result = OptimizationResult(
                optimized_smiles=smiles,
                optimization_score=reward,
                predicted_profile=predicted_profile,
                constraint_satisfaction=constraint_satisfaction,
                optimization_metrics={"rl_reward": reward},
            )
            results.append(result)

        return results

    def _simulated_annealing_optimizer(
        self,
        initial_population: List[str],
        target_profile: PolypharmacologyProfile,
        adverse_constraints: AdverseEffectsConstraints,
        design_constraints: DesignConstraints,
        max_iterations: int = 1000,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        **kwargs,
    ) -> List[OptimizationResult]:
        """Simulated annealing optimizer."""
        # Start with best molecule from initial population
        current_smiles = max(
            initial_population,
            key=lambda s: self._evaluate_fitness(
                s, target_profile, adverse_constraints, design_constraints
            ),
        )
        current_score = self._evaluate_fitness(
            current_smiles, target_profile, adverse_constraints, design_constraints
        )

        best_smiles = current_smiles
        best_score = current_score
        temperature = initial_temperature

        trajectory = []

        for iteration in range(max_iterations):
            # Generate neighbor
            neighbor_smiles = self._molecular_mutation(
                current_smiles, design_constraints
            )
            neighbor_score = self._evaluate_fitness(
                neighbor_smiles, target_profile, adverse_constraints, design_constraints
            )

            # Accept or reject neighbor
            if neighbor_score > current_score or np.random.random() < np.exp(
                (neighbor_score - current_score) / temperature
            ):
                current_smiles = neighbor_smiles
                current_score = neighbor_score

                if current_score > best_score:
                    best_smiles = current_smiles
                    best_score = current_score

            trajectory.append((current_smiles, current_score))
            temperature *= cooling_rate

        # Create result
        predicted_profile = self._predict_molecular_profile(best_smiles)
        constraint_satisfaction = self._evaluate_constraints(
            best_smiles, design_constraints
        )

        result = OptimizationResult(
            optimized_smiles=best_smiles,
            optimization_score=best_score,
            predicted_profile=predicted_profile,
            constraint_satisfaction=constraint_satisfaction,
            generation_history=[smiles for smiles, _ in trajectory],
            optimization_metrics={
                "sa_score": best_score,
                "final_temperature": temperature,
            },
        )

        return [result]

    def _moea_optimizer(
        self,
        initial_population: List[str],
        target_profile: PolypharmacologyProfile,
        adverse_constraints: AdverseEffectsConstraints,
        design_constraints: DesignConstraints,
        max_iterations: int = 2000,
        **kwargs,
    ) -> List[OptimizationResult]:
        """Multi-objective evolutionary algorithm (NSGA-II)."""
        population = initial_population.copy()

        # Define objectives
        objectives = [
            lambda s: self._calculate_efficacy_score(s, target_profile),
            lambda s: self._calculate_safety_score(s, adverse_constraints),
            lambda s: self._calculate_selectivity_score(s, target_profile),
        ]

        for generation in range(max_iterations):
            # Evaluate objectives for population
            objective_values = []
            for smiles in population:
                values = [obj(smiles) for obj in objectives]
                objective_values.append(values)

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(objective_values)

            # Selection for next generation
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= len(initial_population):
                    new_population.extend([population[i] for i in front])
                else:
                    # Calculate crowding distance and select best
                    front_objectives = [objective_values[i] for i in front]
                    crowding_distances = self._calculate_crowding_distance(
                        front_objectives
                    )

                    # Sort by crowding distance
                    sorted_indices = sorted(
                        range(len(front)),
                        key=lambda x: crowding_distances[x],
                        reverse=True,
                    )

                    remaining_slots = len(initial_population) - len(new_population)
                    for i in sorted_indices[:remaining_slots]:
                        new_population.append(population[front[i]])
                    break

            # Generate offspring
            offspring = []
            for _ in range(len(initial_population)):
                # Tournament selection
                parent1 = self._tournament_selection_nsga(
                    [population[i] for i in fronts[0]], [1] * len(fronts[0]), 1
                )[0]
                parent2 = self._tournament_selection_nsga(
                    [population[i] for i in fronts[0]], [1] * len(fronts[0]), 1
                )[0]

                # Crossover and mutation
                child = self._molecular_crossover(parent1, parent2)[0]
                if np.random.random() < 0.1:
                    child = self._molecular_mutation(child, design_constraints)

                offspring.append(child)

            population = offspring

        # Return Pareto front
        final_objectives = []
        for smiles in population:
            values = [obj(smiles) for obj in objectives]
            final_objectives.append(values)

        pareto_front_indices = self._fast_non_dominated_sort(final_objectives)[0]

        results = []
        for i in pareto_front_indices:
            smiles = population[i]
            predicted_profile = self._predict_molecular_profile(smiles)
            constraint_satisfaction = self._evaluate_constraints(
                smiles, design_constraints
            )

            result = OptimizationResult(
                optimized_smiles=smiles,
                optimization_score=sum(final_objectives[i]),
                predicted_profile=predicted_profile,
                constraint_satisfaction=constraint_satisfaction,
                pareto_rank=0,
                optimization_metrics={
                    "efficacy": final_objectives[i][0],
                    "safety": final_objectives[i][1],
                    "selectivity": final_objectives[i][2],
                },
            )
            results.append(result)

        return results

    def _evaluate_fitness(
        self,
        smiles: str,
        target_profile: PolypharmacologyProfile,
        adverse_constraints: AdverseEffectsConstraints,
        design_constraints: DesignConstraints,
    ) -> float:
        """Evaluate overall fitness of a molecule."""
        # Check cache first
        cache_key = (
            f"{smiles}_{hash(str(target_profile))}_{hash(str(adverse_constraints))}"
        )
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        try:
            # Calculate component scores
            efficacy_score = self._calculate_efficacy_score(smiles, target_profile)
            safety_score = self._calculate_safety_score(smiles, adverse_constraints)
            constraint_score = self._calculate_constraint_score(
                smiles, design_constraints
            )

            # Weighted combination
            fitness = 0.4 * efficacy_score + 0.4 * safety_score + 0.2 * constraint_score

            # Cache result
            self.evaluation_cache[cache_key] = fitness
            return fitness

        except Exception:
            return 0.0

    def _calculate_efficacy_score(
        self, smiles: str, target_profile: PolypharmacologyProfile
    ) -> float:
        """Calculate efficacy score based on target engagement."""
        try:
            predictions = self.interaction_predictor.predict_interactions_batch(
                [smiles]
            )
            if smiles not in predictions:
                return 0.0

            predicted_engagements = {}
            for prediction in predictions[smiles]:
                predicted_engagements[
                    prediction.target.target_id
                ] = prediction.interaction_probability

            score = 0.0
            count = 0

            for (
                target_id,
                desired_engagement,
            ) in target_profile.target_engagements.items():
                if target_id in predicted_engagements:
                    predicted = predicted_engagements[target_id]
                    # Use Gaussian similarity
                    similarity = np.exp(-((predicted - desired_engagement) ** 2) / 0.1)
                    score += similarity
                    count += 1

            return score / count if count > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_safety_score(
        self, smiles: str, adverse_constraints: AdverseEffectsConstraints
    ) -> float:
        """Calculate safety score based on adverse effects."""
        try:
            tox_profile = self.toxicology_engine.predict_systems_toxicology(smiles)

            safety_score = tox_profile.overall_safety_score

            # Apply penalties for constraint violations
            for endpoint, limit in adverse_constraints.toxicity_limits.items():
                for tox_pred in tox_profile.toxicity_predictions:
                    if (
                        tox_pred.toxicity_endpoint == endpoint
                        and tox_pred.probability > limit
                    ):
                        penalty = (tox_pred.probability - limit) * 2
                        safety_score -= penalty

            return max(0.0, min(1.0, safety_score))

        except Exception:
            return 0.5  # Neutral score if prediction fails

    def _calculate_constraint_score(
        self, smiles: str, design_constraints: DesignConstraints
    ) -> float:
        """Calculate constraint satisfaction score."""
        try:
            if not _RDKIT_AVAILABLE:
                return 1.0

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            score = 1.0
            violations = 0
            total_constraints = len(design_constraints.molecular_constraints)

            for constraint, (
                min_val,
                max_val,
            ) in design_constraints.molecular_constraints.items():
                evaluator = self.constraint_evaluators.get(constraint)
                if evaluator:
                    value = evaluator(mol)
                    if not (min_val <= value <= max_val):
                        violations += 1

            if total_constraints > 0:
                score = 1.0 - (violations / total_constraints)

            return max(0.0, score)

        except Exception:
            return 0.5

    def _predict_molecular_profile(self, smiles: str) -> Dict[str, Any]:
        """Predict comprehensive molecular profile."""
        profile = {}

        try:
            # Target interactions
            interactions = self.interaction_predictor.predict_interactions_batch(
                [smiles]
            )
            if smiles in interactions:
                profile["target_engagements"] = {
                    pred.target.target_id: pred.interaction_probability
                    for pred in interactions[smiles]
                }

            # Toxicology profile
            tox_profile = self.toxicology_engine.predict_systems_toxicology(smiles)
            profile["safety_score"] = tox_profile.overall_safety_score
            profile["therapeutic_index"] = tox_profile.therapeutic_index_estimate
            profile["toxicity_predictions"] = {
                pred.toxicity_endpoint.value: pred.probability
                for pred in tox_profile.toxicity_predictions
            }

            # Molecular properties
            if _RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    profile["molecular_weight"] = Descriptors.MolWt(mol)
                    profile["logp"] = Descriptors.MolLogP(mol)
                    profile["tpsa"] = Descriptors.TPSA(mol)
                    profile["hbd_count"] = Descriptors.NumHDonors(mol)
                    profile["hba_count"] = Descriptors.NumHAcceptors(mol)

        except Exception as e:
            profile["error"] = str(e)

        return profile

    # Utility methods for molecular operations
    def _generate_initial_population(
        self, size: int, constraints: DesignConstraints
    ) -> List[str]:
        """Generate initial population of molecules."""
        population = []
        for _ in range(size):
            smiles = self._generate_random_molecule(constraints)
            if smiles and self._is_valid_molecule(smiles):
                population.append(smiles)

        # Fill remaining slots with simple molecules if needed
        while len(population) < size:
            population.append("CCO")  # Ethanol as fallback

        return population[:size]

    def _generate_random_molecule(self, constraints: DesignConstraints) -> str:
        """Generate a random molecule satisfying constraints."""
        # Simple implementation - could be enhanced with more sophisticated generation
        base_molecules = [
            "CCO",
            "CCC",
            "CCCO",
            "CC(C)O",
            "CCCCO",
            "CC(C)CO",
            "c1ccccc1",
            "c1ccc(O)cc1",
            "c1ccc(N)cc1",
            "CCN",
            "CCNC",
            "CCCN",
            "CC(C)N",
            "CCc1ccccc1",
        ]

        return np.random.choice(base_molecules)

    def _is_valid_molecule(self, smiles: str) -> bool:
        """Check if SMILES represents a valid molecule."""
        if not _RDKIT_AVAILABLE:
            return len(smiles) > 0

        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def _molecular_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Perform crossover between two molecules."""
        # Simple implementation - could use more sophisticated methods
        if not _RDKIT_AVAILABLE:
            return parent1, parent2

        try:
            mol1 = Chem.MolFromSmiles(parent1)
            mol2 = Chem.MolFromSmiles(parent2)

            if mol1 is None or mol2 is None:
                return parent1, parent2

            # Simple fragment exchange
            # This is a placeholder - real implementation would be more complex
            return parent2, parent1

        except:
            return parent1, parent2

    def _molecular_mutation(self, smiles: str, constraints: DesignConstraints) -> str:
        """Mutate a molecule."""
        if not _RDKIT_AVAILABLE:
            return smiles

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles

            # Simple mutations - could be enhanced
            mutations = [
                self._add_functional_group,
                self._remove_functional_group,
                self._replace_ring,
                self._modify_side_chain,
            ]

            mutation = np.random.choice(mutations)
            mutated_smiles = mutation(smiles)

            if self._is_valid_molecule(mutated_smiles):
                return mutated_smiles
            else:
                return smiles

        except:
            return smiles

    def _add_functional_group(self, smiles: str) -> str:
        """Add a functional group to the molecule."""
        # Placeholder implementation
        return smiles

    def _remove_functional_group(self, smiles: str) -> str:
        """Remove a functional group from the molecule."""
        # Placeholder implementation
        return smiles

    def _replace_ring(self, smiles: str) -> str:
        """Replace a ring in the molecule."""
        # Placeholder implementation
        return smiles

    def _modify_side_chain(self, smiles: str) -> str:
        """Modify a side chain in the molecule."""
        # Placeholder implementation
        return smiles

    # Helper methods for constraint evaluation
    def _evaluate_molecular_weight(self, mol) -> float:
        """Evaluate molecular weight."""
        if not _RDKIT_AVAILABLE:
            return 300.0
        return Descriptors.MolWt(mol)

    def _evaluate_logp(self, mol) -> float:
        """Evaluate LogP."""
        if not _RDKIT_AVAILABLE:
            return 2.0
        return Descriptors.MolLogP(mol)

    def _evaluate_tpsa(self, mol) -> float:
        """Evaluate TPSA."""
        if not _RDKIT_AVAILABLE:
            return 60.0
        return Descriptors.TPSA(mol)

    def _evaluate_hbd_count(self, mol) -> int:
        """Evaluate hydrogen bond donor count."""
        if not _RDKIT_AVAILABLE:
            return 2
        return Descriptors.NumHDonors(mol)

    def _evaluate_hba_count(self, mol) -> int:
        """Evaluate hydrogen bond acceptor count."""
        if not _RDKIT_AVAILABLE:
            return 3
        return Descriptors.NumHAcceptors(mol)

    def _evaluate_rotatable_bonds(self, mol) -> int:
        """Evaluate rotatable bond count."""
        if not _RDKIT_AVAILABLE:
            return 5
        return Descriptors.NumRotatableBonds(mol)

    def _evaluate_aromatic_rings(self, mol) -> int:
        """Evaluate aromatic ring count."""
        if not _RDKIT_AVAILABLE:
            return 1
        return Descriptors.NumAromaticRings(mol)

    def _evaluate_synthetic_accessibility(self, mol) -> float:
        """Evaluate synthetic accessibility score."""
        # Placeholder - would use SAScore in real implementation
        return 0.7

    def _evaluate_drug_likeness(self, mol) -> float:
        """Evaluate drug-likeness score."""
        # Placeholder - would use QED or similar in real implementation
        return 0.8

    # Additional helper methods would be implemented here...

    def _load_molecular_fragments(self) -> List[str]:
        """Load molecular fragments for generation."""
        return ["C", "CC", "CCC", "O", "N", "c1ccccc1"]

    def _load_scaffold_library(self) -> List[str]:
        """Load scaffold library."""
        return ["c1ccccc1", "c1ccc2ccccc2c1", "C1CCC2CCCCC2C1"]

    def _load_functional_groups(self) -> List[str]:
        """Load functional groups."""
        return ["O", "N", "C(=O)", "S", "F", "Cl"]

    def _validate_design_inputs(
        self,
        target_profile: PolypharmacologyProfile,
        adverse_constraints: AdverseEffectsConstraints,
        design_constraints: DesignConstraints,
    ):
        """Validate design inputs."""
        if not target_profile.target_engagements:
            raise ValueError(
                "Target profile must specify at least one target engagement"
            )

        if (
            adverse_constraints.safety_score_threshold < 0
            or adverse_constraints.safety_score_threshold > 1
        ):
            raise ValueError("Safety score threshold must be between 0 and 1")

    def _post_process_results(
        self,
        results: List[OptimizationResult],
        target_profile: PolypharmacologyProfile,
        adverse_constraints: AdverseEffectsConstraints,
    ) -> List[OptimizationResult]:
        """Post-process optimization results."""
        # Calculate profile match scores
        for result in results:
            result.optimization_metrics[
                "profile_match"
            ] = result.get_profile_match_score(target_profile)

        # Sort by optimization score
        results.sort(key=lambda x: x.optimization_score, reverse=True)

        return results

    # Additional methods for advanced algorithms would be implemented here...
    def _tournament_selection(
        self, population: List[str], fitness_scores: List[float], n_select: int
    ) -> List[str]:
        """Tournament selection for genetic algorithm."""
        selected = []
        for _ in range(n_select):
            tournament_indices = np.random.choice(
                len(population), size=3, replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        return selected


# Utility functions
def design_polypharmacology_molecule(
    target_profile: PolypharmacologyProfile,
    adverse_constraints: AdverseEffectsConstraints,
    design_constraints: DesignConstraints,
    use_gpu: bool = False,
    **kwargs,
) -> List[OptimizationResult]:
    """
    Design molecules with desired polypharmacology profiles.

    Args:
        target_profile: Desired polypharmacology profile
        adverse_constraints: Constraints for adverse effects
        design_constraints: Molecular design constraints
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional parameters

    Returns:
        List of optimized molecular designs
    """
    designer = SystemsInverseDesigner(use_gpu=use_gpu)
    return designer.design_polypharmacology_molecule(
        target_profile, adverse_constraints, design_constraints, **kwargs
    )


def optimize_for_safety(
    starting_molecules: List[str],
    adverse_constraints: AdverseEffectsConstraints,
    efficacy_threshold: float = 0.7,
    use_gpu: bool = False,
    **kwargs,
) -> List[OptimizationResult]:
    """
    Optimize molecules for minimal adverse effects while maintaining efficacy.

    Args:
        starting_molecules: Starting molecules for optimization
        adverse_constraints: Constraints for adverse effects
        efficacy_threshold: Minimum required efficacy
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional parameters

    Returns:
        List of safety-optimized molecules
    """
    designer = SystemsInverseDesigner(use_gpu=use_gpu)
    return designer.optimize_for_minimal_adverse_effects(
        starting_molecules, adverse_constraints, efficacy_threshold, **kwargs
    )


def create_polypharmacology_profile(
    target_engagements: Dict[str, float],
    selectivity_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    therapeutic_window: float = 10.0,
) -> PolypharmacologyProfile:
    """
    Create a polypharmacology profile specification.

    Args:
        target_engagements: Target ID to desired engagement mapping
        selectivity_constraints: Optional selectivity constraints
        therapeutic_window: Desired therapeutic index

    Returns:
        Polypharmacology profile object
    """
    return PolypharmacologyProfile(
        target_engagements=target_engagements,
        selectivity_constraints=selectivity_constraints or {},
        off_target_limits={},
        therapeutic_window=therapeutic_window,
    )


def create_safety_constraints(
    toxicity_limits: Dict[ToxicityEndpoint, float], safety_threshold: float = 0.8
) -> AdverseEffectsConstraints:
    """
    Create adverse effects constraints.

    Args:
        toxicity_limits: Toxicity endpoint limits
        safety_threshold: Minimum safety score

    Returns:
        Adverse effects constraints object
    """
    return AdverseEffectsConstraints(
        toxicity_limits=toxicity_limits,
        adr_restrictions={},
        off_target_penalties={},
        safety_score_threshold=safety_threshold,
    )


def create_design_constraints(
    molecular_weight_range: Tuple[float, float] = (150, 500),
    logp_range: Tuple[float, float] = (-2, 5),
    tpsa_range: Tuple[float, float] = (20, 140),
) -> DesignConstraints:
    """
    Create molecular design constraints.

    Args:
        molecular_weight_range: Molecular weight range
        logp_range: LogP range
        tpsa_range: TPSA range

    Returns:
        Design constraints object
    """
    return DesignConstraints(
        molecular_constraints={
            MolecularConstraint.MOLECULAR_WEIGHT: molecular_weight_range,
            MolecularConstraint.LOGP: logp_range,
            MolecularConstraint.TPSA: tpsa_range,
        }
    )


# Exceptions
class SystemsInverseDesignError(Exception):
    """Base exception for systems inverse design errors."""

    pass


class OptimizationError(SystemsInverseDesignError):
    """Exception raised during optimization process."""

    pass


class ConstraintViolationError(SystemsInverseDesignError):
    """Exception raised when constraints cannot be satisfied."""

    pass


class MolecularGenerationError(SystemsInverseDesignError):
    """Exception raised during molecular generation."""

    pass

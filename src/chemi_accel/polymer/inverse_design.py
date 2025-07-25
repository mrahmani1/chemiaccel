"""
Inverse Polymer Design Engine for ChemiAccel.

This module provides specialized inverse design capabilities for polymer systems,
including multi-objective optimization for monomer design, GPU-accelerated
generative algorithms, constraint handling for bio-derived sources and synthetic
accessibility, and Pareto front analysis for polymer design objectives.
"""

import time
import warnings
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
    warnings.warn(
        "JAX not available. Inverse polymer design functionality will be limited."
    )

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    from rdkit.Chem.SA_Score import sascorer

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn(
        "RDKit not available. Some polymer constraint functions will be disabled."
    )

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration will be disabled.")

from ..ai.multi_objective_optimization import (
    MPOptimizer,
    MultiObjectiveResult,
    ObjectiveConfig,
)
from ..core.calculator import Calculator
from ..core.exceptions import ChemiAccelError
from .predictive import PolymerPropertyEngine


class InversePolymerDesignError(ChemiAccelError):
    """Raised when inverse polymer design operations fail."""

    pass


@dataclass
class PolymerConstraintConfig:
    """Configuration for polymer-specific constraints."""

    # Synthetic accessibility constraints
    sa_threshold: float = 3.0
    bio_derived_preference: bool = True
    bio_derived_weight: float = 2.0

    # Polymer property constraints
    tg_range: Optional[Tuple[float, float]] = None  # Glass transition temperature (K)
    youngs_modulus_range: Optional[Tuple[float, float]] = None  # GPa
    density_range: Optional[Tuple[float, float]] = None  # g/cm³

    # Monomer structure constraints
    mw_range: Tuple[float, float] = (50.0, 300.0)  # Monomer molecular weight
    functional_groups_required: Optional[List[str]] = None  # Required functional groups
    forbidden_substructures: Optional[List[str]] = None  # SMARTS patterns to avoid

    # Polymerization constraints
    polymerizability_threshold: float = 0.7  # Minimum polymerizability score
    reactivity_balance: bool = True  # Balance reactivity for controlled polymerization

    # Processing constraints
    solubility_parameter_range: Optional[Tuple[float, float]] = None  # (cal/cm³)^0.5
    thermal_stability_min: Optional[float] = None  # Decomposition temperature (K)

    # Custom constraints
    custom_constraints: Optional[List[Callable]] = None


@dataclass
class PolymerDesignObjective:
    """Configuration for polymer design objectives."""

    name: str
    target_value: Optional[float] = None
    minimize: bool = True
    weight: float = 1.0
    property_type: str = "polymer"  # 'polymer', 'monomer', 'processing'
    units: Optional[str] = None
    bounds: Optional[Tuple[float, float]] = None


@dataclass
class MonomerGenerationResult:
    """Results from monomer generation."""

    generated_monomers: List[str]
    predicted_properties: Dict[str, np.ndarray]
    polymer_properties: Dict[str, np.ndarray]
    constraint_violations: Dict[str, np.ndarray]
    pareto_front_indices: Optional[np.ndarray] = None
    generation_history: Optional[List[Dict[str, Any]]] = None
    bio_derived_scores: Optional[np.ndarray] = None
    synthetic_accessibility: Optional[np.ndarray] = None


class BioDerivedScorer:
    """Score monomers for bio-derived potential."""

    def __init__(self):
        """Initialize bio-derived scorer."""
        # Common bio-derived building blocks and their SMARTS patterns
        self.bio_derived_patterns = {
            "glucose_derived": [
                "[CH2][CH]([OH])[CH]([OH])[CH]([OH])[CH]([OH])[CH2][OH]"
            ],
            "cellulose_derived": ["[CH2][CH]([OH])[CH]([OH])[CH]([OH])"],
            "lignin_derived": ["c1ccc(O)c(OC)c1", "c1cc(O)c(OC)cc1"],
            "fatty_acid_derived": ["CCCCCCCC(=O)O", "CCCCCCCCC(=O)O"],
            "amino_acid_derived": ["[NH2][CH]([CH3])[COOH]", "[NH2][CH2][COOH]"],
            "terpene_derived": ["CC(C)=CCC", "CC(C)=CCCC(C)=C"],
            "furan_derived": ["c1ccoc1", "c1coc(C)c1"],
            "phenolic": ["c1ccc(O)cc1", "c1cc(O)ccc1O"],
            "carbohydrate": ["[CH]([OH])[CH]([OH])[CH2][OH]"],
            "plant_oil": ["CCCCCCCC=CCCCCCCCC(=O)O"],
        }

        if not RDKIT_AVAILABLE:
            warnings.warn("RDKit not available. Bio-derived scoring will be limited.")

    def score_bio_derived_potential(self, monomers: List[str]) -> np.ndarray:
        """
        Score monomers for bio-derived potential.

        Args:
            monomers: List of monomer SMILES

        Returns:
            Array of bio-derived scores (0-1, higher is more bio-derived)
        """
        if not RDKIT_AVAILABLE:
            return np.zeros(len(monomers))

        scores = []

        for smiles in monomers:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0.0)
                continue

            bio_score = 0.0
            total_patterns = len(self.bio_derived_patterns)

            # Check for bio-derived structural patterns
            for pattern_name, smarts_list in self.bio_derived_patterns.items():
                pattern_score = 0.0
                for smarts in smarts_list:
                    try:
                        pattern = Chem.MolFromSmarts(smarts)
                        if pattern and mol.HasSubstructMatch(pattern):
                            pattern_score = 1.0
                            break
                    except:
                        continue

                bio_score += pattern_score

            # Normalize by total number of pattern categories
            bio_score = bio_score / total_patterns

            # Additional scoring based on molecular properties
            # Favor molecules with appropriate complexity for bio-derived sources
            mw = Descriptors.MolWt(mol)
            if 100 <= mw <= 250:  # Typical range for bio-derived monomers
                bio_score += 0.1

            # Favor molecules with oxygen content (common in bio-derived)
            oxygen_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
            total_atoms = mol.GetNumAtoms()
            if total_atoms > 0:
                oxygen_ratio = oxygen_count / total_atoms
                if 0.1 <= oxygen_ratio <= 0.4:  # Reasonable oxygen content
                    bio_score += 0.1

            # Penalize highly synthetic structures
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            if aromatic_rings > 2:  # Too many aromatic rings suggest synthetic origin
                bio_score *= 0.8

            scores.append(min(1.0, bio_score))

        return np.array(scores)


class PolymerConstraintHandler:
    """Handle polymer-specific constraints during optimization."""

    def __init__(
        self,
        config: PolymerConstraintConfig,
        calculator: Calculator,
        polymer_engine: PolymerPropertyEngine,
    ):
        """Initialize polymer constraint handler."""
        self.config = config
        self.calculator = calculator
        self.polymer_engine = polymer_engine
        self.bio_derived_scorer = BioDerivedScorer()

    def evaluate_constraints(self, monomers: List[str]) -> Dict[str, np.ndarray]:
        """Evaluate all polymer constraints for given monomers."""
        constraints = {}

        # Basic molecular constraints
        constraints.update(self._evaluate_molecular_constraints(monomers))

        # Polymer property constraints
        constraints.update(self._evaluate_polymer_property_constraints(monomers))

        # Polymerization constraints
        constraints.update(self._evaluate_polymerization_constraints(monomers))

        # Bio-derived constraints
        if self.config.bio_derived_preference:
            constraints.update(self._evaluate_bio_derived_constraints(monomers))

        # Custom constraints
        if self.config.custom_constraints:
            constraints.update(self._evaluate_custom_constraints(monomers))

        return constraints

    def _evaluate_molecular_constraints(
        self, monomers: List[str]
    ) -> Dict[str, np.ndarray]:
        """Evaluate basic molecular constraints."""
        constraints = {}

        if not RDKIT_AVAILABLE:
            return constraints

        mw_violations = []
        sa_violations = []
        forbidden_violations = []
        required_violations = []

        for smiles in monomers:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mw_violations.append(10.0)
                sa_violations.append(10.0)
                forbidden_violations.append(10.0)
                required_violations.append(10.0)
                continue

            # Molecular weight constraint
            mw = Descriptors.MolWt(mol)
            mw_violation = max(0, mw - self.config.mw_range[1]) + max(
                0, self.config.mw_range[0] - mw
            )
            mw_violations.append(mw_violation)

            # Synthetic accessibility constraint
            try:
                sa_score = sascorer.calculateScore(mol)
                sa_violation = max(0, sa_score - self.config.sa_threshold)
                sa_violations.append(sa_violation)
            except:
                sa_violations.append(10.0)

            # Forbidden substructures
            forbidden_violation = 0.0
            if self.config.forbidden_substructures:
                for smarts in self.config.forbidden_substructures:
                    try:
                        pattern = Chem.MolFromSmarts(smarts)
                        if pattern and mol.HasSubstructMatch(pattern):
                            forbidden_violation += 5.0
                    except:
                        continue
            forbidden_violations.append(forbidden_violation)

            # Required functional groups
            required_violation = 0.0
            if self.config.functional_groups_required:
                for smarts in self.config.functional_groups_required:
                    try:
                        pattern = Chem.MolFromSmarts(smarts)
                        if pattern and not mol.HasSubstructMatch(pattern):
                            required_violation += 5.0
                    except:
                        required_violation += 5.0
            required_violations.append(required_violation)

        constraints["mw_violation"] = np.array(mw_violations)
        constraints["sa_violation"] = np.array(sa_violations)
        constraints["forbidden_violation"] = np.array(forbidden_violations)
        constraints["required_violation"] = np.array(required_violations)

        return constraints

    def _evaluate_polymer_property_constraints(
        self, monomers: List[str]
    ) -> Dict[str, np.ndarray]:
        """Evaluate polymer property constraints."""
        constraints = {}

        try:
            # Predict polymer properties from monomers
            polymer_properties = self.polymer_engine.predict_polymer_properties(
                monomers
            )

            # Glass transition temperature constraint
            if self.config.tg_range and "glass_transition_temp" in polymer_properties:
                tg_values = polymer_properties["glass_transition_temp"]
                tg_violations = []
                for tg in tg_values:
                    violation = max(0, tg - self.config.tg_range[1]) + max(
                        0, self.config.tg_range[0] - tg
                    )
                    tg_violations.append(violation)
                constraints["tg_violation"] = np.array(tg_violations)

            # Young's modulus constraint
            if (
                self.config.youngs_modulus_range
                and "youngs_modulus" in polymer_properties
            ):
                modulus_values = polymer_properties["youngs_modulus"]
                modulus_violations = []
                for modulus in modulus_values:
                    violation = max(
                        0, modulus - self.config.youngs_modulus_range[1]
                    ) + max(0, self.config.youngs_modulus_range[0] - modulus)
                    modulus_violations.append(violation)
                constraints["modulus_violation"] = np.array(modulus_violations)

            # Density constraint
            if self.config.density_range and "density" in polymer_properties:
                density_values = polymer_properties["density"]
                density_violations = []
                for density in density_values:
                    violation = max(0, density - self.config.density_range[1]) + max(
                        0, self.config.density_range[0] - density
                    )
                    density_violations.append(violation)
                constraints["density_violation"] = np.array(density_violations)

        except Exception as e:
            warnings.warn(f"Polymer property constraint evaluation failed: {e}")
            # Add default violations for failed predictions
            n_monomers = len(monomers)
            if self.config.tg_range:
                constraints["tg_violation"] = np.full(n_monomers, 5.0)
            if self.config.youngs_modulus_range:
                constraints["modulus_violation"] = np.full(n_monomers, 5.0)
            if self.config.density_range:
                constraints["density_violation"] = np.full(n_monomers, 5.0)

        return constraints

    def _evaluate_polymerization_constraints(
        self, monomers: List[str]
    ) -> Dict[str, np.ndarray]:
        """Evaluate polymerization-related constraints."""
        constraints = {}

        # This would typically involve more sophisticated analysis
        # For now, implement basic heuristics

        polymerizability_violations = []
        reactivity_violations = []

        for smiles in monomers:
            mol = Chem.MolFromSmiles(smiles) if RDKIT_AVAILABLE else None

            if mol is None:
                polymerizability_violations.append(10.0)
                reactivity_violations.append(10.0)
                continue

            # Simple polymerizability heuristic based on functional groups
            polymerizability_score = self._estimate_polymerizability(mol)
            polymerizability_violation = max(
                0, self.config.polymerizability_threshold - polymerizability_score
            )
            polymerizability_violations.append(polymerizability_violation)

            # Reactivity balance heuristic
            if self.config.reactivity_balance:
                reactivity_score = self._estimate_reactivity_balance(mol)
                reactivity_violation = max(
                    0, 0.5 - reactivity_score
                )  # Want balanced reactivity
                reactivity_violations.append(reactivity_violation)
            else:
                reactivity_violations.append(0.0)

        constraints["polymerizability_violation"] = np.array(
            polymerizability_violations
        )
        constraints["reactivity_violation"] = np.array(reactivity_violations)

        return constraints

    def _evaluate_bio_derived_constraints(
        self, monomers: List[str]
    ) -> Dict[str, np.ndarray]:
        """Evaluate bio-derived preference constraints."""
        bio_scores = self.bio_derived_scorer.score_bio_derived_potential(monomers)

        # Convert to violations (higher bio-derived score = lower violation)
        bio_violations = self.config.bio_derived_weight * (1.0 - bio_scores)

        return {"bio_derived_violation": bio_violations}

    def _evaluate_custom_constraints(
        self, monomers: List[str]
    ) -> Dict[str, np.ndarray]:
        """Evaluate custom constraint functions."""
        constraints = {}

        for i, constraint_fn in enumerate(self.config.custom_constraints):
            violations = []
            for smiles in monomers:
                try:
                    violation = constraint_fn(smiles)
                    violations.append(max(0, violation))
                except:
                    violations.append(10.0)

            constraints[f"custom_{i}_violation"] = np.array(violations)

        return constraints

    def _estimate_polymerizability(self, mol) -> float:
        """Estimate polymerizability score based on functional groups."""
        if not RDKIT_AVAILABLE:
            return 0.5

        # Common polymerizable functional groups (SMARTS patterns)
        polymerizable_groups = [
            "C=C",  # Alkene (addition polymerization)
            "C#C",  # Alkyne
            "C(=O)O",  # Carboxylic acid (condensation)
            "C(=O)N",  # Amide
            "C(=O)OC",  # Ester
            "N",  # Amine (condensation)
            "O",  # Alcohol (condensation)
            "c1ccccc1",  # Aromatic (for aromatic polymerization)
            "C=O",  # Carbonyl
            "S(=O)(=O)Cl",  # Sulfonyl chloride
        ]

        score = 0.0
        for smarts in polymerizable_groups:
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    score += len(matches) * 0.2  # Each match contributes to score
            except:
                continue

        return min(1.0, score)

    def _estimate_reactivity_balance(self, mol) -> float:
        """Estimate reactivity balance for controlled polymerization."""
        if not RDKIT_AVAILABLE:
            return 0.5

        # This is a simplified heuristic
        # In practice, would use more sophisticated reactivity models

        # Count different types of reactive sites
        electron_withdrawing = 0
        electron_donating = 0

        # Simple patterns for electron-withdrawing/donating groups
        ew_patterns = ["C(=O)", "C#N", "N(=O)=O", "S(=O)(=O)"]
        ed_patterns = ["N", "O", "c1ccccc1"]  # Simplified

        for smarts in ew_patterns:
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    electron_withdrawing += len(matches)
            except:
                continue

        for smarts in ed_patterns:
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    electron_donating += len(matches)
            except:
                continue

        # Balance score: closer to 1 when EW and ED groups are balanced
        total_groups = electron_withdrawing + electron_donating
        if total_groups == 0:
            return 0.5

        balance = 1.0 - abs(electron_withdrawing - electron_donating) / total_groups
        return balance


class MonomerGenerator:
    """GPU-accelerated generative algorithms for monomer design."""

    def __init__(self, calculator: Calculator, use_gpu: bool = True):
        """Initialize monomer generator."""
        self.calculator = calculator
        self.use_gpu = use_gpu and (JAX_AVAILABLE or CUPY_AVAILABLE)

        # Initialize generation models (simplified for this implementation)
        self.generation_history = []

    def generate_monomers(
        self,
        seed_monomers: List[str],
        n_generations: int = 100,
        population_size: int = 50,
        mutation_rate: float = 0.1,
    ) -> List[str]:
        """
        Generate novel monomers using evolutionary algorithms.

        Args:
            seed_monomers: Starting monomer structures
            n_generations: Number of evolutionary generations
            population_size: Size of population to maintain
            mutation_rate: Rate of mutations per generation

        Returns:
            List of generated monomer SMILES
        """
        if not RDKIT_AVAILABLE:
            warnings.warn("RDKit not available. Monomer generation will be limited.")
            return seed_monomers

        # Initialize population
        current_population = seed_monomers[:population_size]
        if len(current_population) < population_size:
            # Duplicate seed monomers to reach population size
            while len(current_population) < population_size:
                current_population.extend(seed_monomers)
            current_population = current_population[:population_size]

        # Evolutionary loop
        for generation in range(n_generations):
            # Generate mutations
            mutated_population = self._mutate_population(
                current_population, mutation_rate
            )

            # Combine with current population
            combined_population = current_population + mutated_population

            # Select best candidates (simplified selection)
            current_population = self._select_population(
                combined_population, population_size
            )

            # Record generation
            self.generation_history.append(
                {
                    "generation": generation,
                    "population": current_population.copy(),
                    "diversity": self._calculate_diversity(current_population),
                }
            )

        return current_population

    def _mutate_population(
        self, population: List[str], mutation_rate: float
    ) -> List[str]:
        """Apply mutations to population."""
        mutated = []

        for smiles in population:
            if np.random.random() < mutation_rate:
                mutated_smiles = self._mutate_molecule(smiles)
                if mutated_smiles:
                    mutated.append(mutated_smiles)

        return mutated

    def _mutate_molecule(self, smiles: str) -> Optional[str]:
        """Apply random mutation to a molecule."""
        if not RDKIT_AVAILABLE:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Simple mutation strategies
        mutation_strategies = [
            self._add_functional_group,
            self._remove_atom,
            self._modify_bond,
            self._add_substituent,
        ]

        strategy = np.random.choice(mutation_strategies)
        try:
            mutated_mol = strategy(mol)
            if mutated_mol:
                return Chem.MolToSmiles(mutated_mol)
        except:
            pass

        return None

    def _add_functional_group(self, mol):
        """Add a functional group to the molecule."""
        # Simplified implementation
        # In practice, would use more sophisticated chemistry-aware mutations
        return mol

    def _remove_atom(self, mol):
        """Remove a non-essential atom from the molecule."""
        # Simplified implementation
        return mol

    def _modify_bond(self, mol):
        """Modify bond order in the molecule."""
        # Simplified implementation
        return mol

    def _add_substituent(self, mol):
        """Add a small substituent to the molecule."""
        # Simplified implementation
        return mol

    def _select_population(self, population: List[str], target_size: int) -> List[str]:
        """Select best candidates from population."""
        # Simplified selection based on diversity and validity
        valid_population = []

        for smiles in population:
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_population.append(smiles)
            else:
                valid_population.append(smiles)

        # Remove duplicates
        unique_population = list(set(valid_population))

        # Return up to target_size molecules
        return unique_population[:target_size]

    def _calculate_diversity(self, population: List[str]) -> float:
        """Calculate diversity score for population."""
        if not RDKIT_AVAILABLE or len(population) < 2:
            return 0.0

        # Calculate pairwise Tanimoto similarities
        similarities = []

        for i in range(len(population)):
            mol_i = Chem.MolFromSmiles(population[i])
            if mol_i is None:
                continue

            fp_i = Chem.RDKFingerprint(mol_i)

            for j in range(i + 1, len(population)):
                mol_j = Chem.MolFromSmiles(population[j])
                if mol_j is None:
                    continue

                fp_j = Chem.RDKFingerprint(mol_j)
                similarity = Chem.DataStructs.TanimotoSimilarity(fp_i, fp_j)
                similarities.append(similarity)

        if similarities:
            # Diversity is 1 - average similarity
            return 1.0 - np.mean(similarities)
        else:
            return 0.0


class InversePolymerDesigner:
    """
    Comprehensive inverse polymer design engine.

    This class implements multi-objective optimization for polymer design,
    including GPU-accelerated generative algorithms for monomer design,
    constraint handling for bio-derived sources and synthetic accessibility,
    and Pareto front analysis for polymer design objectives.
    """

    def __init__(
        self,
        calculator: Calculator,
        polymer_engine: PolymerPropertyEngine,
        objectives: List[PolymerDesignObjective],
        constraint_config: Optional[PolymerConstraintConfig] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize inverse polymer designer.

        Args:
            calculator: ChemiAccel calculator for property prediction
            polymer_engine: Polymer property prediction engine
            objectives: List of design objectives
            constraint_config: Configuration for constraints
            use_gpu: Whether to use GPU acceleration
        """
        self.calculator = calculator
        self.polymer_engine = polymer_engine
        self.objectives = objectives
        self.constraint_config = constraint_config or PolymerConstraintConfig()
        self.use_gpu = use_gpu

        # Initialize components
        self.constraint_handler = PolymerConstraintHandler(
            self.constraint_config, calculator, polymer_engine
        )
        self.monomer_generator = MonomerGenerator(calculator, use_gpu)

        # Set up multi-objective optimizer
        mo_objectives = []
        for obj in objectives:
            mo_obj = ObjectiveConfig(
                name=obj.name,
                minimize=obj.minimize,
                weight=obj.weight,
                target=obj.target_value,
                bounds=obj.bounds,
            )
            mo_objectives.append(mo_obj)

        self.multi_objective_optimizer = MPOptimizer(
            objectives=mo_objectives, use_gpu=use_gpu
        )

        # Design state
        self.design_history = []
        self.best_designs = []

    def design_monomers(
        self,
        seed_monomers: List[str],
        n_iterations: int = 50,
        population_size: int = 100,
        generation_size: int = 20,
    ) -> MonomerGenerationResult:
        """
        Design novel monomers optimized for target polymer properties.

        Args:
            seed_monomers: Starting monomer structures
            n_iterations: Number of design iterations
            population_size: Size of population to maintain
            generation_size: Number of new monomers to generate per iteration

        Returns:
            MonomerGenerationResult with optimized monomers and analysis
        """
        start_time = time.time()

        # Initialize population
        current_monomers = seed_monomers[:population_size]
        if len(current_monomers) < population_size:
            # Generate additional monomers to reach population size
            additional_monomers = self.monomer_generator.generate_monomers(
                seed_monomers,
                n_generations=10,
                population_size=population_size - len(current_monomers),
            )
            current_monomers.extend(additional_monomers)
            current_monomers = current_monomers[:population_size]

        # Design optimization loop
        for iteration in range(n_iterations):
            # Generate new candidate monomers
            new_monomers = self.monomer_generator.generate_monomers(
                current_monomers, n_generations=5, population_size=generation_size
            )

            # Combine with current population
            candidate_monomers = current_monomers + new_monomers

            # Evaluate candidates
            evaluation_result = self._evaluate_candidates(candidate_monomers)

            # Multi-objective selection
            selected_monomers = self._select_best_candidates(
                candidate_monomers, evaluation_result, population_size
            )

            current_monomers = selected_monomers

            # Record iteration
            self.design_history.append(
                {
                    "iteration": iteration,
                    "monomers": current_monomers.copy(),
                    "evaluation": evaluation_result,
                    "diversity": self.monomer_generator._calculate_diversity(
                        current_monomers
                    ),
                }
            )

            # Early stopping check
            if self._check_convergence():
                print(f"Design converged at iteration {iteration}")
                break

        # Final evaluation
        final_evaluation = self._evaluate_candidates(current_monomers)

        # Multi-objective optimization
        mo_result = self._perform_multi_objective_optimization(
            current_monomers, final_evaluation
        )

        # Calculate bio-derived scores
        bio_derived_scores = (
            self.constraint_handler.bio_derived_scorer.score_bio_derived_potential(
                current_monomers
            )
        )

        # Calculate synthetic accessibility
        sa_scores = self._calculate_synthetic_accessibility(current_monomers)

        computation_time = time.time() - start_time

        return MonomerGenerationResult(
            generated_monomers=current_monomers,
            predicted_properties=final_evaluation["monomer_properties"],
            polymer_properties=final_evaluation["polymer_properties"],
            constraint_violations=final_evaluation["constraints"],
            pareto_front_indices=mo_result.pareto_front.pareto_indices,
            generation_history=self.design_history,
            bio_derived_scores=bio_derived_scores,
            synthetic_accessibility=sa_scores,
        )

    def _evaluate_candidates(self, monomers: List[str]) -> Dict[str, Any]:
        """Evaluate candidate monomers."""
        evaluation = {}

        # Predict monomer properties
        try:
            monomer_series = pd.Series(monomers)
            monomer_properties = self.calculator.pandas(monomer_series)
            evaluation["monomer_properties"] = {
                col: monomer_properties[col].values
                for col in monomer_properties.columns
            }
        except Exception as e:
            warnings.warn(f"Monomer property prediction failed: {e}")
            evaluation["monomer_properties"] = {}

        # Predict polymer properties
        try:
            polymer_properties = self.polymer_engine.predict_polymer_properties(
                monomers
            )
            evaluation["polymer_properties"] = polymer_properties
        except Exception as e:
            warnings.warn(f"Polymer property prediction failed: {e}")
            evaluation["polymer_properties"] = {}

        # Evaluate constraints
        constraints = self.constraint_handler.evaluate_constraints(monomers)
        evaluation["constraints"] = constraints

        return evaluation

    def _select_best_candidates(
        self, monomers: List[str], evaluation: Dict[str, Any], target_size: int
    ) -> List[str]:
        """Select best candidates using multi-objective optimization."""
        # Combine all properties for multi-objective evaluation
        all_properties = {}
        all_properties.update(evaluation.get("monomer_properties", {}))
        all_properties.update(evaluation.get("polymer_properties", {}))
        all_properties.update(evaluation.get("constraints", {}))

        if not all_properties:
            # Fallback to simple selection
            return monomers[:target_size]

        try:
            # Perform multi-objective optimization
            mo_result = self.multi_objective_optimizer.optimize(
                molecules=monomers,
                properties=all_properties,
                population_size=target_size,
            )

            # Return molecules from Pareto front and best non-dominated solutions
            pareto_molecules = [
                mo_result.molecules[i] for i in mo_result.pareto_front.pareto_indices
            ]

            # Fill remaining slots with diverse solutions
            remaining_slots = target_size - len(pareto_molecules)
            if remaining_slots > 0:
                non_pareto_indices = [
                    i
                    for i in range(len(mo_result.molecules))
                    if i not in mo_result.pareto_front.pareto_indices
                ]
                additional_molecules = [
                    mo_result.molecules[i] for i in non_pareto_indices[:remaining_slots]
                ]
                pareto_molecules.extend(additional_molecules)

            return pareto_molecules[:target_size]

        except Exception as e:
            warnings.warn(f"Multi-objective selection failed: {e}")
            return monomers[:target_size]

    def _perform_multi_objective_optimization(
        self, monomers: List[str], evaluation: Dict[str, Any]
    ) -> MultiObjectiveResult:
        """Perform final multi-objective optimization."""
        # Combine all properties
        all_properties = {}
        all_properties.update(evaluation.get("monomer_properties", {}))
        all_properties.update(evaluation.get("polymer_properties", {}))
        all_properties.update(evaluation.get("constraints", {}))

        return self.multi_objective_optimizer.optimize(
            molecules=monomers, properties=all_properties
        )

    def _calculate_synthetic_accessibility(self, monomers: List[str]) -> np.ndarray:
        """Calculate synthetic accessibility scores."""
        if not RDKIT_AVAILABLE:
            return np.zeros(len(monomers))

        sa_scores = []
        for smiles in monomers:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    sa_score = sascorer.calculateScore(mol)
                    sa_scores.append(sa_score)
                except:
                    sa_scores.append(5.0)  # Default moderate score
            else:
                sa_scores.append(10.0)  # High score for invalid molecules

        return np.array(sa_scores)

    def _check_convergence(self) -> bool:
        """Check if design has converged."""
        if len(self.design_history) < 5:
            return False

        # Check diversity convergence
        recent_diversities = [info["diversity"] for info in self.design_history[-5:]]
        diversity_std = np.std(recent_diversities)

        return diversity_std < 0.01  # Converged if diversity is stable

    def get_pareto_front_monomers(self, result: MonomerGenerationResult) -> List[str]:
        """Extract monomers on the Pareto front."""
        if result.pareto_front_indices is not None:
            return [result.generated_monomers[i] for i in result.pareto_front_indices]
        else:
            return result.generated_monomers

    def analyze_design_objectives(
        self, result: MonomerGenerationResult
    ) -> Dict[str, Any]:
        """Analyze how well design objectives were met."""
        analysis = {}

        if result.pareto_front_indices is not None:
            pareto_monomers = self.get_pareto_front_monomers(result)
            analysis["n_pareto_solutions"] = len(pareto_monomers)

            # Analyze objective achievement for Pareto front
            for obj in self.objectives:
                obj_name = obj.name

                # Get property values for Pareto front
                if obj_name in result.predicted_properties:
                    pareto_values = result.predicted_properties[obj_name][
                        result.pareto_front_indices
                    ]
                elif obj_name in result.polymer_properties:
                    pareto_values = result.polymer_properties[obj_name][
                        result.pareto_front_indices
                    ]
                else:
                    continue

                analysis[f"{obj_name}_pareto_range"] = (
                    np.min(pareto_values),
                    np.max(pareto_values),
                )
                analysis[f"{obj_name}_pareto_mean"] = np.mean(pareto_values)

                if obj.target_value is not None:
                    distances = np.abs(pareto_values - obj.target_value)
                    analysis[f"{obj_name}_target_achievement"] = np.min(distances)

        # Bio-derived analysis
        if result.bio_derived_scores is not None:
            analysis["bio_derived_mean"] = np.mean(result.bio_derived_scores)
            analysis["bio_derived_max"] = np.max(result.bio_derived_scores)
            analysis["highly_bio_derived_count"] = np.sum(
                result.bio_derived_scores > 0.7
            )

        # Synthetic accessibility analysis
        if result.synthetic_accessibility is not None:
            analysis["sa_mean"] = np.mean(result.synthetic_accessibility)
            analysis["sa_min"] = np.min(result.synthetic_accessibility)
            analysis["easily_synthesizable_count"] = np.sum(
                result.synthetic_accessibility < 3.0
            )

        return analysis

    def export_design_results(
        self,
        result: MonomerGenerationResult,
        filename: str = "polymer_design_results.csv",
    ) -> None:
        """Export design results to CSV file."""
        # Create comprehensive results DataFrame
        data = {
            "monomer_smiles": result.generated_monomers,
            "is_pareto_optimal": [False] * len(result.generated_monomers),
        }

        if result.pareto_front_indices is not None:
            for idx in result.pareto_front_indices:
                data["is_pareto_optimal"][idx] = True

        # Add predicted properties
        for prop_name, values in result.predicted_properties.items():
            data[f"monomer_{prop_name}"] = values

        # Add polymer properties
        for prop_name, values in result.polymer_properties.items():
            data[f"polymer_{prop_name}"] = values

        # Add constraint violations
        for constraint_name, values in result.constraint_violations.items():
            data[constraint_name] = values

        # Add bio-derived scores
        if result.bio_derived_scores is not None:
            data["bio_derived_score"] = result.bio_derived_scores

        # Add synthetic accessibility
        if result.synthetic_accessibility is not None:
            data["synthetic_accessibility"] = result.synthetic_accessibility

        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Design results exported to {filename}")


# Utility functions for polymer design
def create_high_performance_polymer_objectives() -> List[PolymerDesignObjective]:
    """Create objectives for high-performance polymer design."""
    return [
        PolymerDesignObjective(
            name="glass_transition_temp",
            target_value=400.0,  # High Tg for thermal stability
            minimize=False,
            weight=2.0,
            property_type="polymer",
            units="K",
        ),
        PolymerDesignObjective(
            name="youngs_modulus",
            target_value=5.0,  # High modulus for stiffness
            minimize=False,
            weight=2.0,
            property_type="polymer",
            units="GPa",
        ),
        PolymerDesignObjective(
            name="synthetic_accessibility",
            target_value=2.0,  # Easy to synthesize
            minimize=True,
            weight=1.0,
            property_type="monomer",
        ),
        PolymerDesignObjective(
            name="bio_derived_score",
            target_value=0.8,  # Prefer bio-derived
            minimize=False,
            weight=1.5,
            property_type="monomer",
        ),
    ]


def create_biodegradable_polymer_objectives() -> List[PolymerDesignObjective]:
    """Create objectives for biodegradable polymer design."""
    return [
        PolymerDesignObjective(
            name="bio_derived_score",
            target_value=0.9,  # Highly bio-derived
            minimize=False,
            weight=3.0,
            property_type="monomer",
        ),
        PolymerDesignObjective(
            name="glass_transition_temp",
            target_value=300.0,  # Moderate Tg
            minimize=False,
            weight=1.0,
            property_type="polymer",
            units="K",
        ),
        PolymerDesignObjective(
            name="synthetic_accessibility",
            target_value=2.5,  # Reasonably easy to synthesize
            minimize=True,
            weight=1.0,
            property_type="monomer",
        ),
    ]


def create_bio_derived_constraints() -> PolymerConstraintConfig:
    """Create constraints favoring bio-derived monomers."""
    return PolymerConstraintConfig(
        sa_threshold=3.5,
        bio_derived_preference=True,
        bio_derived_weight=3.0,
        mw_range=(80.0, 250.0),
        functional_groups_required=["[OH]", "C(=O)O"],  # Require hydroxyl or carboxyl
        forbidden_substructures=["[F]", "[Cl]", "[Br]", "[I]"],  # Avoid halogens
        polymerizability_threshold=0.6,
        reactivity_balance=True,
    )


def create_high_performance_constraints() -> PolymerConstraintConfig:
    """Create constraints for high-performance polymers."""
    return PolymerConstraintConfig(
        sa_threshold=4.0,  # Allow more complex synthesis
        bio_derived_preference=False,
        tg_range=(350.0, 500.0),  # High glass transition
        youngs_modulus_range=(3.0, 20.0),  # High modulus
        mw_range=(100.0, 400.0),
        polymerizability_threshold=0.8,
        reactivity_balance=True,
        thermal_stability_min=450.0,  # High thermal stability
    )

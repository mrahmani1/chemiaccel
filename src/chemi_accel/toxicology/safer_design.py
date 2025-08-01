"""
Inverse design for safer-by-design chemicals.

Multi-objective optimization system for designing chemical alternatives
with reduced toxicity while maintaining desired functional properties.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.backend import Backend
from ..core.exceptions import ChemiAccelError
from .core import ConfidenceLevel, ToxicityData, ToxicityEndpointType


class OptimizationObjective(Enum):
    """Optimization objectives for safer design."""

    MINIMIZE_TOXICITY = "minimize_toxicity"
    MAXIMIZE_EFFICACY = "maximize_efficacy"
    MINIMIZE_PERSISTENCE = "minimize_persistence"
    MINIMIZE_BIOACCUMULATION = "minimize_bioaccumulation"
    MAXIMIZE_BIODEGRADABILITY = "maximize_biodegradability"
    MINIMIZE_VOLATILITY = "minimize_volatility"
    MAXIMIZE_SELECTIVITY = "maximize_selectivity"
    MINIMIZE_COST = "minimize_cost"


class MolecularTransformation(Enum):
    """Types of molecular transformations for optimization."""

    FUNCTIONAL_GROUP_REPLACEMENT = "functional_group_replacement"
    BIOISOSTERIC_REPLACEMENT = "bioisosteric_replacement"
    SCAFFOLD_HOPPING = "scaffold_hopping"
    SIDE_CHAIN_MODIFICATION = "side_chain_modification"
    RING_MODIFICATION = "ring_modification"
    STEREOCHEMISTRY_MODIFICATION = "stereochemistry_modification"
    ATOM_REPLACEMENT = "atom_replacement"
    FRAGMENT_LINKING = "fragment_linking"


@dataclass
class ToxicityConstraint:
    """Constraint on toxicity endpoints for optimization."""

    endpoint_type: ToxicityEndpointType
    constraint_type: str  # "less_than", "greater_than", "range"
    target_value: float
    max_value: Optional[float] = None  # For range constraints
    weight: float = 1.0
    priority: str = "hard"  # "hard" or "soft"

    def evaluate(self, predicted_value: float) -> Tuple[bool, float]:
        """Evaluate constraint satisfaction."""
        satisfied = False
        penalty = 0.0

        if self.constraint_type == "less_than":
            satisfied = predicted_value <= self.target_value
            if not satisfied:
                penalty = (predicted_value - self.target_value) * self.weight

        elif self.constraint_type == "greater_than":
            satisfied = predicted_value >= self.target_value
            if not satisfied:
                penalty = (self.target_value - predicted_value) * self.weight

        elif self.constraint_type == "range":
            satisfied = (
                self.target_value <= predicted_value <= (self.max_value or float("inf"))
            )
            if not satisfied:
                if predicted_value < self.target_value:
                    penalty = (self.target_value - predicted_value) * self.weight
                else:
                    penalty = (
                        predicted_value - (self.max_value or self.target_value)
                    ) * self.weight

        return satisfied, penalty


@dataclass
class SafeDesignCandidate:
    """Candidate molecule from safer design optimization."""

    smiles: str
    predicted_properties: Dict[str, float]
    toxicity_predictions: Optional[ToxicityData] = None
    optimization_score: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)
    transformation_history: List[str] = field(default_factory=list)
    similarity_to_parent: float = 0.0
    synthetic_accessibility: Optional[float] = None
    estimated_cost: Optional[float] = None
    confidence_score: float = 0.0

    @property
    def is_viable(self) -> bool:
        """Check if candidate meets all hard constraints."""
        return len(self.constraint_violations) == 0

    @property
    def overall_desirability(self) -> float:
        """Calculate overall desirability score."""
        base_score = self.optimization_score

        # Penalty for constraint violations
        violation_penalty = len(self.constraint_violations) * 0.2

        # Bonus for high similarity (easier synthesis)
        similarity_bonus = self.similarity_to_parent * 0.1

        # Penalty for low synthetic accessibility
        synthesis_penalty = 0.0
        if self.synthetic_accessibility is not None:
            synthesis_penalty = max(0, (5.0 - self.synthetic_accessibility) * 0.05)

        return max(
            0, base_score - violation_penalty + similarity_bonus - synthesis_penalty
        )


class MolecularGenerator(ABC):
    """Abstract base class for molecular generation."""

    @abstractmethod
    def generate_candidates(
        self,
        parent_smiles: str,
        n_candidates: int,
        transformations: List[MolecularTransformation],
        **kwargs,
    ) -> List[str]:
        """Generate candidate molecules."""
        pass


class FragmentBasedGenerator(MolecularGenerator):
    """Fragment-based molecular generator."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.fragment_library = self._load_fragment_library()
        self.transformation_rules = self._load_transformation_rules()

    def _load_fragment_library(self) -> Dict[str, List[str]]:
        """Load fragment library for replacements."""
        # This would load from a comprehensive fragment database
        return {
            "aromatic_rings": [
                "c1ccccc1",  # benzene
                "c1ccncc1",  # pyridine
                "c1cccnc1",  # pyrimidine
                "c1ccoc1",  # furan
                "c1ccsc1",  # thiophene
            ],
            "aliphatic_chains": ["CC", "CCC", "CCCC", "C(C)C", "CC(C)C"],
            "functional_groups": [
                "C(=O)O",  # carboxylic acid
                "C(=O)N",  # amide
                "S(=O)(=O)N",  # sulfonamide
                "C#N",  # nitrile
                "C(F)(F)F",  # trifluoromethyl
            ],
        }

    def _load_transformation_rules(
        self,
    ) -> Dict[MolecularTransformation, List[Dict[str, Any]]]:
        """Load molecular transformation rules."""
        return {
            MolecularTransformation.FUNCTIONAL_GROUP_REPLACEMENT: [
                {
                    "pattern": "[C:1](=O)[OH:2]",  # Carboxylic acid
                    "replacements": [
                        "[C:1](=O)[NH2]",
                        "[C:1](=O)[O][CH3]",
                    ],  # Amide, ester
                    "description": "Replace carboxylic acid with amide or ester",
                },
                {
                    "pattern": "[NH2:1]",  # Primary amine
                    "replacements": [
                        "[NH:1][CH3]",
                        "[N:1]([CH3])[CH3]",
                    ],  # Secondary, tertiary
                    "description": "Methylate primary amine",
                },
            ],
            MolecularTransformation.BIOISOSTERIC_REPLACEMENT: [
                {
                    "pattern": "c1ccccc1",  # Benzene
                    "replacements": [
                        "c1ccncc1",
                        "c1cccnc1",
                        "c1ccoc1",
                    ],  # Pyridine, pyrimidine, furan
                    "description": "Replace benzene with heterocycle",
                },
                {
                    "pattern": "[CH2:1][CH2:2]",  # Ethyl linker
                    "replacements": [
                        "[CH2:1][O:2]",
                        "[CH2:1][NH:2]",
                    ],  # Ether, amine linker
                    "description": "Replace carbon-carbon with heteroatom",
                },
            ],
        }

    def generate_candidates(
        self,
        parent_smiles: str,
        n_candidates: int,
        transformations: List[MolecularTransformation],
        **kwargs,
    ) -> List[str]:
        """Generate candidate molecules using fragment-based approach."""

        candidates = set()

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            parent_mol = Chem.MolFromSmiles(parent_smiles)
            if parent_mol is None:
                return []

            # Apply each transformation type
            for transformation in transformations:
                if transformation in self.transformation_rules:
                    rules = self.transformation_rules[transformation]

                    for rule in rules:
                        pattern = Chem.MolFromSmarts(rule["pattern"])
                        if pattern is None:
                            continue

                        matches = parent_mol.GetSubstructMatches(pattern)
                        if matches:
                            # Apply each replacement
                            for replacement_smarts in rule["replacements"]:
                                try:
                                    # This is simplified - would use proper reaction SMARTS
                                    candidate = self._apply_transformation(
                                        parent_smiles,
                                        rule["pattern"],
                                        replacement_smarts,
                                    )
                                    if candidate and candidate != parent_smiles:
                                        candidates.add(candidate)

                                except Exception:
                                    continue

                        if (
                            len(candidates) >= n_candidates * 2
                        ):  # Generate extras for filtering
                            break

                if len(candidates) >= n_candidates * 2:
                    break

            # Add some random variations if we don't have enough
            if len(candidates) < n_candidates:
                random_candidates = self._generate_random_variants(
                    parent_smiles, n_candidates
                )
                candidates.update(random_candidates)

            return list(candidates)[:n_candidates]

        except ImportError:
            # Fallback for demo - generate some plausible variations
            return self._generate_demo_candidates(parent_smiles, n_candidates)

    def _apply_transformation(
        self, parent_smiles: str, pattern: str, replacement: str
    ) -> Optional[str]:
        """Apply molecular transformation."""

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            # This is a simplified implementation
            # In practice, would use more sophisticated reaction templates

            mol = Chem.MolFromSmiles(parent_smiles)
            if mol is None:
                return None

            # Simple pattern-based replacement for demo
            if "c1ccccc1" in pattern and "c1ccncc1" in replacement:
                # Replace benzene with pyridine
                new_smiles = parent_smiles.replace("c1ccccc1", "c1ccncc1")
                new_mol = Chem.MolFromSmiles(new_smiles)
                if new_mol:
                    return Chem.MolToSmiles(new_mol)

            elif "[NH2]" in pattern and "[NH][CH3]" in replacement:
                # Methylate primary amine
                new_smiles = parent_smiles.replace("[NH2]", "[NH][CH3]")
                new_mol = Chem.MolFromSmiles(new_smiles)
                if new_mol:
                    return Chem.MolToSmiles(new_mol)

            return None

        except Exception:
            return None

    def _generate_random_variants(
        self, parent_smiles: str, n_variants: int
    ) -> List[str]:
        """Generate random molecular variants."""
        variants = []

        # Simple heuristic variants for demo
        base_variants = [
            parent_smiles + "C",  # Add methyl
            parent_smiles + "O",  # Add hydroxyl
            parent_smiles + "N",  # Add amine
            parent_smiles.replace("C", "N", 1),  # Replace first C with N
            parent_smiles.replace("C", "O", 1),  # Replace first C with O
        ]

        for variant in base_variants[:n_variants]:
            try:
                from rdkit import Chem

                mol = Chem.MolFromSmiles(variant)
                if mol:
                    variants.append(Chem.MolToSmiles(mol))
                else:
                    variants.append(variant)  # Keep even if not valid for demo
            except ImportError:
                variants.append(variant)

        return variants

    def _generate_demo_candidates(
        self, parent_smiles: str, n_candidates: int
    ) -> List[str]:
        """Generate demo candidates when RDKit not available."""
        candidates = []

        # Simple string-based modifications for demo
        modifications = [
            parent_smiles + "C",
            parent_smiles + "O",
            parent_smiles + "N",
            parent_smiles + "F",
            parent_smiles.replace("C", "N", 1)
            if "C" in parent_smiles
            else parent_smiles + "N",
            parent_smiles.replace("C", "O", 1)
            if "C" in parent_smiles
            else parent_smiles + "O",
            parent_smiles + "CC",
            parent_smiles + "CO",
            parent_smiles + "CN",
            parent_smiles + "CF",
        ]

        return modifications[:n_candidates]


class SaferByDesignOptimizer:
    """
    Multi-objective optimization system for safer-by-design chemicals.

    Combines molecular generation, toxicity prediction, and multi-objective
    optimization to design safer chemical alternatives.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.molecular_generator = FragmentBasedGenerator(backend)

        # Initialize predictors (lazy loading)
        self._toxicity_calculator = None
        self._property_predictor = None

        # Optimization parameters
        self.population_size = 100
        self.max_generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def optimize_for_safety(
        self,
        parent_smiles: str,
        target_properties: Dict[str, float],
        toxicity_constraints: List[ToxicityConstraint],
        optimization_objectives: Optional[List[OptimizationObjective]] = None,
        n_candidates: int = 20,
        max_iterations: int = 100,
        **kwargs,
    ) -> List[SafeDesignCandidate]:
        """
        Optimize molecular structure for safety and efficacy.

        Args:
            parent_smiles: Starting molecule
            target_properties: Desired functional properties
            toxicity_constraints: Toxicity limits to satisfy
            optimization_objectives: Objectives to optimize
            n_candidates: Number of final candidates to return
            max_iterations: Maximum optimization iterations

        Returns:
            List of optimized safer design candidates
        """

        # Default objectives if not specified
        if optimization_objectives is None:
            optimization_objectives = [
                OptimizationObjective.MINIMIZE_TOXICITY,
                OptimizationObjective.MAXIMIZE_EFFICACY,
                OptimizationObjective.MINIMIZE_PERSISTENCE,
            ]

        # Generate initial population
        initial_candidates = self._generate_initial_population(
            parent_smiles, self.population_size, **kwargs
        )

        # Evaluate initial population
        evaluated_candidates = self._evaluate_population(
            initial_candidates,
            target_properties,
            toxicity_constraints,
            optimization_objectives,
        )

        # Optimization loop
        best_candidates = evaluated_candidates

        for iteration in range(max_iterations):
            # Selection
            selected = self._selection(best_candidates)

            # Generate new candidates through mutation and crossover
            new_candidates = self._generate_new_generation(
                selected, parent_smiles, **kwargs
            )

            # Evaluate new candidates
            new_evaluated = self._evaluate_population(
                new_candidates,
                target_properties,
                toxicity_constraints,
                optimization_objectives,
            )

            # Combine and select best
            combined = best_candidates + new_evaluated
            best_candidates = self._pareto_selection(combined, n_candidates * 2)

            # Check convergence
            if iteration % 10 == 0:
                print(
                    f"Generation {iteration}: Best score = {max(c.optimization_score for c in best_candidates):.3f}"
                )

        # Final selection and ranking
        final_candidates = self._final_selection(best_candidates, n_candidates)

        return final_candidates

    def _generate_initial_population(
        self, parent_smiles: str, population_size: int, **kwargs
    ) -> List[str]:
        """Generate initial population of candidate molecules."""

        transformations = [
            MolecularTransformation.FUNCTIONAL_GROUP_REPLACEMENT,
            MolecularTransformation.BIOISOSTERIC_REPLACEMENT,
            MolecularTransformation.SIDE_CHAIN_MODIFICATION,
        ]

        candidates = [parent_smiles]  # Include parent

        # Generate variants using different transformation types
        for transformation in transformations:
            batch_size = population_size // len(transformations)
            batch_candidates = self.molecular_generator.generate_candidates(
                parent_smiles, batch_size, [transformation], **kwargs
            )
            candidates.extend(batch_candidates)

        # Fill remaining slots with random variants
        while len(candidates) < population_size:
            additional = self.molecular_generator.generate_candidates(
                parent_smiles,
                population_size - len(candidates),
                transformations,
                **kwargs,
            )
            candidates.extend(additional)

        # Remove duplicates and limit size
        unique_candidates = list(set(candidates))[:population_size]

        return unique_candidates

    def _evaluate_population(
        self,
        candidate_smiles: List[str],
        target_properties: Dict[str, float],
        toxicity_constraints: List[ToxicityConstraint],
        optimization_objectives: List[OptimizationObjective],
    ) -> List[SafeDesignCandidate]:
        """Evaluate population of candidates."""

        evaluated_candidates = []

        for smiles in candidate_smiles:
            try:
                candidate = self._evaluate_candidate(
                    smiles,
                    target_properties,
                    toxicity_constraints,
                    optimization_objectives,
                )
                evaluated_candidates.append(candidate)
            except Exception as e:
                print(f"Warning: Failed to evaluate candidate {smiles}: {e}")
                continue

        return evaluated_candidates

    def _evaluate_candidate(
        self,
        smiles: str,
        target_properties: Dict[str, float],
        toxicity_constraints: List[ToxicityConstraint],
        optimization_objectives: List[OptimizationObjective],
    ) -> SafeDesignCandidate:
        """Evaluate single candidate molecule."""

        # Predict toxicity
        toxicity_data = self.toxicity_calculator.predict_toxicity(smiles)

        # Predict other properties
        predicted_properties = self._predict_properties(smiles, target_properties)

        # Evaluate constraints
        constraint_violations = []
        total_penalty = 0.0

        for constraint in toxicity_constraints:
            endpoint_predictions = toxicity_data.get_endpoint_predictions(
                constraint.endpoint_type
            )
            if endpoint_predictions:
                pred_value = endpoint_predictions[0].predicted_value
                satisfied, penalty = constraint.evaluate(pred_value)

                if not satisfied and constraint.priority == "hard":
                    constraint_violations.append(
                        f"{constraint.endpoint_type.value} > {constraint.target_value}"
                    )

                total_penalty += penalty

        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            predicted_properties,
            target_properties,
            optimization_objectives,
            total_penalty,
        )

        # Calculate similarity to parent (placeholder)
        similarity = self._calculate_similarity(
            smiles, target_properties.get("parent_smiles", "")
        )

        # Estimate synthetic accessibility
        synthetic_accessibility = self._estimate_synthetic_accessibility(smiles)

        return SafeDesignCandidate(
            smiles=smiles,
            predicted_properties=predicted_properties,
            toxicity_predictions=toxicity_data,
            optimization_score=optimization_score,
            constraint_violations=constraint_violations,
            similarity_to_parent=similarity,
            synthetic_accessibility=synthetic_accessibility,
            confidence_score=self._calculate_confidence(toxicity_data),
        )

    def _predict_properties(
        self, smiles: str, target_properties: Dict[str, float]
    ) -> Dict[str, float]:
        """Predict molecular properties."""

        try:
            from rdkit import Chem
            from rdkit.Chem import Crippen, Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Return default values if molecule is invalid
                return {prop: 0.0 for prop in target_properties.keys()}

            properties = {}

            # Common properties
            properties["molecular_weight"] = Descriptors.MolWt(mol)
            properties["logp"] = Crippen.MolLogP(mol)
            properties["tpsa"] = Descriptors.TPSA(mol)
            properties["hbd"] = Descriptors.NumHDonors(mol)
            properties["hba"] = Descriptors.NumHAcceptors(mol)
            properties["rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)

            # Specific properties requested
            for prop_name in target_properties.keys():
                if prop_name not in properties:
                    # Use heuristic or ML model prediction
                    if "solubility" in prop_name.lower():
                        # Estimate solubility from LogP and polar surface area
                        properties[prop_name] = (
                            0.5 - 0.01 * properties["logp"] + 0.001 * properties["tpsa"]
                        )
                    elif (
                        "potency" in prop_name.lower()
                        or "efficacy" in prop_name.lower()
                    ):
                        # Placeholder for biological activity prediction
                        properties[prop_name] = np.random.rand()
                    elif "stability" in prop_name.lower():
                        # Estimate stability from molecular complexity
                        properties[prop_name] = 1.0 / (
                            1.0 + properties["rotatable_bonds"] * 0.1
                        )
                    else:
                        properties[prop_name] = np.random.rand()  # Placeholder

            return properties

        except ImportError:
            # Fallback for demo
            return {prop: np.random.rand() for prop in target_properties.keys()}

    def _calculate_optimization_score(
        self,
        predicted_properties: Dict[str, float],
        target_properties: Dict[str, float],
        optimization_objectives: List[OptimizationObjective],
        penalty: float,
    ) -> float:
        """Calculate multi-objective optimization score."""

        score = 0.0
        total_weight = 0.0

        # Property matching score
        for prop_name, target_value in target_properties.items():
            if prop_name in predicted_properties:
                pred_value = predicted_properties[prop_name]
                # Gaussian similarity with target
                diff = abs(pred_value - target_value) / max(abs(target_value), 1.0)
                property_score = np.exp(-(diff**2))
                score += property_score
                total_weight += 1.0

        # Objective-specific scores
        for objective in optimization_objectives:
            obj_score = self._evaluate_objective(objective, predicted_properties)
            weight = self._get_objective_weight(objective)
            score += obj_score * weight
            total_weight += weight

        # Normalize and apply penalty
        if total_weight > 0:
            normalized_score = score / total_weight
        else:
            normalized_score = 0.0

        final_score = max(0, normalized_score - penalty)

        return final_score

    def _evaluate_objective(
        self, objective: OptimizationObjective, predicted_properties: Dict[str, float]
    ) -> float:
        """Evaluate specific optimization objective."""

        if objective == OptimizationObjective.MINIMIZE_TOXICITY:
            # Higher molecular weight and polar surface area generally less toxic
            mw_score = min(predicted_properties.get("molecular_weight", 300) / 500, 1.0)
            tpsa_score = min(predicted_properties.get("tpsa", 50) / 140, 1.0)
            return (mw_score + tpsa_score) / 2

        elif objective == OptimizationObjective.MAXIMIZE_EFFICACY:
            return predicted_properties.get("potency", 0.5)

        elif objective == OptimizationObjective.MINIMIZE_PERSISTENCE:
            # More rotatable bonds and heteroatoms suggest faster degradation
            rot_bonds = predicted_properties.get("rotatable_bonds", 3)
            return min(rot_bonds / 10, 1.0)

        elif objective == OptimizationObjective.MINIMIZE_BIOACCUMULATION:
            # Lower LogP suggests less bioaccumulation
            logp = predicted_properties.get("logp", 2.0)
            return max(0, 1.0 - logp / 5.0)

        elif objective == OptimizationObjective.MAXIMIZE_BIODEGRADABILITY:
            # Presence of biodegradable functional groups
            return predicted_properties.get("biodegradability_score", 0.5)

        else:
            return 0.5  # Default neutral score

    def _get_objective_weight(self, objective: OptimizationObjective) -> float:
        """Get weight for optimization objective."""
        weights = {
            OptimizationObjective.MINIMIZE_TOXICITY: 2.0,
            OptimizationObjective.MAXIMIZE_EFFICACY: 1.5,
            OptimizationObjective.MINIMIZE_PERSISTENCE: 1.0,
            OptimizationObjective.MINIMIZE_BIOACCUMULATION: 1.0,
            OptimizationObjective.MAXIMIZE_BIODEGRADABILITY: 0.8,
            OptimizationObjective.MINIMIZE_VOLATILITY: 0.5,
            OptimizationObjective.MAXIMIZE_SELECTIVITY: 1.2,
            OptimizationObjective.MINIMIZE_COST: 0.3,
        }
        return weights.get(objective, 1.0)

    def _calculate_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate molecular similarity."""
        try:
            from rdkit import Chem
            from rdkit.Chem import DataStructs, rdMolDescriptors

            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return 0.0

            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)

            return DataStructs.TanimotoSimilarity(fp1, fp2)

        except ImportError:
            # Simple string similarity for demo
            if smiles2:
                return len(set(smiles1) & set(smiles2)) / len(
                    set(smiles1) | set(smiles2)
                )
            return 0.0

    def _estimate_synthetic_accessibility(self, smiles: str) -> float:
        """Estimate synthetic accessibility score (1-10, lower is better)."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 8.0  # Assume difficult if invalid

            # Simple heuristic based on complexity
            # In practice, would use trained SA score model
            complexity = (
                Descriptors.BertzCT(mol) / 100
                + len(mol.GetRings()) * 0.5
                + Descriptors.NumRotatableBonds(mol) * 0.1
            )

            sa_score = min(10, max(1, 3 + complexity))
            return sa_score

        except ImportError:
            # Random score for demo
            return np.random.uniform(2, 8)

    def _calculate_confidence(self, toxicity_data: ToxicityData) -> float:
        """Calculate overall confidence in predictions."""
        if not toxicity_data.predicted_endpoints:
            return 0.1

        confidences = [
            self._confidence_to_score(pred.confidence)
            for pred in toxicity_data.predicted_endpoints
        ]

        return np.mean(confidences)

    def _confidence_to_score(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numeric score."""
        mapping = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MODERATE: 0.5,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.VERY_HIGH: 0.9,
        }
        return mapping.get(confidence, 0.5)

    def _selection(
        self, candidates: List[SafeDesignCandidate]
    ) -> List[SafeDesignCandidate]:
        """Select candidates for next generation using tournament selection."""
        selected = []
        tournament_size = 3

        for _ in range(len(candidates) // 2):
            # Tournament selection
            tournament = np.random.choice(candidates, tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.overall_desirability)
            selected.append(winner)

        return selected

    def _generate_new_generation(
        self, parents: List[SafeDesignCandidate], parent_smiles: str, **kwargs
    ) -> List[str]:
        """Generate new candidate molecules through mutation and crossover."""
        new_candidates = []

        # Mutation
        for parent in parents:
            if np.random.rand() < self.mutation_rate:
                mutated = self._mutate_molecule(parent.smiles)
                if mutated:
                    new_candidates.append(mutated)

        # Crossover (simplified - just generate new variants from best parents)
        best_parents = sorted(
            parents, key=lambda x: x.overall_desirability, reverse=True
        )[:5]

        for parent in best_parents:
            variants = self.molecular_generator.generate_candidates(
                parent.smiles,
                2,
                [MolecularTransformation.FUNCTIONAL_GROUP_REPLACEMENT],
                **kwargs,
            )
            new_candidates.extend(variants)

        return new_candidates

    def _mutate_molecule(self, smiles: str) -> Optional[str]:
        """Apply random mutation to molecule."""
        mutations = self.molecular_generator.generate_candidates(
            smiles, 1, [MolecularTransformation.SIDE_CHAIN_MODIFICATION]
        )
        return mutations[0] if mutations else None

    def _pareto_selection(
        self, candidates: List[SafeDesignCandidate], n_select: int
    ) -> List[SafeDesignCandidate]:
        """Select candidates using Pareto dominance."""
        # Simplified Pareto selection - rank by overall desirability
        ranked = sorted(candidates, key=lambda x: x.overall_desirability, reverse=True)
        return ranked[:n_select]

    def _final_selection(
        self, candidates: List[SafeDesignCandidate], n_final: int
    ) -> List[SafeDesignCandidate]:
        """Final selection and ranking of candidates."""

        # Filter viable candidates (meet hard constraints)
        viable = [c for c in candidates if c.is_viable]

        if len(viable) < n_final:
            # Include some non-viable candidates if needed
            non_viable = [c for c in candidates if not c.is_viable]
            non_viable_sorted = sorted(
                non_viable, key=lambda x: x.overall_desirability, reverse=True
            )
            viable.extend(non_viable_sorted[: n_final - len(viable)])

        # Final ranking by overall desirability
        final = sorted(viable, key=lambda x: x.overall_desirability, reverse=True)

        return final[:n_final]

    @property
    def toxicity_calculator(self):
        """Get toxicity calculator."""
        if self._toxicity_calculator is None:
            from .core import ToxicologyCalculator

            self._toxicity_calculator = ToxicologyCalculator(self.backend)
        return self._toxicity_calculator

    def generate_optimization_report(
        self,
        parent_smiles: str,
        candidates: List[SafeDesignCandidate],
        target_properties: Dict[str, float],
        toxicity_constraints: List[ToxicityConstraint],
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""

        viable_candidates = [c for c in candidates if c.is_viable]
        best_candidate = candidates[0] if candidates else None

        report = {
            "optimization_summary": {
                "parent_compound": parent_smiles,
                "total_candidates_evaluated": len(candidates),
                "viable_candidates": len(viable_candidates),
                "success_rate": len(viable_candidates) / len(candidates)
                if candidates
                else 0,
                "best_candidate": best_candidate.smiles if best_candidate else None,
                "best_score": best_candidate.overall_desirability
                if best_candidate
                else 0,
            },
            "constraint_analysis": {
                "total_constraints": len(toxicity_constraints),
                "constraints_satisfied": sum(
                    1
                    for c in candidates[:5]
                    for constraint in toxicity_constraints
                    if not any(
                        constraint.endpoint_type.value in violation
                        for violation in c.constraint_violations
                    )
                ),
                "common_violations": self._analyze_common_violations(candidates),
            },
            "property_analysis": {
                "target_properties": target_properties,
                "property_improvements": self._analyze_property_improvements(
                    parent_smiles, candidates[:5], target_properties
                ),
            },
            "top_candidates": [
                {
                    "rank": i + 1,
                    "smiles": candidate.smiles,
                    "overall_score": candidate.overall_desirability,
                    "optimization_score": candidate.optimization_score,
                    "viable": candidate.is_viable,
                    "violations": candidate.constraint_violations,
                    "similarity": candidate.similarity_to_parent,
                    "synthetic_accessibility": candidate.synthetic_accessibility,
                }
                for i, candidate in enumerate(candidates[:10])
            ],
            "recommendations": self._generate_optimization_recommendations(
                candidates, toxicity_constraints
            ),
        }

        return report

    def _analyze_common_violations(
        self, candidates: List[SafeDesignCandidate]
    ) -> Dict[str, int]:
        """Analyze most common constraint violations."""
        violation_counts = {}

        for candidate in candidates:
            for violation in candidate.constraint_violations:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1

        return dict(sorted(violation_counts.items(), key=lambda x: x[1], reverse=True))

    def _analyze_property_improvements(
        self,
        parent_smiles: str,
        candidates: List[SafeDesignCandidate],
        target_properties: Dict[str, float],
    ) -> Dict[str, Any]:
        """Analyze property improvements over parent."""

        # Predict parent properties
        parent_properties = self._predict_properties(parent_smiles, target_properties)

        improvements = {}

        for prop_name, target_value in target_properties.items():
            parent_value = parent_properties.get(prop_name, 0)
            parent_error = abs(parent_value - target_value)

            candidate_errors = []
            for candidate in candidates:
                if prop_name in candidate.predicted_properties:
                    cand_value = candidate.predicted_properties[prop_name]
                    cand_error = abs(cand_value - target_value)
                    candidate_errors.append(cand_error)

            if candidate_errors:
                best_error = min(candidate_errors)
                avg_error = np.mean(candidate_errors)

                improvements[prop_name] = {
                    "parent_error": parent_error,
                    "best_candidate_error": best_error,
                    "average_candidate_error": avg_error,
                    "improvement_ratio": parent_error / max(best_error, 1e-6),
                    "success_rate": sum(1 for e in candidate_errors if e < parent_error)
                    / len(candidate_errors),
                }

        return improvements

    def _generate_optimization_recommendations(
        self,
        candidates: List[SafeDesignCandidate],
        toxicity_constraints: List[ToxicityConstraint],
    ) -> List[str]:
        """Generate optimization recommendations."""

        recommendations = []

        viable_candidates = [c for c in candidates if c.is_viable]

        if not viable_candidates:
            recommendations.append(
                "No candidates satisfied all hard constraints - consider relaxing constraints"
            )

            # Analyze constraint violations
            violation_counts = self._analyze_common_violations(candidates)
            if violation_counts:
                most_common = list(violation_counts.keys())[0]
                recommendations.append(
                    f"Most common constraint violation: {most_common}"
                )
        else:
            recommendations.append(
                f"Successfully identified {len(viable_candidates)} viable candidates"
            )

            best = viable_candidates[0]
            recommendations.append(
                f"Best candidate shows {best.overall_desirability:.2f} overall desirability"
            )

            if best.synthetic_accessibility and best.synthetic_accessibility > 6:
                recommendations.append(
                    "Consider synthetic accessibility in final selection"
                )

        # Confidence-based recommendations
        low_confidence = [c for c in candidates[:5] if c.confidence_score < 0.4]
        if low_confidence:
            recommendations.append(
                "Some predictions have low confidence - consider experimental validation"
            )

        # Similarity recommendations
        avg_similarity = np.mean([c.similarity_to_parent for c in candidates[:5]])
        if avg_similarity < 0.3:
            recommendations.append(
                "Candidates show low similarity to parent - verify functional properties"
            )
        elif avg_similarity > 0.8:
            recommendations.append("Consider more diverse structural modifications")

        return recommendations

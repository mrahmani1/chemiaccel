"""
Enhanced Generative Chemistry module with IC-KG integration for ChemiAccel.

This module generates novel molecules using various generative models while leveraging
contextual information from the Integrated Chemistry Knowledge Graph for intelligent
molecular design and validation.
"""

import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import jax
    import jax.numpy as jnp
    from jax import random

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. Some generative features will be limited.")

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Molecular validation will be limited.")

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


class GenerativeChemistryError(ChemiAccelError):
    """Raised when generative chemistry operations fail."""

    pass


class GenerationStrategy(Enum):
    """Strategies for molecule generation."""

    RANDOM_SAMPLING = "random_sampling"
    GUIDED_SAMPLING = "guided_sampling"
    CONSTRAINT_BASED = "constraint_based"
    SIMILARITY_BASED = "similarity_based"
    KNOWLEDGE_GUIDED = "knowledge_guided"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class GenerationConfig:
    """Configuration for molecule generation."""

    strategy: GenerationStrategy = GenerationStrategy.GUIDED_SAMPLING
    num_molecules: int = 100
    max_attempts: int = 1000
    diversity_weight: float = 0.3
    novelty_weight: float = 0.4
    feasibility_weight: float = 0.3

    # Molecular constraints
    molecular_weight_range: Tuple[float, float] = (150.0, 500.0)
    logp_range: Tuple[float, float] = (-2.0, 5.0)
    tpsa_max: float = 140.0
    rotatable_bonds_max: int = 10

    # KG-based constraints
    use_kg_context: bool = True
    kg_context_types: List[KGContextType] = field(
        default_factory=lambda: [
            KGContextType.MOLECULAR_SIMILARITY,
            KGContextType.STRUCTURE_ACTIVITY,
            KGContextType.CHEMICAL_CLASS,
        ]
    )
    similarity_threshold: float = 0.7
    known_actives_weight: float = 0.5

    # Generation parameters
    temperature: float = 1.0
    top_k: int = 50
    seed: Optional[int] = None


@dataclass
class GeneratedMolecule:
    """Information about a generated molecule."""

    smiles: str
    generation_score: float
    properties: Dict[str, float] = field(default_factory=dict)
    kg_context: Optional[Any] = None
    constraints_satisfied: Dict[str, bool] = field(default_factory=dict)
    similarity_to_knowns: List[Tuple[str, float]] = field(default_factory=list)
    generation_rationale: str = ""
    validation_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Results from molecule generation."""

    molecules: List[GeneratedMolecule]
    generation_config: GenerationConfig
    statistics: Dict[str, Any] = field(default_factory=dict)
    kg_update_suggestions: List[KGUpdateSuggestion] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MolecularGenerator(ABC):
    """Abstract base class for molecular generators."""

    @abstractmethod
    def generate_molecules(
        self, config: GenerationConfig, seed_molecules: Optional[List[str]] = None
    ) -> List[str]:
        """Generate molecules based on configuration."""
        pass

    @abstractmethod
    def get_generation_probability(self, molecule: str) -> float:
        """Get probability of generating a specific molecule."""
        pass


class VAEGenerator(MolecularGenerator):
    """Variational Autoencoder-based molecular generator."""

    def __init__(self, model_path: Optional[str] = None, latent_dim: int = 512):
        """Initialize VAE generator."""
        self.model_path = model_path
        self.latent_dim = latent_dim
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load pre-trained VAE model."""
        # Placeholder for model loading
        # In practice, this would load a trained VAE model
        logging.info("VAE model loading not implemented. Using placeholder.")
        self.model = "placeholder_vae"

    def generate_molecules(
        self, config: GenerationConfig, seed_molecules: Optional[List[str]] = None
    ) -> List[str]:
        """Generate molecules using VAE sampling."""
        if not JAX_AVAILABLE:
            raise GenerativeChemistryError("JAX required for VAE generation")

        key = random.PRNGKey(config.seed or 42)
        generated_smiles = []

        for _ in range(config.num_molecules):
            key, subkey = random.split(key)

            # Sample from latent space
            z = random.normal(subkey, (self.latent_dim,))

            # Apply temperature scaling
            z = z * config.temperature

            # Decode to SMILES (placeholder implementation)
            smiles = self._decode_latent_vector(z, seed_molecules)
            if smiles and self._is_valid_molecule(smiles):
                generated_smiles.append(smiles)

        return generated_smiles[: config.num_molecules]

    def _decode_latent_vector(
        self, z: jnp.ndarray, seed_molecules: Optional[List[str]]
    ) -> str:
        """Decode latent vector to SMILES."""
        # Placeholder implementation
        # In practice, this would use the decoder part of the VAE
        if seed_molecules:
            # Simple perturbation of seed molecules
            base_mol = np.random.choice(seed_molecules)
            return self._perturb_molecule(base_mol)
        else:
            # Generate random SMILES for demonstration
            return self._generate_random_smiles()

    def _perturb_molecule(self, smiles: str) -> str:
        """Create a perturbed version of a molecule."""
        if not RDKIT_AVAILABLE:
            return smiles

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # Simple perturbation: add or remove atoms (placeholder)
        return smiles  # Return original for now

    def _generate_random_smiles(self) -> str:
        """Generate a random valid SMILES string."""
        # Placeholder implementation
        simple_molecules = [
            "CCO",
            "CC(=O)O",
            "CC(C)O",
            "c1ccccc1",
            "CCN",
            "CC(=O)N",
            "CCC(=O)O",
            "c1ccccc1O",
            "CC(C)(C)O",
            "CC(=O)OC",
            "CCCN",
            "c1ccc(O)cc1",
            "CC(C)N",
        ]
        return np.random.choice(simple_molecules)

    def get_generation_probability(self, molecule: str) -> float:
        """Get generation probability for a molecule."""
        # Placeholder implementation
        return np.random.uniform(0.1, 0.9)

    def _is_valid_molecule(self, smiles: str) -> bool:
        """Check if SMILES represents a valid molecule."""
        if not RDKIT_AVAILABLE:
            return len(smiles) > 0 and smiles.isascii()

        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False


class RNNGenerator(MolecularGenerator):
    """RNN-based molecular generator."""

    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 100):
        """Initialize RNN generator."""
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load pre-trained RNN model."""
        logging.info("RNN model loading not implemented. Using placeholder.")
        self.model = "placeholder_rnn"

    def generate_molecules(
        self, config: GenerationConfig, seed_molecules: Optional[List[str]] = None
    ) -> List[str]:
        """Generate molecules using RNN sampling."""
        generated_smiles = []

        for _ in range(config.num_molecules):
            if seed_molecules and np.random.random() < 0.5:
                # Start with seed molecule
                seed = np.random.choice(seed_molecules)
                smiles = self._generate_from_seed(seed, config)
            else:
                # Generate from scratch
                smiles = self._generate_from_scratch(config)

            if smiles and self._is_valid_molecule(smiles):
                generated_smiles.append(smiles)

        return generated_smiles[: config.num_molecules]

    def _generate_from_seed(self, seed: str, config: GenerationConfig) -> str:
        """Generate molecule starting from seed."""
        # Placeholder implementation
        return self._generate_random_smiles()

    def _generate_from_scratch(self, config: GenerationConfig) -> str:
        """Generate molecule from scratch."""
        # Placeholder implementation
        return self._generate_random_smiles()

    def _generate_random_smiles(self) -> str:
        """Generate a random valid SMILES string."""
        simple_molecules = [
            "CCO",
            "CC(=O)O",
            "CC(C)O",
            "c1ccccc1",
            "CCN",
            "CC(=O)N",
            "CCC(=O)O",
            "c1ccccc1O",
            "CC(C)(C)O",
        ]
        return np.random.choice(simple_molecules)

    def get_generation_probability(self, molecule: str) -> float:
        """Get generation probability for a molecule."""
        return np.random.uniform(0.1, 0.9)

    def _is_valid_molecule(self, smiles: str) -> bool:
        """Check if SMILES represents a valid molecule."""
        if not RDKIT_AVAILABLE:
            return len(smiles) > 0 and smiles.isascii()

        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False


class GraphGenerator(MolecularGenerator):
    """Graph-based molecular generator."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize graph generator."""
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load pre-trained graph model."""
        logging.info("Graph model loading not implemented. Using placeholder.")
        self.model = "placeholder_graph"

    def generate_molecules(
        self, config: GenerationConfig, seed_molecules: Optional[List[str]] = None
    ) -> List[str]:
        """Generate molecules using graph-based approach."""
        generated_smiles = []

        for _ in range(config.num_molecules):
            smiles = self._generate_graph_molecule(config, seed_molecules)
            if smiles and self._is_valid_molecule(smiles):
                generated_smiles.append(smiles)

        return generated_smiles[: config.num_molecules]

    def _generate_graph_molecule(
        self, config: GenerationConfig, seed_molecules: Optional[List[str]]
    ) -> str:
        """Generate molecule using graph operations."""
        # Placeholder implementation
        if seed_molecules:
            return np.random.choice(seed_molecules)
        else:
            return self._generate_random_smiles()

    def _generate_random_smiles(self) -> str:
        """Generate a random valid SMILES string."""
        simple_molecules = ["CCO", "CC(=O)O", "CC(C)O", "c1ccccc1", "CCN"]
        return np.random.choice(simple_molecules)

    def get_generation_probability(self, molecule: str) -> float:
        """Get generation probability for a molecule."""
        return np.random.uniform(0.1, 0.9)

    def _is_valid_molecule(self, smiles: str) -> bool:
        """Check if SMILES represents a valid molecule."""
        if not RDKIT_AVAILABLE:
            return len(smiles) > 0 and smiles.isascii()

        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False


class KGContextualGenerativeChemistry:
    """
    Knowledge Graph-enhanced Generative Chemistry module.

    This class integrates molecular generation with the IC-KG to provide
    contextually-informed molecule design with intelligent suggestions and explanations.
    """

    def __init__(
        self,
        calculator: Calculator,
        kg_integration_manager: Optional["AIKGIntegrationManager"] = None,
        generator_type: str = "vae",
    ):
        """
        Initialize KG-contextual generative chemistry.

        Args:
            calculator: ChemiAccel calculator for property prediction
            kg_integration_manager: Manager for KG integration
            generator_type: Type of molecular generator ("vae", "rnn", "graph")
        """
        self.calculator = calculator
        self.kg_integration_manager = kg_integration_manager

        # Initialize molecular generator
        if generator_type == "vae":
            self.generator = VAEGenerator()
        elif generator_type == "rnn":
            self.generator = RNNGenerator()
        elif generator_type == "graph":
            self.generator = GraphGenerator()
        else:
            raise GenerativeChemistryError(f"Unknown generator type: {generator_type}")

        self.generation_history = []
        self.kg_update_history = []

    async def generate_contextual_molecules(
        self,
        config: GenerationConfig,
        seed_molecules: Optional[List[str]] = None,
        target_properties: Optional[Dict[str, float]] = None,
    ) -> GenerationResult:
        """
        Generate molecules with knowledge graph context.

        Args:
            config: Generation configuration
            seed_molecules: Optional seed molecules for guided generation
            target_properties: Optional target property values

        Returns:
            GenerationResult with contextually-informed molecules
        """
        start_time = datetime.now()

        # Step 1: Generate base molecules
        logging.info(
            f"Generating {config.num_molecules} molecules using {config.strategy.value}"
        )
        base_molecules = self.generator.generate_molecules(config, seed_molecules)

        if not base_molecules:
            raise GenerativeChemistryError("No valid molecules generated")

        # Step 2: Calculate molecular properties
        logging.info("Calculating molecular properties...")
        properties_df = self._calculate_properties(base_molecules)

        # Step 3: Apply molecular filters
        filtered_molecules = self._apply_molecular_filters(
            base_molecules, properties_df, config
        )

        if not filtered_molecules:
            raise GenerativeChemistryError("No molecules passed filtering criteria")

        # Step 4: Get KG context if available
        kg_contexts = {}
        if config.use_kg_context and self.kg_integration_manager:
            logging.info("Retrieving knowledge graph context...")
            kg_contexts = await self.kg_integration_manager.context_provider.get_molecular_context(
                filtered_molecules, config.kg_context_types
            )

        # Step 5: Score and rank molecules
        logging.info("Scoring and ranking molecules...")
        scored_molecules = await self._score_molecules(
            filtered_molecules, properties_df, kg_contexts, config, target_properties
        )

        # Step 6: Select top molecules based on multi-objective optimization
        final_molecules = self._select_final_molecules(scored_molecules, config)

        # Step 7: Generate contextual predictions if KG integration available
        contextual_predictions = []
        kg_update_suggestions = []

        if self.kg_integration_manager:
            logging.info("Generating contextual predictions...")
            for mol in final_molecules:
                prediction = {
                    "molecules": [{"smiles": mol.smiles, "properties": mol.properties}],
                    "generation_method": "generative_chemistry",
                    "score": mol.generation_score,
                }

                contextual_pred = (
                    await self.kg_integration_manager.enrich_ai_prediction(
                        prediction=prediction,
                        molecules=[mol.smiles],
                        task_type=AITaskType.GENERATIVE_CHEMISTRY,
                        context_types=config.kg_context_types,
                    )
                )

                contextual_predictions.append(contextual_pred)
                mol.kg_context = contextual_pred.kg_context
                mol.generation_rationale = contextual_pred.explanation

            # Generate KG update suggestions
            kg_update_suggestions = (
                self.kg_integration_manager.feedback_manager.suggest_kg_updates(
                    contextual_predictions, AITaskType.GENERATIVE_CHEMISTRY
                )
            )

        # Step 8: Compile results
        execution_time = (datetime.now() - start_time).total_seconds()

        statistics = {
            "total_generated": len(base_molecules),
            "passed_filters": len(filtered_molecules),
            "final_selected": len(final_molecules),
            "avg_generation_score": np.mean(
                [m.generation_score for m in final_molecules]
            ),
            "kg_contexts_retrieved": len(kg_contexts),
            "contextual_predictions": len(contextual_predictions),
        }

        result = GenerationResult(
            molecules=final_molecules,
            generation_config=config,
            statistics=statistics,
            kg_update_suggestions=kg_update_suggestions,
            execution_time=execution_time,
            metadata={
                "generator_type": type(self.generator).__name__,
                "kg_integration_enabled": self.kg_integration_manager is not None,
                "seed_molecules_used": seed_molecules is not None,
            },
        )

        # Store in history
        self.generation_history.append(result)

        logging.info(
            f"Generation completed in {execution_time:.2f}s. "
            f"Selected {len(final_molecules)} molecules."
        )

        return result

    def _calculate_properties(self, molecules: List[str]) -> pd.DataFrame:
        """Calculate molecular properties using ChemiAccel."""
        try:
            mol_series = pd.Series(molecules)
            properties_df = self.calculator.pandas(mol_series)
            properties_df["smiles"] = molecules
            return properties_df
        except Exception as e:
            logging.error(f"Error calculating properties: {e}")
            # Return basic properties if calculator fails
            basic_props = {
                "smiles": molecules,
                "molecular_weight": [0.0] * len(molecules),
                "logp": [0.0] * len(molecules),
                "tpsa": [0.0] * len(molecules),
            }
            return pd.DataFrame(basic_props)

    def _apply_molecular_filters(
        self,
        molecules: List[str],
        properties_df: pd.DataFrame,
        config: GenerationConfig,
    ) -> List[str]:
        """Apply molecular filters based on constraints."""
        filtered = []

        for i, smiles in enumerate(molecules):
            if i >= len(properties_df):
                continue

            props = properties_df.iloc[i]

            # Check molecular weight
            mw = props.get("molecular_weight", 0)
            if not (
                config.molecular_weight_range[0]
                <= mw
                <= config.molecular_weight_range[1]
            ):
                continue

            # Check LogP
            logp = props.get("logp", 0)
            if not (config.logp_range[0] <= logp <= config.logp_range[1]):
                continue

            # Check TPSA
            tpsa = props.get("tpsa", 0)
            if tpsa > config.tpsa_max:
                continue

            # Check rotatable bonds
            rot_bonds = props.get("num_rotatable_bonds", 0)
            if rot_bonds > config.rotatable_bonds_max:
                continue

            filtered.append(smiles)

        return filtered

    async def _score_molecules(
        self,
        molecules: List[str],
        properties_df: pd.DataFrame,
        kg_contexts: Dict[str, Any],
        config: GenerationConfig,
        target_properties: Optional[Dict[str, float]],
    ) -> List[GeneratedMolecule]:
        """Score molecules based on multiple criteria."""
        scored_molecules = []

        for smiles in molecules:
            mol_props = (
                properties_df[properties_df["smiles"] == smiles].iloc[0]
                if not properties_df.empty
                else {}
            )

            # Base generation score
            generation_prob = self.generator.get_generation_probability(smiles)

            # Property-based scoring
            property_score = self._calculate_property_score(
                mol_props, target_properties
            )

            # KG-based scoring
            kg_score = self._calculate_kg_score(smiles, kg_contexts.get(smiles))

            # Diversity score
            diversity_score = self._calculate_diversity_score(smiles, scored_molecules)

            # Combined score
            total_score = (
                config.feasibility_weight * generation_prob
                + (1 - config.novelty_weight - config.diversity_weight) * property_score
                + config.known_actives_weight * kg_score
                + config.diversity_weight * diversity_score
            )

            # Create GeneratedMolecule object
            generated_mol = GeneratedMolecule(
                smiles=smiles,
                generation_score=total_score,
                properties=mol_props.to_dict() if hasattr(mol_props, "to_dict") else {},
                constraints_satisfied=self._check_constraints(mol_props, config),
                similarity_to_knowns=self._find_similar_molecules(
                    smiles, kg_contexts.get(smiles)
                ),
            )

            scored_molecules.append(generated_mol)

        return scored_molecules

    def _calculate_property_score(
        self, mol_props: Dict[str, Any], target_properties: Optional[Dict[str, float]]
    ) -> float:
        """Calculate score based on molecular properties."""
        if not target_properties:
            return 0.5  # Neutral score if no targets

        score = 0.0
        count = 0

        for prop_name, target_value in target_properties.items():
            if prop_name in mol_props:
                actual_value = mol_props[prop_name]
                # Gaussian similarity to target
                diff = abs(actual_value - target_value)
                prop_score = np.exp(-(diff**2) / (2 * (target_value * 0.1) ** 2))
                score += prop_score
                count += 1

        return score / count if count > 0 else 0.0

    def _calculate_kg_score(self, smiles: str, kg_context: Optional[Any]) -> float:
        """Calculate score based on knowledge graph context."""
        if not kg_context:
            return 0.0

        score = 0.0

        # Score based on similarity to known actives
        similar_molecules = [e for e in kg_context.entities if "similarity_score" in e]
        if similar_molecules:
            max_similarity = max(e["similarity_score"] for e in similar_molecules)
            score += max_similarity * 0.4

        # Score based on known activities
        activities = [e for e in kg_context.entities if "activity_type" in e]
        if activities:
            avg_confidence = np.mean([e.get("confidence", 0.0) for e in activities])
            score += avg_confidence * 0.3

        # Score based on target interactions
        target_rels = [r for r in kg_context.relationships if r["type"] == "TARGETS"]
        if target_rels:
            avg_confidence = np.mean([r.get("confidence", 0.0) for r in target_rels])
            score += avg_confidence * 0.3

        return min(1.0, score)

    def _calculate_diversity_score(
        self, smiles: str, existing_molecules: List[GeneratedMolecule]
    ) -> float:
        """Calculate diversity score relative to existing molecules."""
        if not existing_molecules:
            return 1.0

        if not RDKIT_AVAILABLE:
            return 0.5  # Neutral score if RDKit not available

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            fp = Chem.RDKFingerprint(mol)

            max_similarity = 0.0
            for existing_mol in existing_molecules:
                existing_rdkit_mol = Chem.MolFromSmiles(existing_mol.smiles)
                if existing_rdkit_mol:
                    existing_fp = Chem.RDKFingerprint(existing_rdkit_mol)
                    similarity = Chem.DataStructs.TanimotoSimilarity(fp, existing_fp)
                    max_similarity = max(max_similarity, similarity)

            # Return inverse of maximum similarity (higher diversity = lower max similarity)
            return 1.0 - max_similarity

        except Exception:
            return 0.5

    def _check_constraints(
        self, mol_props: Dict[str, Any], config: GenerationConfig
    ) -> Dict[str, bool]:
        """Check which constraints are satisfied."""
        constraints = {}

        mw = mol_props.get("molecular_weight", 0)
        constraints["molecular_weight"] = (
            config.molecular_weight_range[0] <= mw <= config.molecular_weight_range[1]
        )

        logp = mol_props.get("logp", 0)
        constraints["logp"] = config.logp_range[0] <= logp <= config.logp_range[1]

        tpsa = mol_props.get("tpsa", 0)
        constraints["tpsa"] = tpsa <= config.tpsa_max

        rot_bonds = mol_props.get("num_rotatable_bonds", 0)
        constraints["rotatable_bonds"] = rot_bonds <= config.rotatable_bonds_max

        return constraints

    def _find_similar_molecules(
        self, smiles: str, kg_context: Optional[Any]
    ) -> List[Tuple[str, float]]:
        """Find similar molecules from KG context."""
        if not kg_context:
            return []

        similar_mols = []
        for entity in kg_context.entities:
            if "similarity_score" in entity and "smiles" in entity:
                similar_mols.append((entity["smiles"], entity["similarity_score"]))

        return sorted(similar_mols, key=lambda x: x[1], reverse=True)[:5]

    def _select_final_molecules(
        self, scored_molecules: List[GeneratedMolecule], config: GenerationConfig
    ) -> List[GeneratedMolecule]:
        """Select final molecules based on multi-objective optimization."""
        # Sort by score
        sorted_molecules = sorted(
            scored_molecules, key=lambda m: m.generation_score, reverse=True
        )

        # Apply diversity selection
        final_molecules = []
        for mol in sorted_molecules:
            if len(final_molecules) >= config.num_molecules:
                break

            # Check diversity against already selected
            if self._is_sufficiently_diverse(mol, final_molecules, threshold=0.7):
                final_molecules.append(mol)

        return final_molecules

    def _is_sufficiently_diverse(
        self,
        candidate: GeneratedMolecule,
        selected: List[GeneratedMolecule],
        threshold: float = 0.7,
    ) -> bool:
        """Check if candidate is sufficiently diverse from selected molecules."""
        if not selected:
            return True

        if not RDKIT_AVAILABLE:
            return True  # Allow if RDKit not available

        try:
            candidate_mol = Chem.MolFromSmiles(candidate.smiles)
            if candidate_mol is None:
                return False

            candidate_fp = Chem.RDKFingerprint(candidate_mol)

            for selected_mol in selected:
                selected_rdkit_mol = Chem.MolFromSmiles(selected_mol.smiles)
                if selected_rdkit_mol:
                    selected_fp = Chem.RDKFingerprint(selected_rdkit_mol)
                    similarity = Chem.DataStructs.TanimotoSimilarity(
                        candidate_fp, selected_fp
                    )
                    if similarity > threshold:
                        return False

            return True

        except Exception:
            return True  # Allow if calculation fails

    async def update_knowledge_graph(
        self, generation_results: List[GenerationResult]
    ) -> Dict[str, Any]:
        """Update knowledge graph based on generation results."""
        if not self.kg_integration_manager:
            return {"error": "KG integration not available"}

        all_contextual_predictions = []

        for result in generation_results:
            for molecule in result.molecules:
                if hasattr(molecule, "kg_context") and molecule.kg_context:
                    prediction = {
                        "molecules": [
                            {
                                "smiles": molecule.smiles,
                                "properties": molecule.properties,
                            }
                        ],
                        "generation_method": "generative_chemistry",
                        "score": molecule.generation_score,
                    }

                    contextual_pred = (
                        await self.kg_integration_manager.enrich_ai_prediction(
                            prediction=prediction,
                            molecules=[molecule.smiles],
                            task_type=AITaskType.GENERATIVE_CHEMISTRY,
                        )
                    )

                    all_contextual_predictions.append(contextual_pred)

        # Update KG
        update_results = await self.kg_integration_manager.update_kg_from_ai(
            all_contextual_predictions, AITaskType.GENERATIVE_CHEMISTRY
        )

        self.kg_update_history.append(update_results)

        return update_results

    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of generation history."""
        if not self.generation_history:
            return {"message": "No generations completed yet"}

        total_molecules = sum(
            len(result.molecules) for result in self.generation_history
        )
        avg_execution_time = np.mean(
            [result.execution_time for result in self.generation_history]
        )

        return {
            "total_generations": len(self.generation_history),
            "total_molecules_generated": total_molecules,
            "average_execution_time": avg_execution_time,
            "kg_updates_performed": len(self.kg_update_history),
            "latest_generation_stats": (
                self.generation_history[-1].statistics
                if self.generation_history
                else {}
            ),
        }


# Utility functions for generative chemistry
def create_drug_like_generation_config() -> GenerationConfig:
    """Create generation config optimized for drug-like molecules."""
    return GenerationConfig(
        strategy=GenerationStrategy.KNOWLEDGE_GUIDED,
        num_molecules=50,
        molecular_weight_range=(150.0, 500.0),
        logp_range=(-2.0, 5.0),
        tpsa_max=140.0,
        rotatable_bonds_max=10,
        use_kg_context=True,
        kg_context_types=[
            KGContextType.MOLECULAR_SIMILARITY,
            KGContextType.DRUG_TARGET,
            KGContextType.STRUCTURE_ACTIVITY,
            KGContextType.THERAPEUTIC_INDICATION,
        ],
        similarity_threshold=0.7,
        known_actives_weight=0.6,
        diversity_weight=0.3,
        novelty_weight=0.4,
        feasibility_weight=0.3,
    )


def create_lead_optimization_config() -> GenerationConfig:
    """Create generation config for lead optimization."""
    return GenerationConfig(
        strategy=GenerationStrategy.SIMILARITY_BASED,
        num_molecules=30,
        molecular_weight_range=(200.0, 400.0),
        logp_range=(1.0, 4.0),
        tpsa_max=100.0,
        rotatable_bonds_max=8,
        use_kg_context=True,
        kg_context_types=[
            KGContextType.MOLECULAR_SIMILARITY,
            KGContextType.STRUCTURE_ACTIVITY,
            KGContextType.PROPERTY_CORRELATION,
        ],
        similarity_threshold=0.8,
        known_actives_weight=0.7,
        diversity_weight=0.2,
        novelty_weight=0.3,
        feasibility_weight=0.5,
    )


async def run_contextual_generation_campaign(
    calculator: Calculator,
    kg_integration_manager: "AIKGIntegrationManager",
    seed_molecules: List[str],
    target_properties: Dict[str, float],
    n_rounds: int = 3,
    molecules_per_round: int = 50,
) -> List[GenerationResult]:
    """
    Run a multi-round contextual generation campaign.

    Args:
        calculator: ChemiAccel calculator
        kg_integration_manager: KG integration manager
        seed_molecules: Seed molecules for generation
        target_properties: Target properties to optimize
        n_rounds: Number of generation rounds
        molecules_per_round: Molecules to generate per round

    Returns:
        List of GenerationResult from each round
    """
    generator = KGContextualGenerativeChemistry(
        calculator=calculator,
        kg_integration_manager=kg_integration_manager,
        generator_type="vae",
    )

    results = []
    current_seeds = seed_molecules.copy()

    for round_num in range(n_rounds):
        logging.info(f"\n=== Generation Round {round_num + 1}/{n_rounds} ===")

        # Create round-specific config
        config = GenerationConfig(
            strategy=GenerationStrategy.KNOWLEDGE_GUIDED,
            num_molecules=molecules_per_round,
            use_kg_context=True,
            kg_context_types=[
                KGContextType.MOLECULAR_SIMILARITY,
                KGContextType.STRUCTURE_ACTIVITY,
                KGContextType.PROPERTY_CORRELATION,
            ],
        )

        # Generate molecules
        result = await generator.generate_contextual_molecules(
            config=config,
            seed_molecules=current_seeds,
            target_properties=target_properties,
        )

        results.append(result)

        # Update knowledge graph
        await generator.update_knowledge_graph([result])

        # Use top molecules as seeds for next round
        top_molecules = sorted(
            result.molecules, key=lambda m: m.generation_score, reverse=True
        )
        current_seeds = [mol.smiles for mol in top_molecules[:10]]

        logging.info(
            f"Round {round_num + 1} completed. "
            f"Generated {len(result.molecules)} molecules. "
            f"Average score: {np.mean([m.generation_score for m in result.molecules]):.3f}"
        )

    return results

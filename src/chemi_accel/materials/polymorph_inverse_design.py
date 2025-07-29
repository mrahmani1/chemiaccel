"""
Inverse Design for Polymorph Control in ChemiAccel.

This module extends inverse design capabilities to molecular modifications that bias
crystallization towards specific, desired polymorphs. It utilizes AI and optimization
techniques to navigate the molecular design space for solid-state properties.

Requirements: 43.3
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional imports with graceful fallbacks
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = np
    _CUPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from scipy.optimize import basinhopping, differential_evolution, minimize
    from scipy.spatial.distance import pdist, squareform

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False

# Import from our modules
from .crystal_structure_prediction import CrystalSystemType, PackingMotifType
from .intermolecular_interactions import (
    GPUAcceleratedInteractionEngine,
    InteractionConfiguration,
    InteractionType,
    MolecularFragment,
    create_molecular_fragment_from_smiles,
)


class PolymorphType(Enum):
    """Types of polymorphs."""

    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    DELTA = "delta"
    EPSILON = "epsilon"
    ANHYDROUS = "anhydrous"
    HYDRATED = "hydrated"
    SOLVATED = "solvated"
    METASTABLE = "metastable"
    STABLE = "stable"


class ModificationStrategy(Enum):
    """Molecular modification strategies."""

    FUNCTIONAL_GROUP_ADDITION = "functional_group_addition"
    FUNCTIONAL_GROUP_REMOVAL = "functional_group_removal"
    SUBSTITUENT_MODIFICATION = "substituent_modification"
    CONFORMATIONAL_CHANGE = "conformational_change"
    STEREOCHEMICAL_CHANGE = "stereochemical_change"
    RING_MODIFICATION = "ring_modification"
    LINKER_MODIFICATION = "linker_modification"
    HYDROGEN_BOND_MODULATION = "hydrogen_bond_modulation"
    PI_PI_MODULATION = "pi_pi_modulation"
    VDW_MODULATION = "vdw_modulation"


class DesignObjective(Enum):
    """Design objectives for polymorph control."""

    TARGET_SPECIFIC_POLYMORPH = "target_specific_polymorph"
    SUPPRESS_UNDESIRED_POLYMORPH = "suppress_undesired_polymorph"
    ENHANCE_STABILITY = "enhance_stability"
    IMPROVE_SOLUBILITY = "improve_solubility"
    OPTIMIZE_DENSITY = "optimize_density"
    CONTROL_MORPHOLOGY = "control_morphology"
    MODULATE_DISSOLUTION = "modulate_dissolution"
    ENHANCE_MECHANICAL_PROPERTIES = "enhance_mechanical_properties"


class OptimizationAlgorithm(Enum):
    """Optimization algorithms for inverse design."""

    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    VARIATIONAL_AUTOENCODER = "variational_autoencoder"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    MULTI_OBJECTIVE_NSGA = "multi_objective_nsga"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"


@dataclass
class PolymorphTarget:
    """Target polymorph specification."""

    polymorph_type: PolymorphType
    crystal_system: CrystalSystemType
    space_group: Optional[str] = None
    packing_motifs: List[PackingMotifType] = field(default_factory=list)
    target_density: Optional[float] = None
    target_stability: Optional[float] = None
    interaction_preferences: Dict[InteractionType, float] = field(default_factory=dict)
    geometric_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate polymorph target."""
        if self.target_density is not None and self.target_density <= 0:
            raise ValueError("Target density must be positive")
        if self.target_stability is not None and self.target_stability < 0:
            raise ValueError("Target stability must be non-negative")


@dataclass
class MolecularModification:
    """Molecular modification specification."""

    modification_id: str
    strategy: ModificationStrategy
    target_atoms: List[int]
    modification_smarts: str
    replacement_smarts: Optional[str] = None
    modification_parameters: Dict[str, Any] = field(default_factory=dict)
    predicted_effect: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def __post_init__(self):
        """Validate modification."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class DesignCandidate:
    """Design candidate with modified molecule."""

    candidate_id: str
    original_smiles: str
    modified_smiles: str
    modifications: List[MolecularModification]
    predicted_polymorph: PolymorphType
    polymorph_probability: Dict[PolymorphType, float] = field(default_factory=dict)
    stability_score: float = 0.0
    interaction_profile: Dict[InteractionType, float] = field(default_factory=dict)
    synthetic_accessibility: float = 0.0
    novelty_score: float = 0.0
    fitness_score: float = 0.0

    def get_modification_summary(self) -> Dict[str, int]:
        """Get summary of modifications applied."""
        summary = {}
        for mod in self.modifications:
            strategy = mod.strategy.value
            summary[strategy] = summary.get(strategy, 0) + 1
        return summary


@dataclass
class PolymorphDesignConfig:
    """Configuration for polymorph inverse design."""

    target_polymorph: PolymorphTarget
    design_objectives: List[DesignObjective] = field(
        default_factory=lambda: [DesignObjective.TARGET_SPECIFIC_POLYMORPH]
    )
    allowed_modifications: List[ModificationStrategy] = field(default_factory=list)
    modification_limits: Dict[ModificationStrategy, int] = field(default_factory=dict)
    synthetic_accessibility_threshold: float = 0.7
    novelty_threshold: float = 0.5
    max_heavy_atoms: int = 50
    min_heavy_atoms: int = 5
    use_gpu: bool = True
    optimization_algorithm: OptimizationAlgorithm = (
        OptimizationAlgorithm.GENETIC_ALGORITHM
    )
    population_size: int = 100
    max_generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.synthetic_accessibility_threshold <= 1:
            raise ValueError(
                "Synthetic accessibility threshold must be between 0 and 1"
            )
        if not 0 <= self.novelty_threshold <= 1:
            raise ValueError("Novelty threshold must be between 0 and 1")
        if self.max_heavy_atoms <= self.min_heavy_atoms:
            raise ValueError("Max heavy atoms must be greater than min heavy atoms")


@dataclass
class DesignResult:
    """Result of polymorph inverse design."""

    design_candidates: List[DesignCandidate]
    optimization_history: List[float] = field(default_factory=list)
    convergence_achieved: bool = False
    computation_time: float = 0.0
    success_rate: float = 0.0
    diversity_metrics: Dict[str, float] = field(default_factory=dict)

    def get_top_candidates(self, n: int = 10) -> List[DesignCandidate]:
        """Get top n candidates by fitness score."""
        sorted_candidates = sorted(
            self.design_candidates, key=lambda x: x.fitness_score, reverse=True
        )
        return sorted_candidates[:n]

    def get_candidates_by_polymorph(
        self, polymorph_type: PolymorphType
    ) -> List[DesignCandidate]:
        """Get candidates predicted to form specific polymorph."""
        return [
            c for c in self.design_candidates if c.predicted_polymorph == polymorph_type
        ]


class PolymorphPredictor:
    """AI model for predicting polymorph formation."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and _TORCH_AVAILABLE
        self.device = (
            torch.device(
                "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            )
            if _TORCH_AVAILABLE
            else None
        )

        # Initialize neural network
        if _TORCH_AVAILABLE:
            self.model = self._build_model()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.model = None
            warnings.warn(
                "PyTorch not available, using simplified polymorph prediction"
            )

    def _build_model(self) -> nn.Module:
        """Build neural network for polymorph prediction."""

        class PolymorphNet(nn.Module):
            def __init__(self):
                super(PolymorphNet, self).__init__()

                # Molecular descriptor features
                self.descriptor_encoder = nn.Sequential(
                    nn.Linear(200, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )

                # Interaction features
                self.interaction_encoder = nn.Sequential(
                    nn.Linear(50, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                )

                # Combined features
                self.combined_encoder = nn.Sequential(
                    nn.Linear(64 + 16, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                )

                # Polymorph classifier
                self.classifier = nn.Linear(16, len(PolymorphType))

                # Stability regressor
                self.stability_regressor = nn.Linear(16, 1)

            def forward(self, descriptors, interactions):
                desc_features = self.descriptor_encoder(descriptors)
                int_features = self.interaction_encoder(interactions)

                combined = torch.cat([desc_features, int_features], dim=1)
                encoded = self.combined_encoder(combined)

                polymorph_logits = self.classifier(encoded)
                stability = self.stability_regressor(encoded)

                return polymorph_logits, stability

        model = PolymorphNet()
        if self.use_gpu:
            model = model.to(self.device)

        return model

    def predict_polymorph(
        self, smiles: str, interaction_features: Optional[np.ndarray] = None
    ) -> Dict[PolymorphType, float]:
        """Predict polymorph probabilities for a molecule."""
        if not _TORCH_AVAILABLE or self.model is None:
            return self._simple_polymorph_prediction(smiles)

        # Extract molecular descriptors
        descriptors = self._extract_molecular_descriptors(smiles)
        if descriptors is None:
            return self._simple_polymorph_prediction(smiles)

        # Prepare interaction features
        if interaction_features is None:
            interaction_features = np.zeros(50)  # Default features

        # Convert to tensors
        desc_tensor = torch.FloatTensor(descriptors).unsqueeze(0)
        int_tensor = torch.FloatTensor(interaction_features).unsqueeze(0)

        if self.use_gpu:
            desc_tensor = desc_tensor.to(self.device)
            int_tensor = int_tensor.to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            polymorph_logits, stability = self.model(desc_tensor, int_tensor)

            # Convert to probabilities
            probabilities = F.softmax(polymorph_logits, dim=1).cpu().numpy()[0]

        # Map to polymorph types
        polymorph_probs = {}
        for i, polymorph_type in enumerate(PolymorphType):
            polymorph_probs[polymorph_type] = float(probabilities[i])

        return polymorph_probs

    def predict_stability(
        self, smiles: str, interaction_features: Optional[np.ndarray] = None
    ) -> float:
        """Predict crystal stability score."""
        if not _TORCH_AVAILABLE or self.model is None:
            return self._simple_stability_prediction(smiles)

        # Extract features
        descriptors = self._extract_molecular_descriptors(smiles)
        if descriptors is None:
            return 0.5

        if interaction_features is None:
            interaction_features = np.zeros(50)

        # Convert to tensors
        desc_tensor = torch.FloatTensor(descriptors).unsqueeze(0)
        int_tensor = torch.FloatTensor(interaction_features).unsqueeze(0)

        if self.use_gpu:
            desc_tensor = desc_tensor.to(self.device)
            int_tensor = int_tensor.to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            _, stability = self.model(desc_tensor, int_tensor)
            stability_score = torch.sigmoid(stability).cpu().numpy()[0, 0]

        return float(stability_score)

    def _extract_molecular_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        """Extract molecular descriptors from SMILES."""
        if not _RDKIT_AVAILABLE:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Calculate descriptors
            descriptors = []

            # Basic descriptors
            descriptors.extend(
                [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumSaturatedRings(mol),
                    Descriptors.FractionCsp3(mol),
                    Descriptors.BertzCT(mol),
                ]
            )

            # Fingerprint-based descriptors (simplified)
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            fp_array = np.array(fp)

            # Select top features (simplified)
            selected_fp = fp_array[:190]  # Take first 190 bits

            descriptors.extend(selected_fp.tolist())

            # Pad or truncate to 200 features
            if len(descriptors) < 200:
                descriptors.extend([0.0] * (200 - len(descriptors)))
            else:
                descriptors = descriptors[:200]

            return np.array(descriptors, dtype=np.float32)

        except Exception as e:
            warnings.warn(f"Error extracting descriptors: {e}")
            return None

    def _simple_polymorph_prediction(self, smiles: str) -> Dict[PolymorphType, float]:
        """Simple heuristic polymorph prediction."""
        # Very simplified prediction based on molecular features
        probs = {}
        total_prob = 0.0

        # Default uniform distribution
        for polymorph_type in PolymorphType:
            prob = np.random.random() * 0.5 + 0.1
            probs[polymorph_type] = prob
            total_prob += prob

        # Normalize
        for polymorph_type in probs:
            probs[polymorph_type] /= total_prob

        return probs

    def _simple_stability_prediction(self, smiles: str) -> float:
        """Simple stability prediction."""
        # Very simplified based on molecular weight and complexity
        if not _RDKIT_AVAILABLE:
            return 0.5

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.5

            mw = Descriptors.MolWt(mol)
            complexity = Descriptors.BertzCT(mol)

            # Simple heuristic
            stability = (
                1.0 / (1.0 + np.exp(-(mw - 300) / 100)) * np.exp(-complexity / 1000)
            )
            return max(0.1, min(0.9, stability))

        except Exception:
            return 0.5


class MolecularModifier:
    """Engine for applying molecular modifications."""

    def __init__(self):
        self.modification_templates = self._initialize_modification_templates()
        self.functional_groups = self._initialize_functional_groups()

    def _initialize_modification_templates(
        self,
    ) -> Dict[ModificationStrategy, List[str]]:
        """Initialize modification templates."""
        templates = {
            ModificationStrategy.FUNCTIONAL_GROUP_ADDITION: [
                "[CH3]",  # Methyl
                "[OH]",  # Hydroxyl
                "[NH2]",  # Amino
                "[COOH]",  # Carboxyl
                "[F]",  # Fluorine
                "[Cl]",  # Chlorine
                "[CN]",  # Cyano
            ],
            ModificationStrategy.FUNCTIONAL_GROUP_REMOVAL: [
                "[OH]",
                "[NH2]",
                "[CH3]",
                "[F]",
                "[Cl]",
            ],
            ModificationStrategy.SUBSTITUENT_MODIFICATION: [
                ("[CH3]", "[CF3]"),  # Methyl to trifluoromethyl
                ("[OH]", "[NH2]"),  # Hydroxyl to amino
                ("[H]", "[CH3]"),  # Hydrogen to methyl
                ("[F]", "[Cl]"),  # Fluorine to chlorine
                ("[NH2]", "[NHCH3]"),  # Amino to methylamino
            ],
        }
        return templates

    def _initialize_functional_groups(self) -> Dict[str, str]:
        """Initialize functional group SMARTS patterns."""
        return {
            "hydroxyl": "[OH]",
            "amino": "[NH2]",
            "carboxyl": "[CX3](=O)[OX2H1]",
            "ketone": "[CX3]=[OX1]",
            "ester": "[CX3](=O)[OX2H0]",
            "ether": "[OD2]([#6])[#6]",
            "aromatic": "c",
            "halogen": "[F,Cl,Br,I]",
            "nitro": "[N+](=O)[O-]",
            "cyano": "C#N",
        }

    def apply_modification(
        self, smiles: str, modification: MolecularModification
    ) -> Optional[str]:
        """Apply a molecular modification to a SMILES string."""
        if not _RDKIT_AVAILABLE:
            warnings.warn("RDKit not available, cannot apply modifications")
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            if modification.strategy == ModificationStrategy.FUNCTIONAL_GROUP_ADDITION:
                return self._add_functional_group(mol, modification)
            elif modification.strategy == ModificationStrategy.FUNCTIONAL_GROUP_REMOVAL:
                return self._remove_functional_group(mol, modification)
            elif modification.strategy == ModificationStrategy.SUBSTITUENT_MODIFICATION:
                return self._modify_substituent(mol, modification)
            else:
                return self._apply_generic_modification(mol, modification)

        except Exception as e:
            warnings.warn(f"Error applying modification: {e}")
            return None

    def _add_functional_group(
        self, mol: "Chem.Mol", modification: MolecularModification
    ) -> Optional[str]:
        """Add functional group to molecule."""
        # Simplified implementation
        try:
            # Find available attachment points
            attachment_points = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "C" and atom.GetTotalDegree() < 4:
                    attachment_points.append(atom.GetIdx())

            if not attachment_points:
                return None

            # Simple approach: add to random available position
            attach_idx = np.random.choice(attachment_points)

            # Create modified molecule (simplified)
            smiles = Chem.MolToSmiles(mol)

            # Add group (very simplified)
            if "[OH]" in modification.modification_smarts:
                modified_smiles = smiles + ".O"  # Simplified addition
            elif "[CH3]" in modification.modification_smarts:
                modified_smiles = smiles + ".C"  # Simplified addition
            else:
                modified_smiles = smiles

            # Validate
            test_mol = Chem.MolFromSmiles(modified_smiles)
            if test_mol is not None:
                return Chem.MolToSmiles(test_mol)
            else:
                return smiles  # Return original if modification fails

        except Exception:
            return None

    def _remove_functional_group(
        self, mol: "Chem.Mol", modification: MolecularModification
    ) -> Optional[str]:
        """Remove functional group from molecule."""
        try:
            # Find groups matching pattern
            pattern = Chem.MolFromSmarts(modification.modification_smarts)
            if pattern is None:
                return None

            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return Chem.MolToSmiles(mol)  # No matches, return original

            # Remove first match (simplified)
            match = matches[0]

            # Create editable molecule
            em = Chem.EditableMol(mol)

            # Remove atoms in reverse order to maintain indices
            for atom_idx in sorted(match, reverse=True):
                em.RemoveAtom(atom_idx)

            modified_mol = em.GetMol()
            if modified_mol is not None:
                Chem.SanitizeMol(modified_mol)
                return Chem.MolToSmiles(modified_mol)
            else:
                return None

        except Exception:
            return None

    def _modify_substituent(
        self, mol: "Chem.Mol", modification: MolecularModification
    ) -> Optional[str]:
        """Modify existing substituent."""
        try:
            # Find and replace pattern
            if modification.replacement_smarts:
                # Use reaction SMARTS for transformation
                reaction_smarts = f"{modification.modification_smarts}>>{modification.replacement_smarts}"
                reaction = AllChem.ReactionFromSmarts(reaction_smarts)

                if reaction is not None:
                    products = reaction.RunReactants((mol,))
                    if products:
                        product_mol = products[0][0]
                        Chem.SanitizeMol(product_mol)
                        return Chem.MolToSmiles(product_mol)

            return Chem.MolToSmiles(mol)  # Return original if modification fails

        except Exception:
            return None

    def _apply_generic_modification(
        self, mol: "Chem.Mol", modification: MolecularModification
    ) -> Optional[str]:
        """Apply generic modification."""
        # Fallback for other modification types
        return Chem.MolToSmiles(mol)

    def generate_random_modification(
        self, smiles: str, allowed_strategies: List[ModificationStrategy]
    ) -> Optional[MolecularModification]:
        """Generate random modification for a molecule."""
        if not allowed_strategies:
            return None

        strategy = np.random.choice(allowed_strategies)

        if strategy in self.modification_templates:
            templates = self.modification_templates[strategy]
            template = np.random.choice(templates)

            if (
                strategy == ModificationStrategy.SUBSTITUENT_MODIFICATION
                and isinstance(template, tuple)
            ):
                modification_smarts, replacement_smarts = template
            else:
                modification_smarts = template
                replacement_smarts = None

            return MolecularModification(
                modification_id=f"mod_{np.random.randint(10000)}",
                strategy=strategy,
                target_atoms=[],  # Will be determined during application
                modification_smarts=modification_smarts,
                replacement_smarts=replacement_smarts,
                confidence=np.random.random(),
            )

        return None


class PolymorphInverseDesigner:
    """Main inverse design engine for polymorph control."""

    def __init__(self, config: PolymorphDesignConfig):
        self.config = config
        self.use_gpu = config.use_gpu and _CUPY_AVAILABLE

        # Initialize components
        self.polymorph_predictor = PolymorphPredictor(config.use_gpu)
        self.molecular_modifier = MolecularModifier()
        self.interaction_engine = GPUAcceleratedInteractionEngine(
            InteractionConfiguration(use_gpu=config.use_gpu)
        )

        # Design history
        self.design_history = []
        self.evaluation_cache = {}

        # Performance metrics
        self.performance_stats = {
            "total_evaluations": 0,
            "successful_modifications": 0,
            "cache_hits": 0,
        }

    def design_molecules_for_polymorph(
        self, starting_smiles: str, max_modifications: int = 3
    ) -> DesignResult:
        """Design molecules targeting specific polymorph."""
        if (
            self.config.optimization_algorithm
            == OptimizationAlgorithm.GENETIC_ALGORITHM
        ):
            return self._genetic_algorithm_design(starting_smiles, max_modifications)
        elif (
            self.config.optimization_algorithm
            == OptimizationAlgorithm.SIMULATED_ANNEALING
        ):
            return self._simulated_annealing_design(starting_smiles, max_modifications)
        elif (
            self.config.optimization_algorithm
            == OptimizationAlgorithm.REINFORCEMENT_LEARNING
            and _TORCH_AVAILABLE
        ):
            return self._reinforcement_learning_design(
                starting_smiles, max_modifications
            )
        else:
            return self._simple_random_design(starting_smiles, max_modifications)

    def _genetic_algorithm_design(
        self, starting_smiles: str, max_modifications: int
    ) -> DesignResult:
        """Genetic algorithm for polymorph-targeted design."""
        import time

        start_time = time.time()

        # Initialize population
        population = self._initialize_population(starting_smiles, max_modifications)
        optimization_history = []

        best_fitness = 0.0
        best_candidate = None

        for generation in range(self.config.max_generations):
            # Evaluate fitness
            fitness_scores = []
            for candidate in population:
                fitness = self._evaluate_fitness(candidate)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_candidate = candidate

            optimization_history.append(best_fitness)

            # Selection, crossover, and mutation
            new_population = []

            # Elite selection
            elite_size = int(0.1 * self.config.population_size)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])

            # Generate offspring
            while len(new_population) < self.config.population_size:
                parent1, parent2 = self._tournament_selection(
                    population, fitness_scores
                )

                if np.random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                if np.random.random() < self.config.mutation_rate:
                    child1 = self._mutate(child1, max_modifications)
                if np.random.random() < self.config.mutation_rate:
                    child2 = self._mutate(child2, max_modifications)

                new_population.extend([child1, child2])

            population = new_population[: self.config.population_size]

        computation_time = time.time() - start_time

        # Calculate success rate
        successful_candidates = [c for c in population if c.fitness_score > 0.7]
        success_rate = len(successful_candidates) / len(population)

        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(population)

        return DesignResult(
            design_candidates=population,
            optimization_history=optimization_history,
            convergence_achieved=len(optimization_history) >= 10,
            computation_time=computation_time,
            success_rate=success_rate,
            diversity_metrics=diversity_metrics,
        )

    def _simulated_annealing_design(
        self, starting_smiles: str, max_modifications: int
    ) -> DesignResult:
        """Simulated annealing for polymorph-targeted design."""
        import time

        start_time = time.time()

        # Initialize with starting molecule
        current_candidate = self._create_candidate_from_smiles(starting_smiles)
        current_fitness = self._evaluate_fitness(current_candidate)

        best_candidate = current_candidate
        best_fitness = current_fitness

        all_candidates = [current_candidate]
        optimization_history = [current_fitness]

        temperature = 100.0
        cooling_rate = 0.95
        min_temperature = 1.0

        while temperature > min_temperature:
            # Generate neighbor
            neighbor = self._generate_neighbor(current_candidate, max_modifications)
            neighbor_fitness = self._evaluate_fitness(neighbor)

            # Accept or reject
            delta_fitness = neighbor_fitness - current_fitness

            if delta_fitness > 0 or np.random.random() < np.exp(
                delta_fitness / temperature
            ):
                current_candidate = neighbor
                current_fitness = neighbor_fitness
                all_candidates.append(current_candidate)

                if current_fitness > best_fitness:
                    best_candidate = current_candidate
                    best_fitness = current_fitness

            optimization_history.append(best_fitness)
            temperature *= cooling_rate

        computation_time = time.time() - start_time

        return DesignResult(
            design_candidates=all_candidates[-50:],  # Keep last 50 candidates
            optimization_history=optimization_history,
            convergence_achieved=True,
            computation_time=computation_time,
            success_rate=1.0 if best_fitness > 0.7 else 0.0,
            diversity_metrics={},
        )

    def _reinforcement_learning_design(
        self, starting_smiles: str, max_modifications: int
    ) -> DesignResult:
        """Reinforcement learning for polymorph-targeted design."""
        # Placeholder for RL implementation
        warnings.warn(
            "Reinforcement learning design not fully implemented, using random design"
        )
        return self._simple_random_design(starting_smiles, max_modifications)

    def _simple_random_design(
        self, starting_smiles: str, max_modifications: int
    ) -> DesignResult:
        """Simple random design for fallback."""
        import time

        start_time = time.time()

        candidates = []

        for _ in range(self.config.population_size):
            candidate = self._create_candidate_from_smiles(starting_smiles)

            # Apply random modifications
            n_modifications = np.random.randint(1, max_modifications + 1)
            current_smiles = starting_smiles

            for _ in range(n_modifications):
                modification = self.molecular_modifier.generate_random_modification(
                    current_smiles, self.config.allowed_modifications
                )

                if modification:
                    modified_smiles = self.molecular_modifier.apply_modification(
                        current_smiles, modification
                    )
                    if modified_smiles:
                        current_smiles = modified_smiles
                        candidate.modifications.append(modification)

            candidate.modified_smiles = current_smiles
            candidate.fitness_score = self._evaluate_fitness(candidate)
            candidates.append(candidate)

        computation_time = time.time() - start_time

        return DesignResult(
            design_candidates=candidates,
            optimization_history=[],
            convergence_achieved=False,
            computation_time=computation_time,
            success_rate=0.0,
            diversity_metrics={},
        )

    def _initialize_population(
        self, starting_smiles: str, max_modifications: int
    ) -> List[DesignCandidate]:
        """Initialize population for genetic algorithm."""
        population = []

        for i in range(self.config.population_size):
            candidate = self._create_candidate_from_smiles(
                starting_smiles, f"candidate_{i}"
            )

            # Apply random modifications
            n_modifications = np.random.randint(0, max_modifications + 1)
            current_smiles = starting_smiles

            for j in range(n_modifications):
                if self.config.allowed_modifications:
                    modification = self.molecular_modifier.generate_random_modification(
                        current_smiles, self.config.allowed_modifications
                    )

                    if modification:
                        modified_smiles = self.molecular_modifier.apply_modification(
                            current_smiles, modification
                        )
                        if modified_smiles and self._is_valid_molecule(modified_smiles):
                            current_smiles = modified_smiles
                            candidate.modifications.append(modification)

            candidate.modified_smiles = current_smiles
            population.append(candidate)

        return population

    def _create_candidate_from_smiles(
        self, smiles: str, candidate_id: str = None
    ) -> DesignCandidate:
        """Create design candidate from SMILES."""
        if candidate_id is None:
            candidate_id = f"candidate_{np.random.randint(10000)}"

        return DesignCandidate(
            candidate_id=candidate_id,
            original_smiles=smiles,
            modified_smiles=smiles,
            modifications=[],
            predicted_polymorph=PolymorphType.ALPHA,  # Will be updated
            polymorph_probability={},
            stability_score=0.0,
            interaction_profile={},
            synthetic_accessibility=0.5,
            novelty_score=0.5,
            fitness_score=0.0,
        )

    def _evaluate_fitness(self, candidate: DesignCandidate) -> float:
        """Evaluate fitness of design candidate."""
        cache_key = candidate.modified_smiles

        if cache_key in self.evaluation_cache:
            self.performance_stats["cache_hits"] += 1
            cached_result = self.evaluation_cache[cache_key]
            candidate.predicted_polymorph = cached_result["polymorph"]
            candidate.polymorph_probability = cached_result["polymorph_prob"]
            candidate.stability_score = cached_result["stability"]
            candidate.fitness_score = cached_result["fitness"]
            return cached_result["fitness"]

        self.performance_stats["total_evaluations"] += 1

        # Predict polymorph
        polymorph_probs = self.polymorph_predictor.predict_polymorph(
            candidate.modified_smiles
        )
        candidate.polymorph_probability = polymorph_probs

        # Get most likely polymorph
        candidate.predicted_polymorph = max(polymorph_probs, key=polymorph_probs.get)

        # Predict stability
        candidate.stability_score = self.polymorph_predictor.predict_stability(
            candidate.modified_smiles
        )

        # Calculate interaction profile
        candidate.interaction_profile = self._calculate_interaction_profile(
            candidate.modified_smiles
        )

        # Calculate synthetic accessibility and novelty
        candidate.synthetic_accessibility = self._calculate_synthetic_accessibility(
            candidate.modified_smiles
        )
        candidate.novelty_score = self._calculate_novelty_score(
            candidate.modified_smiles, candidate.original_smiles
        )

        # Calculate overall fitness
        fitness = self._calculate_overall_fitness(candidate)
        candidate.fitness_score = fitness

        # Cache result
        self.evaluation_cache[cache_key] = {
            "polymorph": candidate.predicted_polymorph,
            "polymorph_prob": candidate.polymorph_probability,
            "stability": candidate.stability_score,
            "fitness": fitness,
        }

        return fitness

    def _calculate_interaction_profile(
        self, smiles: str
    ) -> Dict[InteractionType, float]:
        """Calculate interaction profile for molecule."""
        try:
            fragment = create_molecular_fragment_from_smiles(smiles)
            if fragment is None:
                return {}

            # Calculate interactions with itself (crystal packing)
            fragments = [fragment, fragment]  # Simplified
            interactions = self.interaction_engine.calculate_all_interactions(fragments)

            profile = {}
            for pair in interactions:
                for interaction_type, energy in pair.interaction_energies.items():
                    profile[interaction_type] = profile.get(interaction_type, 0) + abs(
                        energy
                    )

            return profile

        except Exception:
            return {}

    def _calculate_synthetic_accessibility(self, smiles: str) -> float:
        """Calculate synthetic accessibility score."""
        if not _RDKIT_AVAILABLE:
            return 0.5

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            # Simple heuristic based on molecular complexity
            complexity = Descriptors.BertzCT(mol)
            n_rings = Descriptors.RingCount(mol)
            n_heteroatoms = Descriptors.NumHeteroatoms(mol)

            # Simplified SA score
            sa_score = 1.0 / (
                1.0 + complexity / 100.0 + n_rings * 0.1 + n_heteroatoms * 0.05
            )
            return max(0.1, min(0.9, sa_score))

        except Exception:
            return 0.5

    def _calculate_novelty_score(
        self, modified_smiles: str, original_smiles: str
    ) -> float:
        """Calculate novelty score compared to original molecule."""
        if not _RDKIT_AVAILABLE:
            return 0.5

        try:
            mol1 = Chem.MolFromSmiles(original_smiles)
            mol2 = Chem.MolFromSmiles(modified_smiles)

            if mol1 is None or mol2 is None:
                return 0.5

            # Calculate Tanimoto similarity
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)

            similarity = Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
            novelty = 1.0 - similarity

            return max(0.0, min(1.0, novelty))

        except Exception:
            return 0.5

    def _calculate_overall_fitness(self, candidate: DesignCandidate) -> float:
        """Calculate overall fitness score."""
        # Target polymorph match
        target_prob = candidate.polymorph_probability.get(
            self.config.target_polymorph.polymorph_type, 0.0
        )

        # Stability score
        stability_score = candidate.stability_score

        # Interaction profile match
        interaction_score = self._calculate_interaction_score(
            candidate.interaction_profile
        )

        # Synthetic accessibility
        sa_score = candidate.synthetic_accessibility

        # Novelty score
        novelty_score = candidate.novelty_score

        # Apply constraints
        if sa_score < self.config.synthetic_accessibility_threshold:
            return 0.0

        if novelty_score < self.config.novelty_threshold:
            return 0.0

        # Weighted combination
        fitness = (
            target_prob * 0.4
            + stability_score * 0.25
            + interaction_score * 0.2
            + sa_score * 0.1
            + novelty_score * 0.05
        )

        return max(0.0, min(1.0, fitness))

    def _calculate_interaction_score(
        self, interaction_profile: Dict[InteractionType, float]
    ) -> float:
        """Calculate interaction score based on target preferences."""
        if not self.config.target_polymorph.interaction_preferences:
            return 0.5

        score = 0.0
        total_weight = 0.0

        for (
            interaction_type,
            target_strength,
        ) in self.config.target_polymorph.interaction_preferences.items():
            actual_strength = interaction_profile.get(interaction_type, 0.0)

            # Normalize strengths
            normalized_target = min(1.0, target_strength / 10.0)
            normalized_actual = min(1.0, actual_strength / 10.0)

            # Calculate similarity
            similarity = 1.0 - abs(normalized_target - normalized_actual)
            score += similarity * normalized_target
            total_weight += normalized_target

        if total_weight > 0:
            return score / total_weight
        else:
            return 0.5

    def _is_valid_molecule(self, smiles: str) -> bool:
        """Check if SMILES represents a valid molecule."""
        if not _RDKIT_AVAILABLE:
            return True  # Assume valid if can't check

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Check atom count constraints
            n_heavy_atoms = mol.GetNumHeavyAtoms()
            if not (
                self.config.min_heavy_atoms
                <= n_heavy_atoms
                <= self.config.max_heavy_atoms
            ):
                return False

            return True

        except Exception:
            return False

    def _tournament_selection(
        self,
        population: List[DesignCandidate],
        fitness_scores: List[float],
        tournament_size: int = 3,
    ) -> Tuple[DesignCandidate, DesignCandidate]:
        """Tournament selection for genetic algorithm."""

        def select_one():
            tournament_indices = np.random.choice(
                len(population), tournament_size, replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            return population[winner_idx]

        return select_one(), select_one()

    def _crossover(
        self, parent1: DesignCandidate, parent2: DesignCandidate
    ) -> Tuple[DesignCandidate, DesignCandidate]:
        """Crossover operation for genetic algorithm."""
        # Create children
        child1 = self._create_candidate_from_smiles(parent1.original_smiles)
        child2 = self._create_candidate_from_smiles(parent2.original_smiles)

        # Combine modifications
        all_modifications = parent1.modifications + parent2.modifications

        if all_modifications:
            # Split modifications randomly
            split_point = len(all_modifications) // 2
            np.random.shuffle(all_modifications)

            child1.modifications = all_modifications[:split_point]
            child2.modifications = all_modifications[split_point:]

            # Apply modifications
            child1.modified_smiles = self._apply_modifications(
                child1.original_smiles, child1.modifications
            )
            child2.modified_smiles = self._apply_modifications(
                child2.original_smiles, child2.modifications
            )

        return child1, child2

    def _mutate(
        self, candidate: DesignCandidate, max_modifications: int
    ) -> DesignCandidate:
        """Mutation operation for genetic algorithm."""
        mutated = self._create_candidate_from_smiles(candidate.original_smiles)
        mutated.modifications = candidate.modifications.copy()

        # Random modification operations
        if np.random.random() < 0.5 and len(mutated.modifications) < max_modifications:
            # Add new modification
            new_modification = self.molecular_modifier.generate_random_modification(
                mutated.modified_smiles, self.config.allowed_modifications
            )
            if new_modification:
                mutated.modifications.append(new_modification)

        if np.random.random() < 0.3 and mutated.modifications:
            # Remove random modification
            idx = np.random.randint(len(mutated.modifications))
            mutated.modifications.pop(idx)

        if np.random.random() < 0.2 and mutated.modifications:
            # Replace random modification
            idx = np.random.randint(len(mutated.modifications))
            new_modification = self.molecular_modifier.generate_random_modification(
                mutated.modified_smiles, self.config.allowed_modifications
            )
            if new_modification:
                mutated.modifications[idx] = new_modification

        # Apply all modifications
        mutated.modified_smiles = self._apply_modifications(
            mutated.original_smiles, mutated.modifications
        )

        return mutated

    def _generate_neighbor(
        self, candidate: DesignCandidate, max_modifications: int
    ) -> DesignCandidate:
        """Generate neighbor for simulated annealing."""
        return self._mutate(candidate, max_modifications)

    def _apply_modifications(
        self, smiles: str, modifications: List[MolecularModification]
    ) -> str:
        """Apply list of modifications to molecule."""
        current_smiles = smiles

        for modification in modifications:
            modified_smiles = self.molecular_modifier.apply_modification(
                current_smiles, modification
            )
            if modified_smiles and self._is_valid_molecule(modified_smiles):
                current_smiles = modified_smiles
                self.performance_stats["successful_modifications"] += 1

        return current_smiles

    def _calculate_diversity_metrics(
        self, population: List[DesignCandidate]
    ) -> Dict[str, float]:
        """Calculate diversity metrics for population."""
        if not _RDKIT_AVAILABLE or len(population) < 2:
            return {}

        try:
            # Calculate pairwise similarities
            similarities = []

            for i, candidate1 in enumerate(population):
                mol1 = Chem.MolFromSmiles(candidate1.modified_smiles)
                if mol1 is None:
                    continue

                fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)

                for j, candidate2 in enumerate(population[i + 1 :], i + 1):
                    mol2 = Chem.MolFromSmiles(candidate2.modified_smiles)
                    if mol2 is None:
                        continue

                    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)
                    similarity = Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
                    similarities.append(similarity)

            if similarities:
                return {
                    "mean_similarity": np.mean(similarities),
                    "std_similarity": np.std(similarities),
                    "min_similarity": np.min(similarities),
                    "max_similarity": np.max(similarities),
                    "diversity_score": 1.0 - np.mean(similarities),
                }
            else:
                return {}

        except Exception:
            return {}


# Utility functions


def create_polymorph_target(
    polymorph_type: PolymorphType,
    crystal_system: CrystalSystemType,
    target_density: Optional[float] = None,
    interaction_preferences: Optional[Dict[InteractionType, float]] = None,
) -> PolymorphTarget:
    """Create polymorph target specification."""
    return PolymorphTarget(
        polymorph_type=polymorph_type,
        crystal_system=crystal_system,
        target_density=target_density,
        interaction_preferences=interaction_preferences or {},
    )


def create_design_config(
    target_polymorph: PolymorphTarget,
    allowed_modifications: Optional[List[ModificationStrategy]] = None,
    use_gpu: bool = True,
    population_size: int = 50,
    max_generations: int = 30,
) -> PolymorphDesignConfig:
    """Create design configuration with common settings."""
    if allowed_modifications is None:
        allowed_modifications = [
            ModificationStrategy.FUNCTIONAL_GROUP_ADDITION,
            ModificationStrategy.FUNCTIONAL_GROUP_REMOVAL,
            ModificationStrategy.SUBSTITUENT_MODIFICATION,
        ]

    return PolymorphDesignConfig(
        target_polymorph=target_polymorph,
        allowed_modifications=allowed_modifications,
        use_gpu=use_gpu,
        population_size=population_size,
        max_generations=max_generations,
    )


def design_for_polymorph(
    starting_smiles: str,
    target_polymorph: PolymorphTarget,
    max_modifications: int = 3,
    config: Optional[PolymorphDesignConfig] = None,
) -> DesignResult:
    """Design molecules targeting specific polymorph."""
    if config is None:
        config = create_design_config(target_polymorph)
    else:
        config.target_polymorph = target_polymorph

    designer = PolymorphInverseDesigner(config)
    return designer.design_molecules_for_polymorph(starting_smiles, max_modifications)


def analyze_polymorph_landscape(
    smiles_list: List[str], use_gpu: bool = True
) -> Dict[str, Any]:
    """Analyze polymorph landscape for a set of molecules."""
    predictor = PolymorphPredictor(use_gpu)

    results = {}
    polymorph_distribution = {}
    stability_scores = []

    for smiles in smiles_list:
        polymorph_probs = predictor.predict_polymorph(smiles)
        stability = predictor.predict_stability(smiles)

        results[smiles] = {
            "polymorph_probabilities": polymorph_probs,
            "stability_score": stability,
            "most_likely_polymorph": max(polymorph_probs, key=polymorph_probs.get),
        }

        # Update distribution
        most_likely = results[smiles]["most_likely_polymorph"]
        polymorph_distribution[most_likely.value] = (
            polymorph_distribution.get(most_likely.value, 0) + 1
        )

        stability_scores.append(stability)

    analysis = {
        "individual_results": results,
        "polymorph_distribution": polymorph_distribution,
        "stability_statistics": {
            "mean": np.mean(stability_scores),
            "std": np.std(stability_scores),
            "min": np.min(stability_scores),
            "max": np.max(stability_scores),
        },
        "total_molecules": len(smiles_list),
    }

    return analysis


def predict_polymorph_stability(
    smiles: str, target_polymorph: PolymorphType, use_gpu: bool = True
) -> Dict[str, float]:
    """Predict stability for specific polymorph."""
    predictor = PolymorphPredictor(use_gpu)

    polymorph_probs = predictor.predict_polymorph(smiles)
    overall_stability = predictor.predict_stability(smiles)

    # Calculate polymorph-specific stability
    target_prob = polymorph_probs.get(target_polymorph, 0.0)
    polymorph_stability = overall_stability * target_prob

    return {
        "overall_stability": overall_stability,
        "polymorph_probability": target_prob,
        "polymorph_stability": polymorph_stability,
        "relative_stability": target_prob / max(polymorph_probs.values())
        if polymorph_probs
        else 0.0,
    }


# Exception classes


class PolymorphInverseDesignError(Exception):
    """Base exception for polymorph inverse design."""

    pass


class PolymorphPredictionError(PolymorphInverseDesignError):
    """Exception for polymorph prediction errors."""

    pass


class MolecularModificationError(PolymorphInverseDesignError):
    """Exception for molecular modification errors."""

    pass


class DesignOptimizationError(PolymorphInverseDesignError):
    """Exception for design optimization errors."""

    pass


class ConfigurationError(PolymorphInverseDesignError):
    """Exception for configuration errors."""

    pass

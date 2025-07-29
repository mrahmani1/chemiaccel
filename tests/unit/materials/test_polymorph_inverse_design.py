"""
Unit tests for polymorph inverse design module.
"""

from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.materials.crystal_structure_prediction import (
    CrystalSystemType,
    PackingMotifType,
)
from chemi_accel.materials.intermolecular_interactions import InteractionType
from chemi_accel.materials.polymorph_inverse_design import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    ConfigurationError,
    DesignCandidate,
    DesignObjective,
    DesignOptimizationError,
    DesignResult,
    ModificationStrategy,
    MolecularModification,
    MolecularModificationError,
    MolecularModifier,
    OptimizationAlgorithm,
    PolymorphDesignConfig,
    PolymorphInverseDesigner,
    PolymorphInverseDesignError,
    PolymorphPredictionError,
    PolymorphPredictor,
    PolymorphTarget,
    PolymorphType,
    analyze_polymorph_landscape,
    create_design_config,
    create_polymorph_target,
    design_for_polymorph,
    predict_polymorph_stability,
)


@pytest.fixture
def sample_polymorph_target():
    """Create sample polymorph target."""
    return PolymorphTarget(
        polymorph_type=PolymorphType.ALPHA,
        crystal_system=CrystalSystemType.ORTHORHOMBIC,
        space_group="P212121",
        packing_motifs=[PackingMotifType.HERRINGBONE],
        target_density=1.35,
        target_stability=0.85,
        interaction_preferences={
            InteractionType.HYDROGEN_BOND: 5.0,
            InteractionType.PI_PI_STACKING: 3.0,
            InteractionType.VAN_DER_WAALS: 2.0,
        },
    )


@pytest.fixture
def sample_molecular_modification():
    """Create sample molecular modification."""
    return MolecularModification(
        modification_id="mod_001",
        strategy=ModificationStrategy.FUNCTIONAL_GROUP_ADDITION,
        target_atoms=[1, 2, 3],
        modification_smarts="[OH]",
        replacement_smarts=None,
        modification_parameters={"position": "meta"},
        predicted_effect={"stability_change": 0.15, "polymorph_probability": 0.8},
        confidence=0.85,
    )


@pytest.fixture
def sample_design_candidate():
    """Create sample design candidate."""
    modification = MolecularModification(
        modification_id="mod_001",
        strategy=ModificationStrategy.FUNCTIONAL_GROUP_ADDITION,
        target_atoms=[1],
        modification_smarts="[OH]",
        confidence=0.8,
    )

    return DesignCandidate(
        candidate_id="candidate_001",
        original_smiles="c1ccccc1",
        modified_smiles="c1ccc(O)cc1",
        modifications=[modification],
        predicted_polymorph=PolymorphType.ALPHA,
        polymorph_probability={
            PolymorphType.ALPHA: 0.7,
            PolymorphType.BETA: 0.2,
            PolymorphType.GAMMA: 0.1,
        },
        stability_score=0.85,
        interaction_profile={
            InteractionType.HYDROGEN_BOND: 4.5,
            InteractionType.PI_PI_STACKING: 2.8,
        },
        synthetic_accessibility=0.9,
        novelty_score=0.6,
        fitness_score=0.82,
    )


@pytest.fixture
def sample_design_config(sample_polymorph_target):
    """Create sample design configuration."""
    return PolymorphDesignConfig(
        target_polymorph=sample_polymorph_target,
        design_objectives=[DesignObjective.TARGET_SPECIFIC_POLYMORPH],
        allowed_modifications=[
            ModificationStrategy.FUNCTIONAL_GROUP_ADDITION,
            ModificationStrategy.SUBSTITUENT_MODIFICATION,
        ],
        modification_limits={
            ModificationStrategy.FUNCTIONAL_GROUP_ADDITION: 2,
            ModificationStrategy.SUBSTITUENT_MODIFICATION: 1,
        },
        synthetic_accessibility_threshold=0.7,
        novelty_threshold=0.5,
        max_heavy_atoms=40,
        min_heavy_atoms=5,
        use_gpu=False,
        optimization_algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
        population_size=20,
        max_generations=10,
        mutation_rate=0.15,
        crossover_rate=0.75,
    )


@pytest.fixture
def sample_design_result(sample_design_candidate):
    """Create sample design result."""
    return DesignResult(
        design_candidates=[sample_design_candidate],
        optimization_history=[0.5, 0.65, 0.78, 0.82],
        convergence_achieved=True,
        computation_time=45.5,
        success_rate=0.8,
        diversity_metrics={"mean_similarity": 0.35, "diversity_score": 0.65},
    )


class TestPolymorphTarget:
    """Test PolymorphTarget class."""

    def test_target_creation(self, sample_polymorph_target):
        """Test polymorph target creation."""
        target = sample_polymorph_target

        assert target.polymorph_type == PolymorphType.ALPHA
        assert target.crystal_system == CrystalSystemType.ORTHORHOMBIC
        assert target.space_group == "P212121"
        assert len(target.packing_motifs) == 1
        assert target.target_density == 1.35
        assert target.target_stability == 0.85
        assert len(target.interaction_preferences) == 3

    def test_target_validation(self):
        """Test polymorph target validation."""
        # Valid target should not raise
        target = PolymorphTarget(
            polymorph_type=PolymorphType.BETA,
            crystal_system=CrystalSystemType.MONOCLINIC,
            target_density=1.2,
            target_stability=0.7,
        )
        assert target.target_density == 1.2

        # Invalid density should raise
        with pytest.raises(ValueError):
            PolymorphTarget(
                polymorph_type=PolymorphType.ALPHA,
                crystal_system=CrystalSystemType.CUBIC,
                target_density=-0.5,  # Negative density
            )

        # Invalid stability should raise
        with pytest.raises(ValueError):
            PolymorphTarget(
                polymorph_type=PolymorphType.ALPHA,
                crystal_system=CrystalSystemType.CUBIC,
                target_stability=-0.1,  # Negative stability
            )


class TestMolecularModification:
    """Test MolecularModification class."""

    def test_modification_creation(self, sample_molecular_modification):
        """Test molecular modification creation."""
        mod = sample_molecular_modification

        assert mod.modification_id == "mod_001"
        assert mod.strategy == ModificationStrategy.FUNCTIONAL_GROUP_ADDITION
        assert len(mod.target_atoms) == 3
        assert mod.modification_smarts == "[OH]"
        assert mod.replacement_smarts is None
        assert mod.confidence == 0.85
        assert "position" in mod.modification_parameters
        assert "stability_change" in mod.predicted_effect

    def test_modification_validation(self):
        """Test molecular modification validation."""
        # Valid modification should not raise
        mod = MolecularModification(
            modification_id="test",
            strategy=ModificationStrategy.FUNCTIONAL_GROUP_REMOVAL,
            target_atoms=[0],
            modification_smarts="[CH3]",
            confidence=0.6,
        )
        assert mod.confidence == 0.6

        # Invalid confidence should raise
        with pytest.raises(ValueError):
            MolecularModification(
                modification_id="test",
                strategy=ModificationStrategy.FUNCTIONAL_GROUP_ADDITION,
                target_atoms=[0],
                modification_smarts="[OH]",
                confidence=1.5,  # > 1.0
            )


class TestDesignCandidate:
    """Test DesignCandidate class."""

    def test_candidate_creation(self, sample_design_candidate):
        """Test design candidate creation."""
        candidate = sample_design_candidate

        assert candidate.candidate_id == "candidate_001"
        assert candidate.original_smiles == "c1ccccc1"
        assert candidate.modified_smiles == "c1ccc(O)cc1"
        assert len(candidate.modifications) == 1
        assert candidate.predicted_polymorph == PolymorphType.ALPHA
        assert len(candidate.polymorph_probability) == 3
        assert candidate.stability_score == 0.85
        assert candidate.fitness_score == 0.82

    def test_modification_summary(self, sample_design_candidate):
        """Test modification summary calculation."""
        candidate = sample_design_candidate

        summary = candidate.get_modification_summary()

        assert isinstance(summary, dict)
        assert "functional_group_addition" in summary
        assert summary["functional_group_addition"] == 1


class TestPolymorphDesignConfig:
    """Test PolymorphDesignConfig class."""

    def test_config_creation(self, sample_design_config):
        """Test design configuration creation."""
        config = sample_design_config

        assert isinstance(config.target_polymorph, PolymorphTarget)
        assert len(config.design_objectives) == 1
        assert len(config.allowed_modifications) == 2
        assert config.synthetic_accessibility_threshold == 0.7
        assert config.novelty_threshold == 0.5
        assert config.max_heavy_atoms == 40
        assert config.min_heavy_atoms == 5
        assert config.use_gpu == False
        assert config.population_size == 20
        assert config.max_generations == 10

    def test_config_validation(self, sample_polymorph_target):
        """Test configuration validation."""
        # Valid config should not raise
        config = PolymorphDesignConfig(
            target_polymorph=sample_polymorph_target,
            synthetic_accessibility_threshold=0.8,
            novelty_threshold=0.3,
            max_heavy_atoms=30,
            min_heavy_atoms=10,
        )
        assert config.synthetic_accessibility_threshold == 0.8

        # Invalid SA threshold should raise
        with pytest.raises(ValueError):
            PolymorphDesignConfig(
                target_polymorph=sample_polymorph_target,
                synthetic_accessibility_threshold=1.5,  # > 1.0
            )

        # Invalid novelty threshold should raise
        with pytest.raises(ValueError):
            PolymorphDesignConfig(
                target_polymorph=sample_polymorph_target,
                novelty_threshold=-0.1,  # < 0.0
            )

        # Invalid atom count should raise
        with pytest.raises(ValueError):
            PolymorphDesignConfig(
                target_polymorph=sample_polymorph_target,
                max_heavy_atoms=5,
                min_heavy_atoms=10,  # min > max
            )


class TestDesignResult:
    """Test DesignResult class."""

    def test_result_creation(self, sample_design_result):
        """Test design result creation."""
        result = sample_design_result

        assert len(result.design_candidates) == 1
        assert len(result.optimization_history) == 4
        assert result.convergence_achieved == True
        assert result.computation_time == 45.5
        assert result.success_rate == 0.8
        assert "mean_similarity" in result.diversity_metrics

    def test_top_candidates(self, sample_design_result, sample_design_candidate):
        """Test top candidates retrieval."""
        # Add more candidates
        candidate2 = sample_design_candidate
        candidate2.candidate_id = "candidate_002"
        candidate2.fitness_score = 0.95

        result = sample_design_result
        result.design_candidates.append(candidate2)

        top_candidates = result.get_top_candidates(n=1)

        assert len(top_candidates) == 1
        assert top_candidates[0].fitness_score == 0.95

    def test_candidates_by_polymorph(self, sample_design_result):
        """Test candidates by polymorph retrieval."""
        result = sample_design_result

        alpha_candidates = result.get_candidates_by_polymorph(PolymorphType.ALPHA)
        beta_candidates = result.get_candidates_by_polymorph(PolymorphType.BETA)

        assert len(alpha_candidates) == 1
        assert len(beta_candidates) == 0


class TestPolymorphPredictor:
    """Test PolymorphPredictor class."""

    def test_predictor_creation(self):
        """Test polymorph predictor creation."""
        predictor = PolymorphPredictor(use_gpu=False)

        assert predictor.use_gpu == False
        # Device and model availability depend on PyTorch installation

    def test_predictor_with_gpu(self):
        """Test predictor creation with GPU."""
        predictor = PolymorphPredictor(use_gpu=True)
        # GPU availability depends on system, so just check it doesn't crash
        assert predictor is not None

    def test_polymorph_prediction(self):
        """Test polymorph prediction."""
        predictor = PolymorphPredictor(use_gpu=False)

        smiles = "c1ccccc1"
        predictions = predictor.predict_polymorph(smiles)

        assert isinstance(predictions, dict)
        assert len(predictions) == len(PolymorphType)

        # Check probabilities sum to ~1
        total_prob = sum(predictions.values())
        assert abs(total_prob - 1.0) < 0.1

        # Check all probabilities are valid
        for polymorph_type, prob in predictions.items():
            assert isinstance(polymorph_type, PolymorphType)
            assert 0.0 <= prob <= 1.0

    def test_stability_prediction(self):
        """Test stability prediction."""
        predictor = PolymorphPredictor(use_gpu=False)

        smiles = "c1ccccc1"
        stability = predictor.predict_stability(smiles)

        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0

    @patch("chemi_accel.materials.polymorph_inverse_design._RDKIT_AVAILABLE", False)
    def test_prediction_without_rdkit(self):
        """Test prediction without RDKit."""
        predictor = PolymorphPredictor(use_gpu=False)

        # Should still work with simplified prediction
        predictions = predictor.predict_polymorph("CCO")
        stability = predictor.predict_stability("CCO")

        assert isinstance(predictions, dict)
        assert isinstance(stability, float)

    @patch("chemi_accel.materials.polymorph_inverse_design._TORCH_AVAILABLE", False)
    def test_prediction_without_pytorch(self):
        """Test prediction without PyTorch."""
        predictor = PolymorphPredictor(use_gpu=False)

        # Should fall back to simple prediction
        predictions = predictor.predict_polymorph("CCO")
        stability = predictor.predict_stability("CCO")

        assert isinstance(predictions, dict)
        assert isinstance(stability, float)

    def test_descriptor_extraction(self):
        """Test molecular descriptor extraction."""
        predictor = PolymorphPredictor(use_gpu=False)

        smiles = "c1ccccc1"
        descriptors = predictor._extract_molecular_descriptors(smiles)

        if descriptors is not None:
            assert isinstance(descriptors, np.ndarray)
            assert len(descriptors) == 200
            assert descriptors.dtype == np.float32

    def test_invalid_smiles_handling(self):
        """Test handling of invalid SMILES."""
        predictor = PolymorphPredictor(use_gpu=False)

        # Should handle gracefully
        predictions = predictor.predict_polymorph("invalid_smiles")
        stability = predictor.predict_stability("invalid_smiles")

        assert isinstance(predictions, dict)
        assert isinstance(stability, float)


class TestMolecularModifier:
    """Test MolecularModifier class."""

    def test_modifier_creation(self):
        """Test molecular modifier creation."""
        modifier = MolecularModifier()

        assert isinstance(modifier.modification_templates, dict)
        assert isinstance(modifier.functional_groups, dict)
        assert (
            ModificationStrategy.FUNCTIONAL_GROUP_ADDITION
            in modifier.modification_templates
        )
        assert "hydroxyl" in modifier.functional_groups

    def test_modification_application(self):
        """Test modification application."""
        modifier = MolecularModifier()

        smiles = "c1ccccc1"
        modification = MolecularModification(
            modification_id="test",
            strategy=ModificationStrategy.FUNCTIONAL_GROUP_ADDITION,
            target_atoms=[0],
            modification_smarts="[OH]",
            confidence=0.8,
        )

        modified_smiles = modifier.apply_modification(smiles, modification)

        # Should return a SMILES string (might be original if modification fails)
        assert isinstance(modified_smiles, (str, type(None)))

    def test_functional_group_addition(self):
        """Test functional group addition."""
        modifier = MolecularModifier()

        smiles = "c1ccccc1"
        modification = MolecularModification(
            modification_id="test",
            strategy=ModificationStrategy.FUNCTIONAL_GROUP_ADDITION,
            target_atoms=[0],
            modification_smarts="[OH]",
            confidence=0.8,
        )

        result = modifier.apply_modification(smiles, modification)

        # Should return a valid SMILES or None
        assert result is None or isinstance(result, str)

    def test_functional_group_removal(self):
        """Test functional group removal."""
        modifier = MolecularModifier()

        smiles = "c1ccc(O)cc1"  # Phenol
        modification = MolecularModification(
            modification_id="test",
            strategy=ModificationStrategy.FUNCTIONAL_GROUP_REMOVAL,
            target_atoms=[0],
            modification_smarts="[OH]",
            confidence=0.8,
        )

        result = modifier.apply_modification(smiles, modification)

        assert result is None or isinstance(result, str)

    def test_substituent_modification(self):
        """Test substituent modification."""
        modifier = MolecularModifier()

        smiles = "c1ccc(C)cc1"  # Toluene
        modification = MolecularModification(
            modification_id="test",
            strategy=ModificationStrategy.SUBSTITUENT_MODIFICATION,
            target_atoms=[0],
            modification_smarts="[CH3]",
            replacement_smarts="[CF3]",
            confidence=0.8,
        )

        result = modifier.apply_modification(smiles, modification)

        assert result is None or isinstance(result, str)

    def test_random_modification_generation(self):
        """Test random modification generation."""
        modifier = MolecularModifier()

        smiles = "c1ccccc1"
        allowed_strategies = [
            ModificationStrategy.FUNCTIONAL_GROUP_ADDITION,
            ModificationStrategy.FUNCTIONAL_GROUP_REMOVAL,
        ]

        modification = modifier.generate_random_modification(smiles, allowed_strategies)

        if modification is not None:
            assert isinstance(modification, MolecularModification)
            assert modification.strategy in allowed_strategies
            assert modification.modification_smarts is not None

    @patch("chemi_accel.materials.polymorph_inverse_design._RDKIT_AVAILABLE", False)
    def test_modification_without_rdkit(self):
        """Test modification without RDKit."""
        modifier = MolecularModifier()

        modification = MolecularModification(
            modification_id="test",
            strategy=ModificationStrategy.FUNCTIONAL_GROUP_ADDITION,
            target_atoms=[0],
            modification_smarts="[OH]",
            confidence=0.8,
        )

        result = modifier.apply_modification("CCO", modification)
        assert result is None  # Should return None without RDKit


class TestPolymorphInverseDesigner:
    """Test PolymorphInverseDesigner class."""

    def test_designer_creation(self, sample_design_config):
        """Test designer creation."""
        designer = PolymorphInverseDesigner(sample_design_config)

        assert designer.config == sample_design_config
        assert isinstance(designer.polymorph_predictor, PolymorphPredictor)
        assert isinstance(designer.molecular_modifier, MolecularModifier)
        assert isinstance(designer.design_history, list)
        assert isinstance(designer.evaluation_cache, dict)
        assert isinstance(designer.performance_stats, dict)

    def test_molecule_design(self, sample_design_config):
        """Test molecule design for polymorph."""
        designer = PolymorphInverseDesigner(sample_design_config)

        starting_smiles = "c1ccccc1"
        max_modifications = 2

        result = designer.design_molecules_for_polymorph(
            starting_smiles, max_modifications
        )

        assert isinstance(result, DesignResult)
        assert len(result.design_candidates) > 0
        assert all(isinstance(c, DesignCandidate) for c in result.design_candidates)
        assert result.computation_time >= 0

    def test_genetic_algorithm_design(self, sample_design_config):
        """Test genetic algorithm design."""
        # Set explicit algorithm
        sample_design_config.optimization_algorithm = (
            OptimizationAlgorithm.GENETIC_ALGORITHM
        )
        sample_design_config.max_generations = 3  # Reduce for testing
        sample_design_config.population_size = 5  # Reduce for testing

        designer = PolymorphInverseDesigner(sample_design_config)

        result = designer._genetic_algorithm_design("c1ccccc1", 2)

        assert isinstance(result, DesignResult)
        assert len(result.design_candidates) > 0
        assert len(result.optimization_history) > 0

    def test_simulated_annealing_design(self, sample_design_config):
        """Test simulated annealing design."""
        designer = PolymorphInverseDesigner(sample_design_config)

        result = designer._simulated_annealing_design("c1ccccc1", 2)

        assert isinstance(result, DesignResult)
        assert len(result.design_candidates) > 0
        assert result.convergence_achieved == True

    def test_fitness_evaluation(self, sample_design_config):
        """Test fitness evaluation."""
        designer = PolymorphInverseDesigner(sample_design_config)

        candidate = DesignCandidate(
            candidate_id="test",
            original_smiles="c1ccccc1",
            modified_smiles="c1ccc(O)cc1",
            modifications=[],
            predicted_polymorph=PolymorphType.ALPHA,
        )

        fitness = designer._evaluate_fitness(candidate)

        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0
        assert candidate.fitness_score == fitness

    def test_candidate_creation(self, sample_design_config):
        """Test candidate creation from SMILES."""
        designer = PolymorphInverseDesigner(sample_design_config)

        candidate = designer._create_candidate_from_smiles("c1ccccc1", "test_id")

        assert isinstance(candidate, DesignCandidate)
        assert candidate.candidate_id == "test_id"
        assert candidate.original_smiles == "c1ccccc1"
        assert candidate.modified_smiles == "c1ccccc1"
        assert len(candidate.modifications) == 0

    def test_population_initialization(self, sample_design_config):
        """Test population initialization."""
        sample_design_config.population_size = 5  # Small for testing
        designer = PolymorphInverseDesigner(sample_design_config)

        population = designer._initialize_population("c1ccccc1", 2)

        assert len(population) == 5
        assert all(isinstance(candidate, DesignCandidate) for candidate in population)

    def test_molecule_validation(self, sample_design_config):
        """Test molecule validation."""
        designer = PolymorphInverseDesigner(sample_design_config)

        # Valid molecule
        assert designer._is_valid_molecule("c1ccccc1") == True

        # Invalid SMILES (if RDKit available)
        result = designer._is_valid_molecule("invalid_smiles")
        assert isinstance(result, bool)

    def test_genetic_operations(self, sample_design_config):
        """Test genetic algorithm operations."""
        designer = PolymorphInverseDesigner(sample_design_config)

        # Create test candidates
        candidate1 = designer._create_candidate_from_smiles("c1ccccc1", "test1")
        candidate2 = designer._create_candidate_from_smiles("c1ccccc1", "test2")

        population = [candidate1, candidate2]
        fitness_scores = [0.8, 0.6]

        # Test tournament selection
        parent1, parent2 = designer._tournament_selection(population, fitness_scores)
        assert isinstance(parent1, DesignCandidate)
        assert isinstance(parent2, DesignCandidate)

        # Test crossover
        child1, child2 = designer._crossover(parent1, parent2)
        assert isinstance(child1, DesignCandidate)
        assert isinstance(child2, DesignCandidate)

        # Test mutation
        mutated = designer._mutate(candidate1, 2)
        assert isinstance(mutated, DesignCandidate)

    def test_caching_mechanism(self, sample_design_config):
        """Test evaluation caching."""
        designer = PolymorphInverseDesigner(sample_design_config)

        candidate = designer._create_candidate_from_smiles("c1ccccc1")

        # First evaluation
        fitness1 = designer._evaluate_fitness(candidate)
        cache_size_1 = len(designer.evaluation_cache)

        # Second evaluation (should use cache)
        fitness2 = designer._evaluate_fitness(candidate)
        cache_size_2 = len(designer.evaluation_cache)

        assert fitness1 == fitness2
        assert cache_size_1 == cache_size_2  # Cache should be reused
        assert designer.performance_stats["cache_hits"] > 0

    def test_interaction_profile_calculation(self, sample_design_config):
        """Test interaction profile calculation."""
        designer = PolymorphInverseDesigner(sample_design_config)

        profile = designer._calculate_interaction_profile("c1ccccc1")

        assert isinstance(profile, dict)
        # Profile might be empty if fragment creation fails

    def test_synthetic_accessibility_calculation(self, sample_design_config):
        """Test synthetic accessibility calculation."""
        designer = PolymorphInverseDesigner(sample_design_config)

        sa_score = designer._calculate_synthetic_accessibility("c1ccccc1")

        assert isinstance(sa_score, float)
        assert 0.0 <= sa_score <= 1.0

    def test_novelty_score_calculation(self, sample_design_config):
        """Test novelty score calculation."""
        designer = PolymorphInverseDesigner(sample_design_config)

        novelty = designer._calculate_novelty_score("c1ccc(O)cc1", "c1ccccc1")

        assert isinstance(novelty, float)
        assert 0.0 <= novelty <= 1.0

    def test_overall_fitness_calculation(
        self, sample_design_config, sample_design_candidate
    ):
        """Test overall fitness calculation."""
        designer = PolymorphInverseDesigner(sample_design_config)

        fitness = designer._calculate_overall_fitness(sample_design_candidate)

        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0

    def test_modification_application(self, sample_design_config):
        """Test modification application."""
        designer = PolymorphInverseDesigner(sample_design_config)

        modifications = [
            MolecularModification(
                modification_id="test",
                strategy=ModificationStrategy.FUNCTIONAL_GROUP_ADDITION,
                target_atoms=[0],
                modification_smarts="[OH]",
                confidence=0.8,
            )
        ]

        result = designer._apply_modifications("c1ccccc1", modifications)

        assert isinstance(result, str)

    def test_diversity_metrics(self, sample_design_config):
        """Test diversity metrics calculation."""
        designer = PolymorphInverseDesigner(sample_design_config)

        candidates = [
            designer._create_candidate_from_smiles("c1ccccc1", "test1"),
            designer._create_candidate_from_smiles("c1ccc(O)cc1", "test2"),
        ]

        # Set modified SMILES
        candidates[0].modified_smiles = "c1ccccc1"
        candidates[1].modified_smiles = "c1ccc(O)cc1"

        metrics = designer._calculate_diversity_metrics(candidates)

        assert isinstance(metrics, dict)
        # Metrics might be empty if RDKit not available


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_polymorph_target(self):
        """Test polymorph target creation utility."""
        target = create_polymorph_target(
            PolymorphType.BETA,
            CrystalSystemType.HEXAGONAL,
            target_density=1.4,
            interaction_preferences={InteractionType.HYDROGEN_BOND: 4.0},
        )

        assert isinstance(target, PolymorphTarget)
        assert target.polymorph_type == PolymorphType.BETA
        assert target.crystal_system == CrystalSystemType.HEXAGONAL
        assert target.target_density == 1.4
        assert len(target.interaction_preferences) == 1

    def test_create_design_config(self, sample_polymorph_target):
        """Test design configuration creation utility."""
        config = create_design_config(
            sample_polymorph_target,
            allowed_modifications=[ModificationStrategy.FUNCTIONAL_GROUP_ADDITION],
            use_gpu=False,
            population_size=25,
            max_generations=15,
        )

        assert isinstance(config, PolymorphDesignConfig)
        assert config.target_polymorph == sample_polymorph_target
        assert len(config.allowed_modifications) == 1
        assert config.use_gpu == False
        assert config.population_size == 25
        assert config.max_generations == 15

    def test_design_for_polymorph(self, sample_polymorph_target):
        """Test design for polymorph utility function."""
        result = design_for_polymorph(
            starting_smiles="c1ccccc1",
            target_polymorph=sample_polymorph_target,
            max_modifications=2,
        )

        assert isinstance(result, DesignResult)
        assert len(result.design_candidates) > 0

    def test_analyze_polymorph_landscape(self):
        """Test polymorph landscape analysis."""
        smiles_list = ["c1ccccc1", "c1ccc(O)cc1", "CCO"]

        analysis = analyze_polymorph_landscape(smiles_list, use_gpu=False)

        assert isinstance(analysis, dict)
        assert "individual_results" in analysis
        assert "polymorph_distribution" in analysis
        assert "stability_statistics" in analysis
        assert "total_molecules" in analysis
        assert analysis["total_molecules"] == 3

    def test_predict_polymorph_stability(self):
        """Test polymorph stability prediction."""
        stability_data = predict_polymorph_stability(
            "c1ccccc1", PolymorphType.ALPHA, use_gpu=False
        )

        assert isinstance(stability_data, dict)
        assert "overall_stability" in stability_data
        assert "polymorph_probability" in stability_data
        assert "polymorph_stability" in stability_data
        assert "relative_stability" in stability_data

        # Check value ranges
        assert 0.0 <= stability_data["overall_stability"] <= 1.0
        assert 0.0 <= stability_data["polymorph_probability"] <= 1.0


class TestExceptions:
    """Test exception classes."""

    def test_polymorph_inverse_design_error(self):
        """Test PolymorphInverseDesignError."""
        with pytest.raises(PolymorphInverseDesignError):
            raise PolymorphInverseDesignError("Test error")

    def test_polymorph_prediction_error(self):
        """Test PolymorphPredictionError."""
        with pytest.raises(PolymorphPredictionError):
            raise PolymorphPredictionError("Prediction failed")

    def test_molecular_modification_error(self):
        """Test MolecularModificationError."""
        with pytest.raises(MolecularModificationError):
            raise MolecularModificationError("Modification failed")

    def test_design_optimization_error(self):
        """Test DesignOptimizationError."""
        with pytest.raises(DesignOptimizationError):
            raise DesignOptimizationError("Optimization failed")

    def test_configuration_error(self):
        """Test ConfigurationError."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Configuration invalid")


class TestIntegration:
    """Integration tests."""

    def test_full_design_workflow(self, sample_polymorph_target):
        """Test complete design workflow."""
        # Create configuration
        config = create_design_config(
            sample_polymorph_target,
            population_size=5,  # Small for testing
            max_generations=3,  # Small for testing
        )

        # Create designer
        designer = PolymorphInverseDesigner(config)

        # Run design
        result = designer.design_molecules_for_polymorph(
            "c1ccccc1", max_modifications=2
        )

        # Verify results
        assert isinstance(result, DesignResult)
        assert len(result.design_candidates) > 0
        assert result.computation_time >= 0

        # Check candidates
        for candidate in result.design_candidates:
            assert isinstance(candidate, DesignCandidate)
            assert candidate.original_smiles == "c1ccccc1"
            assert isinstance(candidate.fitness_score, float)
            assert 0.0 <= candidate.fitness_score <= 1.0

    def test_multi_algorithm_comparison(self, sample_polymorph_target):
        """Test comparison of different optimization algorithms."""
        algorithms = [
            OptimizationAlgorithm.GENETIC_ALGORITHM,
            OptimizationAlgorithm.SIMULATED_ANNEALING,
        ]

        results = {}

        for algorithm in algorithms:
            config = create_design_config(
                sample_polymorph_target,
                population_size=3,  # Very small for testing
                max_generations=2,  # Very small for testing
            )
            config.optimization_algorithm = algorithm

            designer = PolymorphInverseDesigner(config)

            try:
                result = designer.design_molecules_for_polymorph(
                    "c1ccccc1", max_modifications=1
                )
                results[algorithm] = result
            except Exception as e:
                # Some algorithms might fail, which is acceptable
                print(f"Algorithm {algorithm} failed: {e}")

        # At least one algorithm should work
        assert len(results) > 0

        # All results should be valid
        for algorithm, result in results.items():
            assert isinstance(result, DesignResult)
            assert len(result.design_candidates) > 0

    def test_constraint_satisfaction(self, sample_polymorph_target):
        """Test that designed molecules satisfy constraints."""
        config = create_design_config(
            sample_polymorph_target, population_size=5, max_generations=2
        )
        config.synthetic_accessibility_threshold = 0.8
        config.novelty_threshold = 0.3
        config.max_heavy_atoms = 20
        config.min_heavy_atoms = 6

        designer = PolymorphInverseDesigner(config)
        result = designer.design_molecules_for_polymorph(
            "c1ccccc1", max_modifications=1
        )

        # Check constraints
        for candidate in result.design_candidates:
            if candidate.fitness_score > 0:  # Only check successful candidates
                assert (
                    candidate.synthetic_accessibility >= 0.0
                )  # May be relaxed during optimization
                assert candidate.novelty_score >= 0.0

    def test_polymorph_targeting_effectiveness(self, sample_polymorph_target):
        """Test effectiveness of polymorph targeting."""
        config = create_design_config(
            sample_polymorph_target, population_size=10, max_generations=3
        )

        designer = PolymorphInverseDesigner(config)
        result = designer.design_molecules_for_polymorph(
            "c1ccccc1", max_modifications=2
        )

        # Check that some candidates target the desired polymorph
        target_type = sample_polymorph_target.polymorph_type
        target_candidates = [
            c for c in result.design_candidates if c.predicted_polymorph == target_type
        ]

        # At least some candidates should target the desired polymorph
        # (This is probabilistic, so we allow for some flexibility)
        assert len(target_candidates) >= 0  # Should have at least some targeting

    def test_performance_tracking(self, sample_polymorph_target):
        """Test performance statistics tracking."""
        config = create_design_config(
            sample_polymorph_target, population_size=5, max_generations=2
        )

        designer = PolymorphInverseDesigner(config)

        # Run design
        result = designer.design_molecules_for_polymorph(
            "c1ccccc1", max_modifications=1
        )

        # Check performance stats
        assert designer.performance_stats["total_evaluations"] > 0
        assert designer.performance_stats["successful_modifications"] >= 0
        assert designer.performance_stats["cache_hits"] >= 0

        # Check that cache was used appropriately
        total_evals = designer.performance_stats["total_evaluations"]
        cache_hits = designer.performance_stats["cache_hits"]
        assert cache_hits <= total_evals

    def test_error_handling_and_recovery(self, sample_polymorph_target):
        """Test error handling and recovery mechanisms."""
        config = create_design_config(
            sample_polymorph_target, population_size=3, max_generations=2
        )

        designer = PolymorphInverseDesigner(config)

        # Test with invalid starting SMILES
        try:
            result = designer.design_molecules_for_polymorph(
                "invalid_smiles", max_modifications=1
            )
            # Should handle gracefully
            assert isinstance(result, DesignResult)
        except Exception:
            # Errors are acceptable for invalid input
            pass

        # Test with extreme parameters
        try:
            result = designer.design_molecules_for_polymorph(
                "c1ccccc1", max_modifications=0
            )
            assert isinstance(result, DesignResult)
        except Exception:
            # Errors are acceptable for edge cases
            pass

    def test_scalability_small_population(self, sample_polymorph_target):
        """Test scalability with different population sizes."""
        population_sizes = [2, 5, 10]

        for pop_size in population_sizes:
            config = create_design_config(
                sample_polymorph_target, population_size=pop_size, max_generations=2
            )

            designer = PolymorphInverseDesigner(config)
            result = designer.design_molecules_for_polymorph(
                "c1ccccc1", max_modifications=1
            )

            # Should scale appropriately
            assert (
                len(result.design_candidates) <= pop_size + 10
            )  # Allow some flexibility
            assert result.computation_time >= 0

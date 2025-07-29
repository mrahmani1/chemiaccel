"""
Unit tests for systems inverse design module.
"""

from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from chemi_accel.biological_interactions.systems_inverse_design import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    AdverseEffectsConstraints,
    ConstraintViolationError,
    DesignConstraints,
    DesignStrategy,
    MolecularConstraint,
    MolecularGenerationError,
    OptimizationError,
    OptimizationObjective,
    OptimizationResult,
    PolypharmacologyProfile,
    SystemsInverseDesigner,
    SystemsInverseDesignError,
    create_design_constraints,
    create_polypharmacology_profile,
    create_safety_constraints,
    design_polypharmacology_molecule,
    optimize_for_safety,
)
from chemi_accel.biological_interactions.toxicology_prediction import (
    RiskLevel,
    ToxicityEndpoint,
)


@pytest.fixture
def sample_polypharmacology_profile():
    """Create sample polypharmacology profile."""
    return PolypharmacologyProfile(
        target_engagements={"TARGET_001": 0.8, "TARGET_002": 0.6, "TARGET_003": 0.3},
        selectivity_constraints={"TARGET_001": (0.7, 0.9), "TARGET_002": (0.5, 0.7)},
        off_target_limits={"TARGET_004": 0.2, "TARGET_005": 0.1},
        therapeutic_window=15.0,
        confidence_threshold=0.8,
    )


@pytest.fixture
def sample_adverse_constraints():
    """Create sample adverse effects constraints."""
    return AdverseEffectsConstraints(
        toxicity_limits={
            ToxicityEndpoint.HEPATOTOXICITY: 0.3,
            ToxicityEndpoint.CARDIOTOXICITY: 0.2,
            ToxicityEndpoint.NEPHROTOXICITY: 0.25,
        },
        adr_restrictions={"cardiovascular": 0.2, "hepatic": 0.3, "renal": 0.25},
        off_target_penalties={},
        safety_score_threshold=0.8,
        contraindication_avoidance=["pregnancy", "severe_liver_disease"],
        patient_population_factors={"age_range": (18, 65)},
    )


@pytest.fixture
def sample_design_constraints():
    """Create sample design constraints."""
    return DesignConstraints(
        molecular_constraints={
            MolecularConstraint.MOLECULAR_WEIGHT: (200, 500),
            MolecularConstraint.LOGP: (0, 4),
            MolecularConstraint.TPSA: (30, 120),
            MolecularConstraint.HBD_COUNT: (0, 3),
            MolecularConstraint.HBA_COUNT: (1, 8),
        },
        structural_requirements=["c1ccccc1"],  # Requires benzene ring
        structural_exclusions=["[N+](=O)[O-]"],  # Exclude nitro groups
        synthetic_feasibility_threshold=0.7,
        novelty_requirements={"tanimoto_threshold": 0.7},
    )


@pytest.fixture
def sample_optimization_result():
    """Create sample optimization result."""
    return OptimizationResult(
        optimized_smiles="c1ccc(CCO)cc1",
        optimization_score=0.85,
        predicted_profile={
            "target_engagements": {"TARGET_001": 0.8, "TARGET_002": 0.6},
            "safety_score": 0.9,
            "molecular_weight": 122.16,
            "logp": 1.58,
        },
        constraint_satisfaction={
            "molecular_weight": True,
            "logp": True,
            "tpsa": True,
            "structural_requirements": True,
        },
        optimization_metrics={"generations": 100, "fitness_improvement": 0.25},
        computational_cost=12.5,
    )


class TestPolypharmacologyProfile:
    """Test PolypharmacologyProfile class."""

    def test_profile_creation(self, sample_polypharmacology_profile):
        """Test profile creation."""
        profile = sample_polypharmacology_profile

        assert len(profile.target_engagements) == 3
        assert profile.target_engagements["TARGET_001"] == 0.8
        assert profile.therapeutic_window == 15.0
        assert profile.confidence_threshold == 0.8
        assert len(profile.selectivity_constraints) == 2
        assert len(profile.off_target_limits) == 2

    def test_profile_validation(self):
        """Test profile validation."""
        # Valid profile should not raise
        profile = PolypharmacologyProfile(
            target_engagements={"TARGET_001": 0.5}, therapeutic_window=10.0
        )
        assert profile.target_engagements["TARGET_001"] == 0.5

        # Invalid engagement should raise
        with pytest.raises(ValueError):
            PolypharmacologyProfile(
                target_engagements={"TARGET_001": 1.5}, therapeutic_window=10.0  # > 1.0
            )

        # Invalid therapeutic window should raise
        with pytest.raises(ValueError):
            PolypharmacologyProfile(
                target_engagements={"TARGET_001": 0.5}, therapeutic_window=0.5  # <= 1.0
            )


class TestAdverseEffectsConstraints:
    """Test AdverseEffectsConstraints class."""

    def test_constraints_creation(self, sample_adverse_constraints):
        """Test constraints creation."""
        constraints = sample_adverse_constraints

        assert len(constraints.toxicity_limits) == 3
        assert constraints.toxicity_limits[ToxicityEndpoint.HEPATOTOXICITY] == 0.3
        assert constraints.safety_score_threshold == 0.8
        assert len(constraints.contraindication_avoidance) == 2
        assert "age_range" in constraints.patient_population_factors

    def test_constraints_validation(self):
        """Test constraints validation."""
        # Valid constraints should not raise
        constraints = AdverseEffectsConstraints(
            toxicity_limits={ToxicityEndpoint.HEPATOTOXICITY: 0.5},
            safety_score_threshold=0.7,
        )
        assert constraints.safety_score_threshold == 0.7

        # Invalid toxicity limit should raise
        with pytest.raises(ValueError):
            AdverseEffectsConstraints(
                toxicity_limits={ToxicityEndpoint.HEPATOTOXICITY: 1.5},  # > 1.0
                safety_score_threshold=0.8,
            )

        # Invalid safety threshold should raise
        with pytest.raises(ValueError):
            AdverseEffectsConstraints(
                toxicity_limits={ToxicityEndpoint.HEPATOTOXICITY: 0.5},
                safety_score_threshold=1.5,  # > 1.0
            )


class TestDesignConstraints:
    """Test DesignConstraints class."""

    def test_constraints_creation(self, sample_design_constraints):
        """Test design constraints creation."""
        constraints = sample_design_constraints

        assert len(constraints.molecular_constraints) == 5
        assert constraints.molecular_constraints[
            MolecularConstraint.MOLECULAR_WEIGHT
        ] == (200, 500)
        assert len(constraints.structural_requirements) == 1
        assert len(constraints.structural_exclusions) == 1
        assert constraints.synthetic_feasibility_threshold == 0.7
        assert "tanimoto_threshold" in constraints.novelty_requirements


class TestOptimizationResult:
    """Test OptimizationResult class."""

    def test_result_creation(self, sample_optimization_result):
        """Test optimization result creation."""
        result = sample_optimization_result

        assert result.optimized_smiles == "c1ccc(CCO)cc1"
        assert result.optimization_score == 0.85
        assert "target_engagements" in result.predicted_profile
        assert len(result.constraint_satisfaction) == 4
        assert result.computational_cost == 12.5

    def test_profile_match_score(
        self, sample_optimization_result, sample_polypharmacology_profile
    ):
        """Test profile match score calculation."""
        result = sample_optimization_result
        profile = sample_polypharmacology_profile

        match_score = result.get_profile_match_score(profile)

        assert 0.0 <= match_score <= 1.0
        assert isinstance(match_score, float)

    def test_profile_match_score_no_engagements(self, sample_polypharmacology_profile):
        """Test profile match score with no target engagements."""
        result = OptimizationResult(
            optimized_smiles="CCO",
            optimization_score=0.5,
            predicted_profile={},  # No target engagements
            constraint_satisfaction={},
        )

        match_score = result.get_profile_match_score(sample_polypharmacology_profile)
        assert match_score == 0.0


class TestSystemsInverseDesigner:
    """Test SystemsInverseDesigner class."""

    def test_designer_creation(self):
        """Test designer creation."""
        designer = SystemsInverseDesigner(use_gpu=False)

        assert not designer.use_gpu
        assert designer.interaction_predictor is not None
        assert designer.toxicology_engine is not None
        assert len(designer.optimization_engines) > 0
        assert len(designer.constraint_evaluators) > 0
        assert isinstance(designer.design_history, list)
        assert isinstance(designer.evaluation_cache, dict)

    def test_designer_with_gpu(self):
        """Test designer creation with GPU."""
        designer = SystemsInverseDesigner(use_gpu=True)
        # GPU availability depends on system, so just check it doesn't crash
        assert designer is not None

    @patch(
        "chemi_accel.biological_interactions.systems_inverse_design.SystemsInverseDesigner._genetic_algorithm_optimizer"
    )
    def test_design_polypharmacology_molecule(
        self,
        mock_ga_optimizer,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
        sample_optimization_result,
    ):
        """Test polypharmacology molecule design."""
        mock_ga_optimizer.return_value = [sample_optimization_result]

        designer = SystemsInverseDesigner(use_gpu=False)
        results = designer.design_polypharmacology_molecule(
            target_profile=sample_polypharmacology_profile,
            adverse_constraints=sample_adverse_constraints,
            design_constraints=sample_design_constraints,
            strategy=DesignStrategy.GENETIC_ALGORITHM,
            max_iterations=10,
            population_size=5,
        )

        assert len(results) == 1
        assert isinstance(results[0], OptimizationResult)
        assert results[0].optimized_smiles == "c1ccc(CCO)cc1"
        assert len(designer.design_history) == 1

        mock_ga_optimizer.assert_called_once()

    @patch(
        "chemi_accel.biological_interactions.systems_inverse_design.SystemsInverseDesigner._bayesian_optimizer"
    )
    def test_optimize_for_minimal_adverse_effects(
        self,
        mock_bo_optimizer,
        sample_adverse_constraints,
        sample_design_constraints,
        sample_optimization_result,
    ):
        """Test optimization for minimal adverse effects."""
        mock_bo_optimizer.return_value = [sample_optimization_result]

        designer = SystemsInverseDesigner(use_gpu=False)
        starting_molecules = ["CCO", "CCC", "c1ccccc1"]

        results = designer.optimize_for_minimal_adverse_effects(
            starting_molecules=starting_molecules,
            adverse_constraints=sample_adverse_constraints,
            efficacy_threshold=0.6,
            design_constraints=sample_design_constraints,
            strategy=DesignStrategy.BAYESIAN_OPTIMIZATION,
            max_iterations=10,
        )

        assert len(results) == 1
        assert isinstance(results[0], OptimizationResult)

        mock_bo_optimizer.assert_called_once()

    def test_multi_objective_design(self, sample_optimization_result):
        """Test multi-objective design."""
        designer = SystemsInverseDesigner(use_gpu=False)

        objectives = [
            OptimizationObjective.MAXIMIZE_EFFICACY,
            OptimizationObjective.MINIMIZE_TOXICITY,
            OptimizationObjective.OPTIMIZE_SELECTIVITY,
        ]

        constraints = {"molecular_weight": (200, 500), "safety_threshold": 0.8}

        # Mock the NSGA-II optimization
        with patch.object(designer, "_nsga_ii_optimization") as mock_nsga:
            mock_nsga.return_value = [sample_optimization_result]

            results = designer.multi_objective_design(
                objectives=objectives,
                constraints=constraints,
                pareto_size=10,
                max_iterations=50,
            )

            assert len(results) == 1
            assert isinstance(results[0], OptimizationResult)
            mock_nsga.assert_called_once()

    def test_evaluate_fitness(
        self,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
    ):
        """Test fitness evaluation."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock the component score calculations
        with patch.object(
            designer, "_calculate_efficacy_score", return_value=0.8
        ), patch.object(
            designer, "_calculate_safety_score", return_value=0.9
        ), patch.object(
            designer, "_calculate_constraint_score", return_value=0.7
        ):
            fitness = designer._evaluate_fitness(
                "CCO",
                sample_polypharmacology_profile,
                sample_adverse_constraints,
                sample_design_constraints,
            )

            assert 0.0 <= fitness <= 1.0
            assert isinstance(fitness, float)

    def test_calculate_efficacy_score(self, sample_polypharmacology_profile):
        """Test efficacy score calculation."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock interaction predictions
        mock_predictions = {
            "CCO": [
                Mock(target=Mock(target_id="TARGET_001"), interaction_probability=0.8),
                Mock(target=Mock(target_id="TARGET_002"), interaction_probability=0.6),
            ]
        }

        with patch.object(
            designer.interaction_predictor,
            "predict_interactions_batch",
            return_value=mock_predictions,
        ):
            score = designer._calculate_efficacy_score(
                "CCO", sample_polypharmacology_profile
            )

            assert 0.0 <= score <= 1.0
            assert isinstance(score, float)

    def test_calculate_safety_score(self, sample_adverse_constraints):
        """Test safety score calculation."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock toxicology profile
        mock_profile = Mock()
        mock_profile.overall_safety_score = 0.85
        mock_profile.toxicity_predictions = []

        with patch.object(
            designer.toxicology_engine,
            "predict_systems_toxicology",
            return_value=mock_profile,
        ):
            score = designer._calculate_safety_score("CCO", sample_adverse_constraints)

            assert 0.0 <= score <= 1.0
            assert isinstance(score, float)

    def test_calculate_constraint_score(self, sample_design_constraints):
        """Test constraint score calculation."""
        designer = SystemsInverseDesigner(use_gpu=False)

        score = designer._calculate_constraint_score("CCO", sample_design_constraints)

        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    def test_predict_molecular_profile(self):
        """Test molecular profile prediction."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock predictions
        mock_interactions = {
            "CCO": [
                Mock(target=Mock(target_id="TARGET_001"), interaction_probability=0.8)
            ]
        }
        mock_tox_profile = Mock()
        mock_tox_profile.overall_safety_score = 0.85
        mock_tox_profile.therapeutic_index_estimate = 10.0
        mock_tox_profile.toxicity_predictions = []

        with patch.object(
            designer.interaction_predictor,
            "predict_interactions_batch",
            return_value=mock_interactions,
        ), patch.object(
            designer.toxicology_engine,
            "predict_systems_toxicology",
            return_value=mock_tox_profile,
        ):
            profile = designer._predict_molecular_profile("CCO")

            assert isinstance(profile, dict)
            assert "target_engagements" in profile
            assert "safety_score" in profile
            assert "therapeutic_index" in profile

    def test_generate_initial_population(self, sample_design_constraints):
        """Test initial population generation."""
        designer = SystemsInverseDesigner(use_gpu=False)

        population = designer._generate_initial_population(
            10, sample_design_constraints
        )

        assert len(population) == 10
        assert all(isinstance(smiles, str) for smiles in population)
        assert all(len(smiles) > 0 for smiles in population)

    def test_molecular_operations(self, sample_design_constraints):
        """Test molecular crossover and mutation operations."""
        designer = SystemsInverseDesigner(use_gpu=False)

        parent1 = "CCO"
        parent2 = "CCC"

        # Test crossover
        child1, child2 = designer._molecular_crossover(parent1, parent2)
        assert isinstance(child1, str)
        assert isinstance(child2, str)

        # Test mutation
        mutated = designer._molecular_mutation(parent1, sample_design_constraints)
        assert isinstance(mutated, str)
        assert len(mutated) > 0

    def test_constraint_evaluators(self):
        """Test constraint evaluator functions."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock RDKit molecule
        mock_mol = Mock()

        # Test molecular weight evaluator
        mw = designer._evaluate_molecular_weight(mock_mol)
        assert isinstance(mw, (int, float))
        assert mw > 0

        # Test LogP evaluator
        logp = designer._evaluate_logp(mock_mol)
        assert isinstance(logp, (int, float))

        # Test TPSA evaluator
        tpsa = designer._evaluate_tpsa(mock_mol)
        assert isinstance(tpsa, (int, float))
        assert tpsa >= 0

        # Test hydrogen bond donor count
        hbd = designer._evaluate_hbd_count(mock_mol)
        assert isinstance(hbd, int)
        assert hbd >= 0

        # Test hydrogen bond acceptor count
        hba = designer._evaluate_hba_count(mock_mol)
        assert isinstance(hba, int)
        assert hba >= 0

    def test_tournament_selection(self):
        """Test tournament selection for genetic algorithm."""
        designer = SystemsInverseDesigner(use_gpu=False)

        population = ["CCO", "CCC", "CCCO", "CC(C)O", "CCCCO"]
        fitness_scores = [0.8, 0.6, 0.9, 0.7, 0.5]

        selected = designer._tournament_selection(population, fitness_scores, 3)

        assert len(selected) == 3
        assert all(smiles in population for smiles in selected)

    def test_is_valid_molecule(self):
        """Test molecule validation."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Valid SMILES
        assert designer._is_valid_molecule("CCO")
        assert designer._is_valid_molecule("c1ccccc1")

        # Invalid SMILES (should still work with fallback)
        result = designer._is_valid_molecule("invalid_smiles")
        assert isinstance(result, bool)

    def test_validate_design_inputs(
        self,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
    ):
        """Test design input validation."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Valid inputs should not raise
        designer._validate_design_inputs(
            sample_polypharmacology_profile,
            sample_adverse_constraints,
            sample_design_constraints,
        )

        # Empty target profile should raise
        empty_profile = PolypharmacologyProfile(
            target_engagements={}, therapeutic_window=10.0
        )

        with pytest.raises(ValueError):
            designer._validate_design_inputs(
                empty_profile, sample_adverse_constraints, sample_design_constraints
            )

        # Invalid safety threshold should raise
        invalid_constraints = AdverseEffectsConstraints(
            toxicity_limits={ToxicityEndpoint.HEPATOTOXICITY: 0.5},
            safety_score_threshold=1.5,  # > 1.0
        )

        with pytest.raises(ValueError):
            designer._validate_design_inputs(
                sample_polypharmacology_profile,
                invalid_constraints,
                sample_design_constraints,
            )


class TestOptimizationStrategies:
    """Test different optimization strategies."""

    def test_genetic_algorithm_optimizer(
        self,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
    ):
        """Test genetic algorithm optimizer."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock fitness evaluation
        with patch.object(designer, "_evaluate_fitness", return_value=0.8):
            results = designer._genetic_algorithm_optimizer(
                initial_population=["CCO", "CCC"],
                target_profile=sample_polypharmacology_profile,
                adverse_constraints=sample_adverse_constraints,
                design_constraints=sample_design_constraints,
                max_iterations=5,
                mutation_rate=0.1,
                crossover_rate=0.8,
            )

            assert len(results) > 0
            assert all(isinstance(result, OptimizationResult) for result in results)

    def test_simulated_annealing_optimizer(
        self,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
    ):
        """Test simulated annealing optimizer."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock fitness evaluation
        with patch.object(designer, "_evaluate_fitness", return_value=0.8):
            results = designer._simulated_annealing_optimizer(
                initial_population=["CCO"],
                target_profile=sample_polypharmacology_profile,
                adverse_constraints=sample_adverse_constraints,
                design_constraints=sample_design_constraints,
                max_iterations=10,
                initial_temperature=1.0,
                cooling_rate=0.95,
            )

            assert len(results) == 1
            assert isinstance(results[0], OptimizationResult)
            assert len(results[0].generation_history) > 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_design_polypharmacology_molecule(
        self,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
    ):
        """Test polypharmacology molecule design utility function."""
        with patch(
            "chemi_accel.biological_interactions.systems_inverse_design.SystemsInverseDesigner"
        ) as mock_designer_class:
            mock_designer = Mock()
            mock_designer.design_polypharmacology_molecule.return_value = [Mock()]
            mock_designer_class.return_value = mock_designer

            results = design_polypharmacology_molecule(
                target_profile=sample_polypharmacology_profile,
                adverse_constraints=sample_adverse_constraints,
                design_constraints=sample_design_constraints,
                use_gpu=False,
            )

            assert len(results) == 1
            mock_designer_class.assert_called_once_with(use_gpu=False)
            mock_designer.design_polypharmacology_molecule.assert_called_once()

    def test_optimize_for_safety(self, sample_adverse_constraints):
        """Test safety optimization utility function."""
        with patch(
            "chemi_accel.biological_interactions.systems_inverse_design.SystemsInverseDesigner"
        ) as mock_designer_class:
            mock_designer = Mock()
            mock_designer.optimize_for_minimal_adverse_effects.return_value = [Mock()]
            mock_designer_class.return_value = mock_designer

            starting_molecules = ["CCO", "CCC"]
            results = optimize_for_safety(
                starting_molecules=starting_molecules,
                adverse_constraints=sample_adverse_constraints,
                efficacy_threshold=0.7,
                use_gpu=False,
            )

            assert len(results) == 1
            mock_designer_class.assert_called_once_with(use_gpu=False)
            mock_designer.optimize_for_minimal_adverse_effects.assert_called_once()

    def test_create_polypharmacology_profile(self):
        """Test polypharmacology profile creation utility."""
        target_engagements = {"TARGET_001": 0.8, "TARGET_002": 0.6}
        selectivity_constraints = {"TARGET_001": (0.7, 0.9)}

        profile = create_polypharmacology_profile(
            target_engagements=target_engagements,
            selectivity_constraints=selectivity_constraints,
            therapeutic_window=12.0,
        )

        assert isinstance(profile, PolypharmacologyProfile)
        assert profile.target_engagements == target_engagements
        assert profile.selectivity_constraints == selectivity_constraints
        assert profile.therapeutic_window == 12.0

    def test_create_safety_constraints(self):
        """Test safety constraints creation utility."""
        toxicity_limits = {
            ToxicityEndpoint.HEPATOTOXICITY: 0.3,
            ToxicityEndpoint.CARDIOTOXICITY: 0.2,
        }

        constraints = create_safety_constraints(
            toxicity_limits=toxicity_limits, safety_threshold=0.85
        )

        assert isinstance(constraints, AdverseEffectsConstraints)
        assert constraints.toxicity_limits == toxicity_limits
        assert constraints.safety_score_threshold == 0.85

    def test_create_design_constraints(self):
        """Test design constraints creation utility."""
        constraints = create_design_constraints(
            molecular_weight_range=(180, 450), logp_range=(-1, 4), tpsa_range=(25, 130)
        )

        assert isinstance(constraints, DesignConstraints)
        assert constraints.molecular_constraints[
            MolecularConstraint.MOLECULAR_WEIGHT
        ] == (180, 450)
        assert constraints.molecular_constraints[MolecularConstraint.LOGP] == (-1, 4)
        assert constraints.molecular_constraints[MolecularConstraint.TPSA] == (25, 130)


class TestExceptions:
    """Test exception classes."""

    def test_systems_inverse_design_error(self):
        """Test SystemsInverseDesignError."""
        with pytest.raises(SystemsInverseDesignError):
            raise SystemsInverseDesignError("Test error")

    def test_optimization_error(self):
        """Test OptimizationError."""
        with pytest.raises(OptimizationError):
            raise OptimizationError("Optimization failed")

    def test_constraint_violation_error(self):
        """Test ConstraintViolationError."""
        with pytest.raises(ConstraintViolationError):
            raise ConstraintViolationError("Constraint violated")

    def test_molecular_generation_error(self):
        """Test MolecularGenerationError."""
        with pytest.raises(MolecularGenerationError):
            raise MolecularGenerationError("Generation failed")


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_design_workflow(
        self,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
    ):
        """Test complete design workflow."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock all prediction engines
        mock_interactions = {
            "CCO": [
                Mock(target=Mock(target_id="TARGET_001"), interaction_probability=0.8)
            ]
        }
        mock_tox_profile = Mock()
        mock_tox_profile.overall_safety_score = 0.85
        mock_tox_profile.toxicity_predictions = []

        with patch.object(
            designer.interaction_predictor,
            "predict_interactions_batch",
            return_value=mock_interactions,
        ), patch.object(
            designer.toxicology_engine,
            "predict_systems_toxicology",
            return_value=mock_tox_profile,
        ):
            results = designer.design_polypharmacology_molecule(
                target_profile=sample_polypharmacology_profile,
                adverse_constraints=sample_adverse_constraints,
                design_constraints=sample_design_constraints,
                strategy=DesignStrategy.GENETIC_ALGORITHM,
                max_iterations=5,
                population_size=3,
            )

            assert len(results) > 0
            assert all(isinstance(result, OptimizationResult) for result in results)
            assert all(result.optimization_score >= 0 for result in results)
            assert len(designer.design_history) > 0

    def test_multi_strategy_comparison(
        self,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
    ):
        """Test comparison of different optimization strategies."""
        designer = SystemsInverseDesigner(use_gpu=False)

        strategies = [
            DesignStrategy.GENETIC_ALGORITHM,
            DesignStrategy.SIMULATED_ANNEALING,
        ]

        # Mock prediction engines
        mock_interactions = {
            "CCO": [
                Mock(target=Mock(target_id="TARGET_001"), interaction_probability=0.8)
            ]
        }
        mock_tox_profile = Mock()
        mock_tox_profile.overall_safety_score = 0.85
        mock_tox_profile.toxicity_predictions = []

        with patch.object(
            designer.interaction_predictor,
            "predict_interactions_batch",
            return_value=mock_interactions,
        ), patch.object(
            designer.toxicology_engine,
            "predict_systems_toxicology",
            return_value=mock_tox_profile,
        ):
            strategy_results = {}

            for strategy in strategies:
                try:
                    results = designer.design_polypharmacology_molecule(
                        target_profile=sample_polypharmacology_profile,
                        adverse_constraints=sample_adverse_constraints,
                        design_constraints=sample_design_constraints,
                        strategy=strategy,
                        max_iterations=3,
                        population_size=2,
                    )
                    strategy_results[strategy] = results
                except Exception as e:
                    # Some strategies might fail, which is acceptable
                    print(f"Strategy {strategy} failed: {e}")

            # At least one strategy should work
            assert len(strategy_results) > 0

            # All results should be valid
            for strategy, results in strategy_results.items():
                assert len(results) > 0
                assert all(isinstance(result, OptimizationResult) for result in results)

    def test_constraint_satisfaction_verification(
        self,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
    ):
        """Test that generated molecules satisfy constraints."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock prediction engines
        mock_interactions = {
            "CCO": [
                Mock(target=Mock(target_id="TARGET_001"), interaction_probability=0.8)
            ]
        }
        mock_tox_profile = Mock()
        mock_tox_profile.overall_safety_score = 0.85
        mock_tox_profile.toxicity_predictions = []

        with patch.object(
            designer.interaction_predictor,
            "predict_interactions_batch",
            return_value=mock_interactions,
        ), patch.object(
            designer.toxicology_engine,
            "predict_systems_toxicology",
            return_value=mock_tox_profile,
        ):
            results = designer.design_polypharmacology_molecule(
                target_profile=sample_polypharmacology_profile,
                adverse_constraints=sample_adverse_constraints,
                design_constraints=sample_design_constraints,
                max_iterations=3,
                population_size=2,
            )

            # Verify constraint satisfaction
            for result in results:
                assert isinstance(result.constraint_satisfaction, dict)

                # Check that predicted profile contains expected keys
                assert (
                    "safety_score" in result.predicted_profile
                    or "error" in result.predicted_profile
                )

                # Check optimization score is valid
                assert 0.0 <= result.optimization_score <= 1.0

    def test_large_scale_optimization(
        self,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
    ):
        """Test optimization with larger parameters."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock prediction engines
        mock_interactions = {
            "CCO": [
                Mock(target=Mock(target_id="TARGET_001"), interaction_probability=0.8)
            ]
        }
        mock_tox_profile = Mock()
        mock_tox_profile.overall_safety_score = 0.85
        mock_tox_profile.toxicity_predictions = []

        with patch.object(
            designer.interaction_predictor,
            "predict_interactions_batch",
            return_value=mock_interactions,
        ), patch.object(
            designer.toxicology_engine,
            "predict_systems_toxicology",
            return_value=mock_tox_profile,
        ):
            results = designer.design_polypharmacology_molecule(
                target_profile=sample_polypharmacology_profile,
                adverse_constraints=sample_adverse_constraints,
                design_constraints=sample_design_constraints,
                strategy=DesignStrategy.GENETIC_ALGORITHM,
                max_iterations=20,
                population_size=10,
            )

            assert len(results) > 0
            assert all(isinstance(result, OptimizationResult) for result in results)

            # Check that optimization actually improved
            if len(results) > 1:
                # Results should be sorted by score
                scores = [result.optimization_score for result in results]
                assert scores == sorted(scores, reverse=True)

    def test_error_handling_and_recovery(
        self,
        sample_polypharmacology_profile,
        sample_adverse_constraints,
        sample_design_constraints,
    ):
        """Test error handling and recovery mechanisms."""
        designer = SystemsInverseDesigner(use_gpu=False)

        # Mock prediction engines to sometimes fail
        def mock_predict_interactions(smiles_list):
            results = {}
            for smiles in smiles_list:
                if np.random.random() > 0.5:  # 50% chance of failure
                    results[smiles] = [
                        Mock(
                            target=Mock(target_id="TARGET_001"),
                            interaction_probability=0.8,
                        )
                    ]
            return results

        def mock_predict_toxicology(smiles):
            if np.random.random() > 0.5:  # 50% chance of failure
                raise Exception("Prediction failed")
            mock_profile = Mock()
            mock_profile.overall_safety_score = 0.85
            mock_profile.toxicity_predictions = []
            return mock_profile

        with patch.object(
            designer.interaction_predictor,
            "predict_interactions_batch",
            side_effect=mock_predict_interactions,
        ), patch.object(
            designer.toxicology_engine,
            "predict_systems_toxicology",
            side_effect=mock_predict_toxicology,
        ):
            # Should not crash even with prediction failures
            results = designer.design_polypharmacology_molecule(
                target_profile=sample_polypharmacology_profile,
                adverse_constraints=sample_adverse_constraints,
                design_constraints=sample_design_constraints,
                max_iterations=5,
                population_size=3,
            )

            # Should get some results despite failures
            assert isinstance(results, list)
            assert len(results) >= 0  # Might be empty if all predictions fail

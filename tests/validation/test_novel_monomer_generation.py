"""
Validation tests for novel monomer generation capabilities.

This module validates the InversePolymerDesigner's ability to generate
novel monomers that meet specified design criteria and constraints.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.chemi_accel.core.calculator import Calculator

# Import the modules to test
from src.chemi_accel.polymer.inverse_design import (
    BioDerivedScorer,
    InversePolymerDesigner,
    PolymerConstraintConfig,
    PolymerDesignObjective,
    create_bio_derived_constraints,
    create_biodegradable_polymer_objectives,
    create_high_performance_constraints,
    create_high_performance_polymer_objectives,
)
from src.chemi_accel.polymer.predictive import PolymerPropertyEngine


class TestNovelMonomerGeneration:
    """Validation tests for novel monomer generation."""

    def setup_method(self):
        """Set up test fixtures with realistic polymer design scenarios."""
        # Create mock calculator with realistic responses
        self.mock_calculator = Mock(spec=Calculator)

        # Create mock polymer engine with realistic responses
        self.mock_polymer_engine = Mock(spec=PolymerPropertyEngine)

        # Realistic seed monomers from different chemical classes
        self.bio_derived_seeds = [
            "CC(O)C(=O)O",  # Lactic acid (bio-derived)
            "CCCCCCCC(=O)O",  # Octanoic acid (fatty acid)
            "c1ccc(O)c(OC)c1",  # Guaiacol (lignin-derived)
            "OCC(O)CO",  # Glycerol (bio-derived)
            "c1ccoc1",  # Furan (bio-derived platform)
        ]

        self.synthetic_seeds = [
            "c1ccc(C(=O)O)cc1",  # Benzoic acid
            "c1ccc(N)cc1",  # Aniline
            "CC(C)(C)C(=O)O",  # tert-Butyl carboxylic acid
            "c1ccc(C#N)cc1",  # Benzonitrile
            "CC(=O)Cl",  # Acetyl chloride
        ]

        self.all_seeds = self.bio_derived_seeds + self.synthetic_seeds

    def _setup_realistic_mock_responses(self, monomers):
        """Set up realistic mock responses based on monomer structures."""
        n_monomers = len(monomers)

        # Mock calculator responses (monomer properties)
        # Generate realistic property values based on chemical intuition
        molecular_weights = []
        logp_values = []
        tpsa_values = []

        for smiles in monomers:
            # Estimate properties based on structure (simplified)
            if "c1ccc" in smiles:  # Aromatic
                mw = np.random.uniform(120, 180)
                logp = np.random.uniform(1.5, 3.5)
                tpsa = np.random.uniform(20, 60)
            elif "C(=O)O" in smiles:  # Carboxylic acid
                mw = np.random.uniform(60, 150)
                logp = np.random.uniform(-0.5, 2.0)
                tpsa = np.random.uniform(35, 80)
            elif "O" in smiles:  # Contains oxygen
                mw = np.random.uniform(50, 120)
                logp = np.random.uniform(-1.0, 1.5)
                tpsa = np.random.uniform(20, 100)
            else:  # Hydrocarbon
                mw = np.random.uniform(40, 100)
                logp = np.random.uniform(1.0, 4.0)
                tpsa = np.random.uniform(0, 30)

            molecular_weights.append(mw)
            logp_values.append(logp)
            tpsa_values.append(tpsa)

        mock_df = pd.DataFrame(
            {
                "molecular_weight": molecular_weights,
                "logp": logp_values,
                "tpsa": tpsa_values,
                "num_rotatable_bonds": np.random.randint(0, 8, n_monomers),
                "num_aromatic_rings": [
                    1 if "c1ccc" in smiles else 0 for smiles in monomers
                ],
            }
        )
        self.mock_calculator.pandas.return_value = mock_df

        # Mock polymer engine responses (polymer properties)
        # Generate realistic polymer properties
        glass_transition_temps = []
        youngs_moduli = []
        densities = []

        for smiles in monomers:
            # Estimate polymer properties based on monomer structure
            if "c1ccc" in smiles:  # Aromatic - higher Tg and modulus
                tg = np.random.uniform(350, 450)
                modulus = np.random.uniform(3, 8)
                density = np.random.uniform(1.1, 1.4)
            elif "C(=O)O" in smiles:  # Carboxylic acid - moderate properties
                tg = np.random.uniform(280, 380)
                modulus = np.random.uniform(2, 5)
                density = np.random.uniform(1.0, 1.3)
            elif "O" in smiles:  # Oxygen-containing - variable properties
                tg = np.random.uniform(250, 350)
                modulus = np.random.uniform(1, 4)
                density = np.random.uniform(0.9, 1.2)
            else:  # Hydrocarbon - lower properties
                tg = np.random.uniform(200, 300)
                modulus = np.random.uniform(0.5, 2)
                density = np.random.uniform(0.8, 1.1)

            glass_transition_temps.append(tg)
            youngs_moduli.append(modulus)
            densities.append(density)

        self.mock_polymer_engine.predict_polymer_properties.return_value = {
            "glass_transition_temp": np.array(glass_transition_temps),
            "youngs_modulus": np.array(youngs_moduli),
            "density": np.array(densities),
            "thermal_stability": np.random.uniform(300, 500, n_monomers),
        }

    def test_bio_derived_monomer_design(self):
        """Test design of bio-derived monomers with appropriate constraints."""
        # Set up bio-derived design scenario
        objectives = create_biodegradable_polymer_objectives()
        constraints = create_bio_derived_constraints()

        designer = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=objectives,
            constraint_config=constraints,
            use_gpu=False,
        )

        # Set up mocks
        self._setup_realistic_mock_responses(self.bio_derived_seeds)

        with patch(
            "src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE", True
        ), patch("src.chemi_accel.polymer.inverse_design.Chem") as mock_chem, patch(
            "src.chemi_accel.polymer.inverse_design.sascorer"
        ) as mock_sascorer:
            # Mock RDKit for bio-derived patterns
            mock_mol = Mock()
            mock_mol.GetAtoms.return_value = [Mock(GetSymbol=lambda: "C")] * 4 + [
                Mock(GetSymbol=lambda: "O")
            ] * 3
            mock_mol.GetNumAtoms.return_value = 7
            mock_mol.HasSubstructMatch.return_value = True  # Bio-derived patterns found
            mock_mol.GetSubstructMatches.return_value = [(0,), (1,)]

            mock_chem.MolFromSmiles.return_value = mock_mol
            mock_chem.MolFromSmarts.return_value = Mock()
            mock_chem.MolToSmiles.return_value = "CC(O)C(=O)O"
            mock_chem.Descriptors.MolWt.return_value = 90.0
            mock_chem.Descriptors.NumAromaticRings.return_value = (
                0  # Non-aromatic for bio-derived
            )
            mock_chem.RDKFingerprint.return_value = Mock()
            mock_chem.DataStructs.TanimotoSimilarity.return_value = 0.6

            mock_sascorer.calculateScore.return_value = 2.0  # Easy to synthesize

            # Run design
            result = designer.design_monomers(
                seed_monomers=self.bio_derived_seeds,
                n_iterations=3,
                population_size=8,
                generation_size=4,
            )

            # Validate bio-derived focus
            assert result.bio_derived_scores is not None
            assert len(result.bio_derived_scores) > 0

            # Check that bio-derived scores are generally high
            mean_bio_score = np.mean(result.bio_derived_scores)
            assert (
                mean_bio_score > 0.3
            ), f"Expected higher bio-derived scores, got {mean_bio_score}"

            # Check synthetic accessibility (should be reasonable for bio-derived)
            if result.synthetic_accessibility is not None:
                mean_sa = np.mean(result.synthetic_accessibility)
                assert (
                    mean_sa < 4.0
                ), f"Expected reasonable synthetic accessibility, got {mean_sa}"

            # Analyze objectives achievement
            analysis = designer.analyze_design_objectives(result)
            assert "bio_derived_mean" in analysis
            assert analysis["bio_derived_mean"] > 0.3

    def test_high_performance_monomer_design(self):
        """Test design of high-performance polymer monomers."""
        # Set up high-performance design scenario
        objectives = create_high_performance_polymer_objectives()
        constraints = create_high_performance_constraints()

        designer = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=objectives,
            constraint_config=constraints,
            use_gpu=False,
        )

        # Set up mocks for high-performance scenario
        self._setup_realistic_mock_responses(self.synthetic_seeds)

        with patch(
            "src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE", True
        ), patch("src.chemi_accel.polymer.inverse_design.Chem") as mock_chem, patch(
            "src.chemi_accel.polymer.inverse_design.sascorer"
        ) as mock_sascorer:
            # Mock RDKit for aromatic/high-performance structures
            mock_mol = Mock()
            mock_mol.GetAtoms.return_value = [Mock(GetSymbol=lambda: "C")] * 6 + [
                Mock(GetSymbol=lambda: "N")
            ] * 1
            mock_mol.GetNumAtoms.return_value = 7
            mock_mol.HasSubstructMatch.return_value = True
            mock_mol.GetSubstructMatches.return_value = [(0,), (1,), (2,)]

            mock_chem.MolFromSmiles.return_value = mock_mol
            mock_chem.MolFromSmarts.return_value = Mock()
            mock_chem.MolToSmiles.return_value = "c1ccc(C(=O)O)cc1"
            mock_chem.Descriptors.MolWt.return_value = 150.0
            mock_chem.Descriptors.NumAromaticRings.return_value = (
                1  # Aromatic for high-performance
            )
            mock_chem.RDKFingerprint.return_value = Mock()
            mock_chem.DataStructs.TanimotoSimilarity.return_value = 0.4

            mock_sascorer.calculateScore.return_value = (
                3.5  # Moderate synthetic complexity
            )

            # Run design
            result = designer.design_monomers(
                seed_monomers=self.synthetic_seeds,
                n_iterations=3,
                population_size=8,
                generation_size=4,
            )

            # Validate high-performance focus
            assert "glass_transition_temp" in result.polymer_properties
            assert "youngs_modulus" in result.polymer_properties

            # Check that polymer properties are in high-performance range
            tg_values = result.polymer_properties["glass_transition_temp"]
            modulus_values = result.polymer_properties["youngs_modulus"]

            # Should have some high-performance candidates
            high_tg_count = np.sum(tg_values > 350)
            high_modulus_count = np.sum(modulus_values > 3.0)

            assert high_tg_count > 0, "Expected some high Tg candidates"
            assert high_modulus_count > 0, "Expected some high modulus candidates"

            # Analyze objectives achievement
            analysis = designer.analyze_design_objectives(result)
            if (
                result.pareto_front_indices is not None
                and len(result.pareto_front_indices) > 0
            ):
                assert (
                    "glass_transition_temp_pareto_mean" in analysis
                    or "glass_transition_temp_pareto_range" in analysis
                )

    def test_pareto_front_diversity(self):
        """Test that Pareto front contains diverse solutions."""
        # Use mixed objectives to create trade-offs
        objectives = [
            PolymerDesignObjective(
                name="glass_transition_temp",
                target_value=400.0,
                minimize=False,
                weight=1.0,
            ),
            PolymerDesignObjective(
                name="synthetic_accessibility",
                target_value=2.0,
                minimize=True,
                weight=1.0,
            ),
            PolymerDesignObjective(
                name="bio_derived_score", target_value=0.8, minimize=False, weight=1.0
            ),
        ]

        designer = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=objectives,
            use_gpu=False,
        )

        # Set up mocks with diverse properties to create trade-offs
        self._setup_realistic_mock_responses(self.all_seeds)

        with patch(
            "src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE", True
        ), patch("src.chemi_accel.polymer.inverse_design.Chem") as mock_chem, patch(
            "src.chemi_accel.polymer.inverse_design.sascorer"
        ) as mock_sascorer:
            # Mock RDKit with diverse responses
            mock_mol = Mock()
            mock_mol.GetAtoms.return_value = [Mock(GetSymbol=lambda: "C")] * 5 + [
                Mock(GetSymbol=lambda: "O")
            ] * 2
            mock_mol.GetNumAtoms.return_value = 7
            mock_mol.HasSubstructMatch.return_value = True
            mock_mol.GetSubstructMatches.return_value = [(0,), (1,)]

            mock_chem.MolFromSmiles.return_value = mock_mol
            mock_chem.MolFromSmarts.return_value = Mock()
            mock_chem.MolToSmiles.return_value = "CC(O)C(=O)O"
            mock_chem.Descriptors.MolWt.return_value = 120.0
            mock_chem.Descriptors.NumAromaticRings.return_value = 0
            mock_chem.RDKFingerprint.return_value = Mock()
            mock_chem.DataStructs.TanimotoSimilarity.return_value = 0.5

            # Vary SA scores to create trade-offs
            sa_scores = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 2.2, 2.8, 3.2, 3.8]
            mock_sascorer.calculateScore.side_effect = sa_scores

            # Run design
            result = designer.design_monomers(
                seed_monomers=self.all_seeds,
                n_iterations=3,
                population_size=10,
                generation_size=5,
            )

            # Validate Pareto front diversity
            if (
                result.pareto_front_indices is not None
                and len(result.pareto_front_indices) > 1
            ):
                pareto_monomers = designer.get_pareto_front_monomers(result)

                # Check that Pareto front contains multiple solutions
                assert (
                    len(pareto_monomers) >= 2
                ), "Expected multiple Pareto optimal solutions"

                # Check diversity in objectives for Pareto front
                pareto_tg = result.polymer_properties["glass_transition_temp"][
                    result.pareto_front_indices
                ]
                pareto_sa = result.synthetic_accessibility[result.pareto_front_indices]
                pareto_bio = result.bio_derived_scores[result.pareto_front_indices]

                # Should have some diversity in objective values
                tg_range = np.max(pareto_tg) - np.min(pareto_tg)
                sa_range = np.max(pareto_sa) - np.min(pareto_sa)
                bio_range = np.max(pareto_bio) - np.min(pareto_bio)

                # At least one objective should show diversity
                assert (
                    tg_range > 10 or sa_range > 0.2 or bio_range > 0.1
                ), "Expected diversity in Pareto front objectives"

    def test_constraint_satisfaction(self):
        """Test that generated monomers satisfy specified constraints."""
        # Set up strict constraints
        strict_constraints = PolymerConstraintConfig(
            sa_threshold=3.0,
            bio_derived_preference=True,
            bio_derived_weight=2.0,
            mw_range=(80.0, 200.0),
            tg_range=(300.0, 400.0),
            functional_groups_required=["[OH]"],  # Require hydroxyl group
            forbidden_substructures=["[F]", "[Cl]"],  # No halogens
            polymerizability_threshold=0.6,
        )

        objectives = [
            PolymerDesignObjective(
                name="glass_transition_temp", target_value=350.0, minimize=False
            )
        ]

        designer = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=objectives,
            constraint_config=strict_constraints,
            use_gpu=False,
        )

        # Set up mocks
        self._setup_realistic_mock_responses(self.bio_derived_seeds)

        with patch(
            "src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE", True
        ), patch("src.chemi_accel.polymer.inverse_design.Chem") as mock_chem, patch(
            "src.chemi_accel.polymer.inverse_design.sascorer"
        ) as mock_sascorer:
            # Mock RDKit to satisfy constraints
            mock_mol = Mock()
            mock_mol.GetAtoms.return_value = [Mock(GetSymbol=lambda: "C")] * 4 + [
                Mock(GetSymbol=lambda: "O")
            ] * 2
            mock_mol.GetNumAtoms.return_value = 6
            mock_mol.HasSubstructMatch.side_effect = (
                lambda pattern: True
            )  # Has required groups
            mock_mol.GetSubstructMatches.return_value = [(0,)]

            mock_chem.MolFromSmiles.return_value = mock_mol
            mock_chem.MolFromSmarts.return_value = Mock()
            mock_chem.MolToSmiles.return_value = "CC(O)C(=O)O"
            mock_chem.Descriptors.MolWt.return_value = 120.0  # Within MW range
            mock_chem.Descriptors.NumAromaticRings.return_value = 0
            mock_chem.RDKFingerprint.return_value = Mock()
            mock_chem.DataStructs.TanimotoSimilarity.return_value = 0.6

            mock_sascorer.calculateScore.return_value = 2.5  # Within SA threshold

            # Run design
            result = designer.design_monomers(
                seed_monomers=self.bio_derived_seeds, n_iterations=2, population_size=6
            )

            # Validate constraint satisfaction
            constraints = result.constraint_violations

            # Check that constraint violations are reasonable
            if "mw_violation" in constraints:
                assert (
                    np.mean(constraints["mw_violation"]) < 5.0
                ), "MW constraints not well satisfied"

            if "sa_violation" in constraints:
                assert (
                    np.mean(constraints["sa_violation"]) < 2.0
                ), "SA constraints not well satisfied"

            if "tg_violation" in constraints:
                assert (
                    np.mean(constraints["tg_violation"]) < 50.0
                ), "Tg constraints not well satisfied"

            # Check bio-derived preference
            if result.bio_derived_scores is not None:
                mean_bio_score = np.mean(result.bio_derived_scores)
                assert (
                    mean_bio_score > 0.4
                ), f"Bio-derived preference not satisfied: {mean_bio_score}"

    def test_generation_convergence(self):
        """Test that the generation process converges appropriately."""
        objectives = create_biodegradable_polymer_objectives()

        designer = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=objectives,
            use_gpu=False,
        )

        # Set up mocks
        self._setup_realistic_mock_responses(self.bio_derived_seeds)

        with patch(
            "src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE", True
        ), patch("src.chemi_accel.polymer.inverse_design.Chem") as mock_chem, patch(
            "src.chemi_accel.polymer.inverse_design.sascorer"
        ) as mock_sascorer:
            # Mock RDKit
            mock_mol = Mock()
            mock_mol.GetAtoms.return_value = [Mock(GetSymbol=lambda: "C")] * 4 + [
                Mock(GetSymbol=lambda: "O")
            ] * 2
            mock_mol.GetNumAtoms.return_value = 6
            mock_mol.HasSubstructMatch.return_value = True
            mock_mol.GetSubstructMatches.return_value = [(0,)]

            mock_chem.MolFromSmiles.return_value = mock_mol
            mock_chem.MolFromSmarts.return_value = Mock()
            mock_chem.MolToSmiles.return_value = "CC(O)C(=O)O"
            mock_chem.Descriptors.MolWt.return_value = 100.0
            mock_chem.Descriptors.NumAromaticRings.return_value = 0
            mock_chem.RDKFingerprint.return_value = Mock()

            # Simulate decreasing diversity over iterations
            similarities = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.78, 0.79, 0.795, 0.798]
            mock_chem.DataStructs.TanimotoSimilarity.side_effect = similarities

            mock_sascorer.calculateScore.return_value = 2.0

            # Run design with more iterations to test convergence
            result = designer.design_monomers(
                seed_monomers=self.bio_derived_seeds, n_iterations=10, population_size=8
            )

            # Check that design history was recorded
            assert len(designer.design_history) > 0

            # Check that diversity generally decreased (indicating convergence)
            if len(designer.design_history) > 3:
                early_diversity = designer.design_history[1]["diversity"]
                late_diversity = designer.design_history[-1]["diversity"]

                # Diversity should generally decrease or stabilize
                assert (
                    late_diversity <= early_diversity + 0.1
                ), f"Expected convergence, but diversity increased from {early_diversity} to {late_diversity}"

    def test_bio_derived_scoring_validation(self):
        """Validate bio-derived scoring against known bio-derived structures."""
        scorer = BioDerivedScorer()

        # Test with known bio-derived structures
        bio_derived_monomers = [
            "CC(O)C(=O)O",  # Lactic acid - definitely bio-derived
            "CCCCCCCC(=O)O",  # Octanoic acid - fatty acid, bio-derived
            "c1ccc(O)c(OC)c1",  # Guaiacol - lignin-derived
            "OCC(O)CO",  # Glycerol - bio-derived
        ]

        # Test with synthetic structures
        synthetic_monomers = [
            "c1ccc(C#N)cc1",  # Benzonitrile - synthetic
            "CC(C)(C)C(=O)O",  # tert-Butyl acid - synthetic
            "c1ccc(C(F)(F)F)cc1",  # Trifluoromethylbenzene - synthetic
        ]

        with patch(
            "src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE", True
        ), patch("src.chemi_accel.polymer.inverse_design.Chem") as mock_chem:

            def mock_mol_from_smiles(smiles):
                mock_mol = Mock()
                # Simulate bio-derived pattern matching
                if any(
                    pattern in smiles
                    for pattern in ["C(O)", "CCCCCC", "c(O)c(OC)", "OCC(O)"]
                ):
                    mock_mol.HasSubstructMatch.return_value = True
                    mock_mol.GetAtoms.return_value = [
                        Mock(GetSymbol=lambda: "C")
                    ] * 4 + [Mock(GetSymbol=lambda: "O")] * 2
                    mock_mol.GetNumAtoms.return_value = 6
                else:
                    mock_mol.HasSubstructMatch.return_value = False
                    mock_mol.GetAtoms.return_value = [Mock(GetSymbol=lambda: "C")] * 6
                    mock_mol.GetNumAtoms.return_value = 6

                return mock_mol

            mock_chem.MolFromSmiles.side_effect = mock_mol_from_smiles
            mock_chem.MolFromSmarts.return_value = Mock()
            mock_chem.Descriptors.MolWt.return_value = 120.0
            mock_chem.Descriptors.NumAromaticRings.return_value = 0

            # Score bio-derived monomers
            bio_scores = scorer.score_bio_derived_potential(bio_derived_monomers)

            # Score synthetic monomers
            synthetic_scores = scorer.score_bio_derived_potential(synthetic_monomers)

            # Bio-derived should generally score higher
            mean_bio_score = np.mean(bio_scores)
            mean_synthetic_score = np.mean(synthetic_scores)

            assert (
                mean_bio_score > mean_synthetic_score
            ), f"Bio-derived monomers should score higher: {mean_bio_score} vs {mean_synthetic_score}"

            # Bio-derived scores should be reasonable
            assert mean_bio_score > 0.2, f"Bio-derived scores too low: {mean_bio_score}"
            assert (
                mean_bio_score <= 1.0
            ), f"Bio-derived scores too high: {mean_bio_score}"


if __name__ == "__main__":
    pytest.main([__file__])

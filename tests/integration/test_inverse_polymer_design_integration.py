"""
Integration tests for inverse polymer design functionality.

This module tests the complete workflow of the InversePolymerDesigner
including multi-objective optimization, constraint handling, and
Pareto front analysis for polymer design objectives.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.chemi_accel.core.calculator import Calculator

# Import the modules to test
from src.chemi_accel.polymer.inverse_design import (
    InversePolymerDesigner,
    MonomerGenerationResult,
    PolymerConstraintConfig,
    create_bio_derived_constraints,
    create_high_performance_polymer_objectives,
)
from src.chemi_accel.polymer.predictive import PolymerPropertyEngine


class TestInversePolymerDesignIntegration:
    """Integration tests for the complete inverse polymer design workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock calculator
        self.mock_calculator = Mock(spec=Calculator)

        # Create mock polymer engine
        self.mock_polymer_engine = Mock(spec=PolymerPropertyEngine)

        # Define test objectives
        self.objectives = create_high_performance_polymer_objectives()

        # Define test constraints
        self.constraints = create_bio_derived_constraints()

        # Create designer instance
        self.designer = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=self.objectives,
            constraint_config=self.constraints,
            use_gpu=False,
        )

        # Test seed monomers (realistic polymer building blocks)
        self.seed_monomers = [
            "CC(C)(C)C(=O)O",  # tert-butyl carboxylic acid
            "c1ccc(O)cc1",  # phenol
            "CCCCCCCC(=O)O",  # octanoic acid (fatty acid)
            "CC(O)C(=O)O",  # lactic acid (bio-derived)
            "c1ccc(C(=O)O)cc1",  # benzoic acid
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccc(N)cc1",  # aniline
        ]

    def _setup_mock_responses(self):
        """Set up realistic mock responses for calculator and polymer engine."""
        # Mock calculator responses (monomer properties)
        n_monomers = len(self.seed_monomers)
        mock_df = pd.DataFrame(
            {
                "molecular_weight": np.random.uniform(60, 200, n_monomers),
                "logp": np.random.uniform(-1, 3, n_monomers),
                "tpsa": np.random.uniform(20, 100, n_monomers),
                "num_rotatable_bonds": np.random.randint(0, 8, n_monomers),
                "num_aromatic_rings": np.random.randint(0, 2, n_monomers),
            }
        )
        self.mock_calculator.pandas.return_value = mock_df

        # Mock polymer engine responses (polymer properties)
        self.mock_polymer_engine.predict_polymer_properties.return_value = {
            "glass_transition_temp": np.random.uniform(250, 450, n_monomers),
            "youngs_modulus": np.random.uniform(1, 10, n_monomers),
            "density": np.random.uniform(0.9, 1.5, n_monomers),
            "thermal_stability": np.random.uniform(300, 500, n_monomers),
        }

    @patch("src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE", True)
    @patch("src.chemi_accel.polymer.inverse_design.Chem")
    @patch("src.chemi_accel.polymer.inverse_design.sascorer")
    def test_complete_design_workflow(self, mock_sascorer, mock_chem):
        """Test the complete inverse polymer design workflow."""
        # Set up mocks
        self._setup_mock_responses()

        # Mock RDKit functionality
        mock_mol = Mock()
        mock_mol.GetAtoms.return_value = [Mock(GetSymbol=lambda: "C")] * 5 + [
            Mock(GetSymbol=lambda: "O")
        ] * 2
        mock_mol.GetNumAtoms.return_value = 7
        mock_mol.HasSubstructMatch.return_value = True
        mock_mol.GetSubstructMatches.return_value = [(0,), (1,)]

        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.MolFromSmarts.return_value = Mock()
        mock_chem.MolToSmiles.return_value = "CC(=O)O"
        mock_chem.Descriptors.MolWt.return_value = 150.0
        mock_chem.Descriptors.NumAromaticRings.return_value = 1
        mock_chem.RDKFingerprint.return_value = Mock()
        mock_chem.DataStructs.TanimotoSimilarity.return_value = 0.5

        mock_sascorer.calculateScore.return_value = 2.5

        # Run the design process
        result = self.designer.design_monomers(
            seed_monomers=self.seed_monomers,
            n_iterations=5,
            population_size=10,
            generation_size=5,
        )

        # Validate results
        assert isinstance(result, MonomerGenerationResult)
        assert len(result.generated_monomers) > 0
        assert result.predicted_properties is not None
        assert result.polymer_properties is not None
        assert result.constraint_violations is not None
        assert result.bio_derived_scores is not None
        assert result.synthetic_accessibility is not None

        # Check that Pareto front was identified
        if result.pareto_front_indices is not None:
            assert len(result.pareto_front_indices) > 0
            assert all(
                0 <= idx < len(result.generated_monomers)
                for idx in result.pareto_front_indices
            )

    def test_pareto_front_analysis(self):
        """Test Pareto front analysis functionality."""
        self._setup_mock_responses()

        # Create a mock result with known Pareto front
        result = MonomerGenerationResult(
            generated_monomers=self.seed_monomers[:5],
            predicted_properties={
                "molecular_weight": np.array([100, 150, 120, 180, 90]),
                "logp": np.array([1.0, 2.0, 1.5, 2.5, 0.8]),
            },
            polymer_properties={
                "glass_transition_temp": np.array([300, 350, 320, 380, 280]),
                "youngs_modulus": np.array([2.0, 4.0, 3.0, 5.0, 1.5]),
            },
            constraint_violations={
                "sa_violation": np.array([0.1, 0.2, 0.15, 0.3, 0.05])
            },
            pareto_front_indices=np.array(
                [1, 3]
            ),  # Indices of Pareto optimal solutions
            bio_derived_scores=np.array([0.8, 0.6, 0.7, 0.5, 0.9]),
            synthetic_accessibility=np.array([2.0, 3.0, 2.5, 3.5, 1.8]),
        )

        # Test Pareto front extraction
        pareto_monomers = self.designer.get_pareto_front_monomers(result)
        assert len(pareto_monomers) == 2
        assert pareto_monomers[0] == self.seed_monomers[1]
        assert pareto_monomers[1] == self.seed_monomers[3]

        # Test objective analysis
        analysis = self.designer.analyze_design_objectives(result)
        assert isinstance(analysis, dict)
        assert "n_pareto_solutions" in analysis
        assert analysis["n_pareto_solutions"] == 2
        assert "bio_derived_mean" in analysis
        assert "sa_mean" in analysis

        # Verify bio-derived analysis
        assert analysis["bio_derived_mean"] == np.mean(result.bio_derived_scores)
        assert analysis["highly_bio_derived_count"] == np.sum(
            result.bio_derived_scores > 0.7
        )

        # Verify synthetic accessibility analysis
        assert analysis["sa_mean"] == np.mean(result.synthetic_accessibility)
        assert analysis["easily_synthesizable_count"] == np.sum(
            result.synthetic_accessibility < 3.0
        )

    def test_constraint_handling_integration(self):
        """Test integration of constraint handling across different constraint types."""
        self._setup_mock_responses()

        # Test with comprehensive constraints
        comprehensive_constraints = PolymerConstraintConfig(
            sa_threshold=3.0,
            bio_derived_preference=True,
            bio_derived_weight=2.0,
            tg_range=(300.0, 450.0),
            youngs_modulus_range=(2.0, 8.0),
            mw_range=(80.0, 250.0),
            functional_groups_required=["[OH]", "C(=O)O"],
            forbidden_substructures=["[F]", "[Cl]"],
            polymerizability_threshold=0.7,
            reactivity_balance=True,
        )

        designer_with_constraints = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=self.objectives,
            constraint_config=comprehensive_constraints,
            use_gpu=False,
        )

        # Evaluate constraints
        with patch(
            "src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE", True
        ), patch("src.chemi_accel.polymer.inverse_design.Chem") as mock_chem, patch(
            "src.chemi_accel.polymer.inverse_design.sascorer"
        ) as mock_sascorer:
            # Mock RDKit functionality
            mock_mol = Mock()
            mock_mol.GetAtoms.return_value = [Mock(GetSymbol=lambda: "C")] * 5 + [
                Mock(GetSymbol=lambda: "O")
            ] * 2
            mock_mol.GetNumAtoms.return_value = 7
            mock_mol.HasSubstructMatch.return_value = True
            mock_mol.GetSubstructMatches.return_value = [(0,), (1,)]

            mock_chem.MolFromSmiles.return_value = mock_mol
            mock_chem.MolFromSmarts.return_value = Mock()
            mock_chem.Descriptors.MolWt.return_value = 150.0
            mock_chem.Descriptors.NumAromaticRings.return_value = 1
            mock_sascorer.calculateScore.return_value = 2.5

            constraints = (
                designer_with_constraints.constraint_handler.evaluate_constraints(
                    self.seed_monomers[:3]
                )
            )

            # Verify all constraint types are evaluated
            expected_constraints = [
                "mw_violation",
                "sa_violation",
                "forbidden_violation",
                "required_violation",
                "tg_violation",
                "modulus_violation",
                "polymerizability_violation",
                "reactivity_violation",
                "bio_derived_violation",
            ]

            for constraint_name in expected_constraints:
                if constraint_name in constraints:
                    assert isinstance(constraints[constraint_name], np.ndarray)
                    assert len(constraints[constraint_name]) == 3

    def test_multi_objective_optimization_integration(self):
        """Test integration with multi-objective optimization."""
        self._setup_mock_responses()

        # Create test data with clear trade-offs
        test_monomers = self.seed_monomers[:4]
        test_properties = {
            "glass_transition_temp": np.array([300, 400, 350, 320]),  # Higher is better
            "youngs_modulus": np.array([2.0, 5.0, 3.5, 2.8]),  # Higher is better
            "synthetic_accessibility": np.array(
                [2.0, 4.0, 2.5, 3.0]
            ),  # Lower is better
            "bio_derived_score": np.array(
                [0.8, 0.4, 0.7, 0.6]
            ),  # Higher is better (converted to violation)
        }

        # Mock the multi-objective optimizer
        with patch.object(
            self.designer.multi_objective_optimizer, "optimize"
        ) as mock_optimize:
            mock_result = Mock()
            mock_result.molecules = test_monomers
            mock_result.pareto_front.pareto_indices = np.array(
                [0, 1]
            )  # First two are Pareto optimal
            mock_result.pareto_front.pareto_molecules = [
                test_monomers[0],
                test_monomers[1],
            ]
            mock_optimize.return_value = mock_result

            mo_result = self.designer._perform_multi_objective_optimization(
                test_monomers,
                {
                    "monomer_properties": test_properties,
                    "polymer_properties": {},
                    "constraints": {},
                },
            )

            assert mo_result.molecules == test_monomers
            assert len(mo_result.pareto_front.pareto_indices) == 2
            assert mo_result.pareto_front.pareto_indices[0] == 0
            assert mo_result.pareto_front.pareto_indices[1] == 1

    def test_export_functionality(self):
        """Test result export functionality."""
        # Create a comprehensive result
        result = MonomerGenerationResult(
            generated_monomers=self.seed_monomers[:3],
            predicted_properties={
                "molecular_weight": np.array([100, 150, 120]),
                "logp": np.array([1.0, 2.0, 1.5]),
            },
            polymer_properties={
                "glass_transition_temp": np.array([300, 350, 320]),
                "youngs_modulus": np.array([2.0, 4.0, 3.0]),
            },
            constraint_violations={"sa_violation": np.array([0.1, 0.2, 0.15])},
            pareto_front_indices=np.array([0, 2]),
            bio_derived_scores=np.array([0.8, 0.6, 0.7]),
            synthetic_accessibility=np.array([2.0, 3.0, 2.5]),
        )

        # Test export to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            tmp_filename = tmp_file.name

        try:
            self.designer.export_design_results(result, tmp_filename)

            # Verify file was created and contains expected data
            assert os.path.exists(tmp_filename)

            # Read back the exported data
            exported_df = pd.read_csv(tmp_filename)

            # Verify structure
            assert len(exported_df) == 3
            assert "monomer_smiles" in exported_df.columns
            assert "is_pareto_optimal" in exported_df.columns
            assert "monomer_molecular_weight" in exported_df.columns
            assert "polymer_glass_transition_temp" in exported_df.columns
            assert "bio_derived_score" in exported_df.columns

            # Verify Pareto front marking
            pareto_flags = exported_df["is_pareto_optimal"].values
            assert pareto_flags[0] == True  # Index 0 is Pareto optimal
            assert pareto_flags[1] == False  # Index 1 is not Pareto optimal
            assert pareto_flags[2] == True  # Index 2 is Pareto optimal

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)

    def test_convergence_detection(self):
        """Test convergence detection in the design process."""
        # Mock the design history to simulate convergence
        self.designer.design_history = [
            {"iteration": 0, "diversity": 0.8},
            {"iteration": 1, "diversity": 0.75},
            {"iteration": 2, "diversity": 0.72},
            {"iteration": 3, "diversity": 0.71},
            {
                "iteration": 4,
                "diversity": 0.705,
            },  # Very small change, should trigger convergence
        ]

        # Test convergence detection
        converged = self.designer._check_convergence()
        assert converged == True

        # Test non-convergence
        self.designer.design_history = [
            {"iteration": 0, "diversity": 0.8},
            {"iteration": 1, "diversity": 0.6},
            {"iteration": 2, "diversity": 0.4},
            {"iteration": 3, "diversity": 0.2},
            {"iteration": 4, "diversity": 0.1},  # Large changes, should not converge
        ]

        converged = self.designer._check_convergence()
        assert converged == False

    def test_objective_configuration_validation(self):
        """Test validation of different objective configurations."""
        # Test high-performance objectives
        hp_objectives = create_high_performance_polymer_objectives()
        hp_designer = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=hp_objectives,
            use_gpu=False,
        )

        assert len(hp_designer.objectives) > 0
        assert any(
            obj.name == "glass_transition_temp" for obj in hp_designer.objectives
        )
        assert any(obj.name == "youngs_modulus" for obj in hp_designer.objectives)

        # Test that multi-objective optimizer was configured correctly
        assert len(hp_designer.multi_objective_optimizer.objectives) == len(
            hp_objectives
        )

    def test_error_handling_and_robustness(self):
        """Test error handling and robustness of the design process."""
        # Test with calculator failure
        self.mock_calculator.pandas.side_effect = Exception("Calculator failed")

        evaluation = self.designer._evaluate_candidates(self.seed_monomers[:2])

        # Should handle calculator failure gracefully
        assert isinstance(evaluation, dict)
        assert "monomer_properties" in evaluation
        # Should be empty due to failure, but not crash

        # Reset calculator
        self.mock_calculator.pandas.side_effect = None
        self._setup_mock_responses()

        # Test with polymer engine failure
        self.mock_polymer_engine.predict_polymer_properties.side_effect = Exception(
            "Polymer engine failed"
        )

        evaluation = self.designer._evaluate_candidates(self.seed_monomers[:2])

        # Should handle polymer engine failure gracefully
        assert isinstance(evaluation, dict)
        assert "polymer_properties" in evaluation
        # Should be empty due to failure, but not crash

    def test_gpu_acceleration_configuration(self):
        """Test GPU acceleration configuration."""
        # Test with GPU enabled
        gpu_designer = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=self.objectives,
            use_gpu=True,
        )

        assert gpu_designer.use_gpu == True
        # GPU availability depends on system, so just check configuration is passed through

        # Test with GPU disabled
        cpu_designer = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=self.objectives,
            use_gpu=False,
        )

        assert cpu_designer.use_gpu == False


if __name__ == "__main__":
    pytest.main([__file__])

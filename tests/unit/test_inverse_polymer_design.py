"""
Unit tests for inverse polymer design functionality.

This module tests the InversePolymerDesigner and related components
for novel monomer generation with multi-objective optimization.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from src.chemi_accel.polymer.inverse_design import (
    InversePolymerDesigner,
    PolymerConstraintConfig,
    PolymerDesignObjective,
    MonomerGenerationResult,
    BioDerivedScorer,
    PolymerConstraintHandler,
    MonomerGenerator,
    create_high_performance_polymer_objectives,
    create_biodegradable_polymer_objectives,
    create_bio_derived_constraints,
    create_high_performance_constraints
)
from src.chemi_accel.core.calculator import Calculator
from src.chemi_accel.polymer.predictive import PolymerPropertyEngine


class TestBioDerivedScorer:
    """Test bio-derived scoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = BioDerivedScorer()
        self.test_monomers = [
            'CC(C)(C)C(=O)O',  # Simple carboxylic acid
            'c1ccc(O)cc1',     # Phenol (lignin-derived pattern)
            'CCCCCCCC(=O)O',   # Fatty acid
            'CCc1ccccc1',      # Ethylbenzene (synthetic)
            'invalid_smiles'    # Invalid SMILES
        ]
    
    def test_initialization(self):
        """Test BioDerivedScorer initialization."""
        assert self.scorer is not None
        assert isinstance(self.scorer.bio_derived_patterns, dict)
        assert len(self.scorer.bio_derived_patterns) > 0
    
    @patch('src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE', True)
    @patch('src.chemi_accel.polymer.inverse_design.Chem')
    def test_score_bio_derived_potential_with_rdkit(self, mock_chem):
        """Test bio-derived scoring with RDKit available."""
        # Mock RDKit functionality
        mock_mol = Mock()
        mock_mol.GetAtoms.return_value = [Mock(GetSymbol=lambda: 'C')] * 5 + [Mock(GetSymbol=lambda: 'O')] * 2
        mock_mol.GetNumAtoms.return_value = 7
        mock_mol.HasSubstructMatch.return_value = True
        
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.MolFromSmarts.return_value = Mock()
        mock_chem.Descriptors.MolWt.return_value = 150.0
        mock_chem.Descriptors.NumAromaticRings.return_value = 1
        
        scores = self.scorer.score_bio_derived_potential(self.test_monomers[:3])
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert all(0.0 <= score <= 1.0 for score in scores)
    
    @patch('src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE', False)
    def test_score_bio_derived_potential_without_rdkit(self):
        """Test bio-derived scoring without RDKit."""
        scores = self.scorer.score_bio_derived_potential(self.test_monomers)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(self.test_monomers)
        assert all(score == 0.0 for score in scores)


class TestPolymerConstraintHandler:
    """Test polymer constraint handling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_calculator = Mock(spec=Calculator)
        self.mock_polymer_engine = Mock(spec=PolymerPropertyEngine)
        self.config = PolymerConstraintConfig(
            sa_threshold=3.0,
            bio_derived_preference=True,
            mw_range=(100.0, 300.0),
            tg_range=(250.0, 400.0)
        )
        self.handler = PolymerConstraintHandler(
            self.config, self.mock_calculator, self.mock_polymer_engine
        )
        self.test_monomers = ['CC(=O)O', 'c1ccc(O)cc1', 'CCO']
    
    def test_initialization(self):
        """Test PolymerConstraintHandler initialization."""
        assert self.handler.config == self.config
        assert self.handler.calculator == self.mock_calculator
        assert self.handler.polymer_engine == self.mock_polymer_engine
        assert isinstance(self.handler.bio_derived_scorer, BioDerivedScorer)
    
    @patch('src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE', True)
    @patch('src.chemi_accel.polymer.inverse_design.Chem')
    @patch('src.chemi_accel.polymer.inverse_design.sascorer')
    def test_evaluate_molecular_constraints(self, mock_sascorer, mock_chem):
        """Test molecular constraint evaluation."""
        # Mock RDKit functionality
        mock_mol = Mock()
        mock_mol.HasSubstructMatch.return_value = False
        
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.MolFromSmarts.return_value = Mock()
        mock_chem.Descriptors.MolWt.return_value = 150.0
        mock_sascorer.calculateScore.return_value = 2.5
        
        constraints = self.handler._evaluate_molecular_constraints(self.test_monomers)
        
        assert isinstance(constraints, dict)
        assert 'mw_violation' in constraints
        assert 'sa_violation' in constraints
        assert 'forbidden_violation' in constraints
        assert 'required_violation' in constraints
        
        for key, values in constraints.items():
            assert isinstance(values, np.ndarray)
            assert len(values) == len(self.test_monomers)
    
    def test_evaluate_polymer_property_constraints(self):
        """Test polymer property constraint evaluation."""
        # Mock polymer property predictions
        self.mock_polymer_engine.predict_polymer_properties.return_value = {
            'glass_transition_temp': np.array([300.0, 350.0, 280.0]),
            'youngs_modulus': np.array([2.0, 4.0, 1.5])
        }
        
        constraints = self.handler._evaluate_polymer_property_constraints(self.test_monomers)
        
        assert isinstance(constraints, dict)
        assert 'tg_violation' in constraints
        assert isinstance(constraints['tg_violation'], np.ndarray)
        assert len(constraints['tg_violation']) == len(self.test_monomers)
    
    def test_evaluate_constraints_integration(self):
        """Test full constraint evaluation integration."""
        # Mock all dependencies
        with patch.object(self.handler, '_evaluate_molecular_constraints') as mock_mol, \
             patch.object(self.handler, '_evaluate_polymer_property_constraints') as mock_poly, \
             patch.object(self.handler, '_evaluate_polymerization_constraints') as mock_polym, \
             patch.object(self.handler, '_evaluate_bio_derived_constraints') as mock_bio:
            
            mock_mol.return_value = {'mw_violation': np.array([0.0, 1.0, 0.5])}
            mock_poly.return_value = {'tg_violation': np.array([0.2, 0.0, 0.8])}
            mock_polym.return_value = {'polymerizability_violation': np.array([0.1, 0.3, 0.0])}
            mock_bio.return_value = {'bio_derived_violation': np.array([0.5, 0.2, 0.7])}
            
            constraints = self.handler.evaluate_constraints(self.test_monomers)
            
            assert isinstance(constraints, dict)
            assert len(constraints) == 4
            assert all(isinstance(v, np.ndarray) for v in constraints.values())


class TestMonomerGenerator:
    """Test monomer generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_calculator = Mock(spec=Calculator)
        self.generator = MonomerGenerator(self.mock_calculator, use_gpu=False)
        self.seed_monomers = ['CC(=O)O', 'CCO', 'c1ccccc1']
    
    def test_initialization(self):
        """Test MonomerGenerator initialization."""
        assert self.generator.calculator == self.mock_calculator
        assert isinstance(self.generator.generation_history, list)
        assert len(self.generator.generation_history) == 0
    
    @patch('src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE', True)
    @patch('src.chemi_accel.polymer.inverse_design.Chem')
    def test_generate_monomers_with_rdkit(self, mock_chem):
        """Test monomer generation with RDKit available."""
        # Mock RDKit functionality
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.MolToSmiles.return_value = 'CC(=O)O'
        mock_chem.RDKFingerprint.return_value = Mock()
        mock_chem.DataStructs.TanimotoSimilarity.return_value = 0.5
        
        generated = self.generator.generate_monomers(
            self.seed_monomers, n_generations=5, population_size=10
        )
        
        assert isinstance(generated, list)
        assert len(generated) <= 10
        assert len(self.generator.generation_history) == 5
    
    @patch('src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE', False)
    def test_generate_monomers_without_rdkit(self):
        """Test monomer generation without RDKit."""
        generated = self.generator.generate_monomers(
            self.seed_monomers, n_generations=3, population_size=5
        )
        
        assert isinstance(generated, list)
        assert len(generated) <= 5
        # Should return seed monomers when RDKit is not available
        assert all(mol in self.seed_monomers for mol in generated)
    
    def test_calculate_diversity(self):
        """Test diversity calculation."""
        with patch('src.chemi_accel.polymer.inverse_design.RDKIT_AVAILABLE', False):
            diversity = self.generator._calculate_diversity(self.seed_monomers)
            assert diversity == 0.0  # Should return 0 without RDKit
        
        # Test with empty population
        diversity = self.generator._calculate_diversity([])
        assert diversity == 0.0


class TestInversePolymerDesigner:
    """Test the main InversePolymerDesigner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_calculator = Mock(spec=Calculator)
        self.mock_polymer_engine = Mock(spec=PolymerPropertyEngine)
        
        self.objectives = [
            PolymerDesignObjective(
                name='glass_transition_temp',
                target_value=350.0,
                minimize=False,
                weight=2.0
            ),
            PolymerDesignObjective(
                name='synthetic_accessibility',
                target_value=2.0,
                minimize=True,
                weight=1.0
            )
        ]
        
        self.designer = InversePolymerDesigner(
            calculator=self.mock_calculator,
            polymer_engine=self.mock_polymer_engine,
            objectives=self.objectives,
            use_gpu=False
        )
        
        self.seed_monomers = ['CC(=O)O', 'CCO', 'c1ccc(O)cc1']
    
    def test_initialization(self):
        """Test InversePolymerDesigner initialization."""
        assert self.designer.calculator == self.mock_calculator
        assert self.designer.polymer_engine == self.mock_polymer_engine
        assert self.designer.objectives == self.objectives
        assert isinstance(self.designer.constraint_config, PolymerConstraintConfig)
        assert isinstance(self.designer.constraint_handler, PolymerConstraintHandler)
        assert isinstance(self.designer.monomer_generator, MonomerGenerator)
    
    def test_evaluate_candidates(self):
        """Test candidate evaluation."""
        # Mock calculator response
        mock_df = pd.DataFrame({
            'molecular_weight': [100.0, 150.0, 120.0],
            'logp': [1.0, 2.0, 1.5]
        })
        self.mock_calculator.pandas.return_value = mock_df
        
        # Mock polymer engine response
        self.mock_polymer_engine.predict_polymer_properties.return_value = {
            'glass_transition_temp': np.array([300.0, 350.0, 320.0])
        }
        
        # Mock constraint evaluation
        with patch.object(self.designer.constraint_handler, 'evaluate_constraints') as mock_constraints:
            mock_constraints.return_value = {
                'sa_violation': np.array([0.1, 0.2, 0.15])
            }
            
            evaluation = self.designer._evaluate_candidates(self.seed_monomers)
            
            assert isinstance(evaluation, dict)
            assert 'monomer_properties' in evaluation
            assert 'polymer_properties' in evaluation
            assert 'constraints' in evaluation
    
    def test_design_monomers_integration(self):
        """Test the main design_monomers method."""
        # Mock all dependencies
        with patch.object(self.designer.monomer_generator, 'generate_monomers') as mock_gen, \
             patch.object(self.designer, '_evaluate_candidates') as mock_eval, \
             patch.object(self.designer, '_select_best_candidates') as mock_select, \
             patch.object(self.designer, '_perform_multi_objective_optimization') as mock_mo, \
             patch.object(self.designer.constraint_handler.bio_derived_scorer, 'score_bio_derived_potential') as mock_bio, \
             patch.object(self.designer, '_calculate_synthetic_accessibility') as mock_sa:
            
            # Set up mocks
            mock_gen.return_value = self.seed_monomers
            mock_eval.return_value = {
                'monomer_properties': {'molecular_weight': np.array([100.0, 150.0, 120.0])},
                'polymer_properties': {'glass_transition_temp': np.array([300.0, 350.0, 320.0])},
                'constraints': {'sa_violation': np.array([0.1, 0.2, 0.15])}
            }
            mock_select.return_value = self.seed_monomers
            
            mock_mo_result = Mock()
            mock_mo_result.pareto_front.pareto_indices = np.array([0, 2])
            mock_mo.return_value = mock_mo_result
            
            mock_bio.return_value = np.array([0.8, 0.6, 0.7])
            mock_sa.return_value = np.array([2.0, 3.0, 2.5])
            
            # Run design
            result = self.designer.design_monomers(
                seed_monomers=self.seed_monomers,
                n_iterations=3,
                population_size=5
            )
            
            # Verify result
            assert isinstance(result, MonomerGenerationResult)
            assert result.generated_monomers == self.seed_monomers
            assert result.pareto_front_indices is not None
            assert len(result.pareto_front_indices) == 2
    
    def test_get_pareto_front_monomers(self):
        """Test Pareto front monomer extraction."""
        result = MonomerGenerationResult(
            generated_monomers=self.seed_monomers,
            predicted_properties={},
            polymer_properties={},
            constraint_violations={},
            pareto_front_indices=np.array([0, 2])
        )
        
        pareto_monomers = self.designer.get_pareto_front_monomers(result)
        
        assert len(pareto_monomers) == 2
        assert pareto_monomers[0] == self.seed_monomers[0]
        assert pareto_monomers[1] == self.seed_monomers[2]
    
    def test_analyze_design_objectives(self):
        """Test design objective analysis."""
        result = MonomerGenerationResult(
            generated_monomers=self.seed_monomers,
            predicted_properties={'molecular_weight': np.array([100.0, 150.0, 120.0])},
            polymer_properties={'glass_transition_temp': np.array([300.0, 350.0, 320.0])},
            constraint_violations={},
            pareto_front_indices=np.array([0, 2]),
            bio_derived_scores=np.array([0.8, 0.6, 0.7]),
            synthetic_accessibility=np.array([2.0, 3.0, 2.5])
        )
        
        analysis = self.designer.analyze_design_objectives(result)
        
        assert isinstance(analysis, dict)
        assert 'n_pareto_solutions' in analysis
        assert analysis['n_pareto_solutions'] == 2
        assert 'bio_derived_mean' in analysis
        assert 'sa_mean' in analysis


class TestUtilityFunctions:
    """Test utility functions for polymer design."""
    
    def test_create_high_performance_polymer_objectives(self):
        """Test high-performance polymer objective creation."""
        objectives = create_high_performance_polymer_objectives()
        
        assert isinstance(objectives, list)
        assert len(objectives) > 0
        assert all(isinstance(obj, PolymerDesignObjective) for obj in objectives)
        
        # Check for expected objectives
        obj_names = [obj.name for obj in objectives]
        assert 'glass_transition_temp' in obj_names
        assert 'youngs_modulus' in obj_names
    
    def test_create_biodegradable_polymer_objectives(self):
        """Test biodegradable polymer objective creation."""
        objectives = create_biodegradable_polymer_objectives()
        
        assert isinstance(objectives, list)
        assert len(objectives) > 0
        assert all(isinstance(obj, PolymerDesignObjective) for obj in objectives)
        
        # Check for bio-derived focus
        obj_names = [obj.name for obj in objectives]
        assert 'bio_derived_score' in obj_names
    
    def test_create_bio_derived_constraints(self):
        """Test bio-derived constraint creation."""
        constraints = create_bio_derived_constraints()
        
        assert isinstance(constraints, PolymerConstraintConfig)
        assert constraints.bio_derived_preference is True
        assert constraints.bio_derived_weight > 1.0
        assert constraints.functional_groups_required is not None
        assert constraints.forbidden_substructures is not None
    
    def test_create_high_performance_constraints(self):
        """Test high-performance constraint creation."""
        constraints = create_high_performance_constraints()
        
        assert isinstance(constraints, PolymerConstraintConfig)
        assert constraints.tg_range is not None
        assert constraints.youngs_modulus_range is not None
        assert constraints.tg_range[0] > 300.0  # High glass transition
        assert constraints.thermal_stability_min is not None


class TestDataStructures:
    """Test data structure classes."""
    
    def test_polymer_constraint_config(self):
        """Test PolymerConstraintConfig dataclass."""
        config = PolymerConstraintConfig(
            sa_threshold=3.5,
            bio_derived_preference=True,
            mw_range=(100.0, 250.0)
        )
        
        assert config.sa_threshold == 3.5
        assert config.bio_derived_preference is True
        assert config.mw_range == (100.0, 250.0)
        assert config.bio_derived_weight == 2.0  # Default value
    
    def test_polymer_design_objective(self):
        """Test PolymerDesignObjective dataclass."""
        objective = PolymerDesignObjective(
            name='glass_transition_temp',
            target_value=350.0,
            minimize=False,
            weight=2.0,
            property_type='polymer',
            units='K'
        )
        
        assert objective.name == 'glass_transition_temp'
        assert objective.target_value == 350.0
        assert objective.minimize is False
        assert objective.weight == 2.0
        assert objective.property_type == 'polymer'
        assert objective.units == 'K'
    
    def test_monomer_generation_result(self):
        """Test MonomerGenerationResult dataclass."""
        result = MonomerGenerationResult(
            generated_monomers=['CC(=O)O', 'CCO'],
            predicted_properties={'mw': np.array([60.0, 46.0])},
            polymer_properties={'tg': np.array([300.0, 250.0])},
            constraint_violations={'sa': np.array([2.0, 1.5])},
            pareto_front_indices=np.array([0])
        )
        
        assert len(result.generated_monomers) == 2
        assert 'mw' in result.predicted_properties
        assert 'tg' in result.polymer_properties
        assert 'sa' in result.constraint_violations
        assert len(result.pareto_front_indices) == 1


if __name__ == '__main__':
    pytest.main([__file__])
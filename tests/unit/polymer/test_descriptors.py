"""
Unit tests for the Polymer Descriptor Calculator.

This module tests the specialized polymer descriptor calculation algorithms
that are used for property prediction.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Import the modules to test
from src.chemi_accel.polymer.predictive.descriptors import PolymerDescriptorCalculator
from src.chemi_accel.core.calculator import Calculator


class TestPolymerDescriptorCalculator:
    """Test cases for PolymerDescriptorCalculator."""
    
    @pytest.fixture
    def mock_calculator(self):
        """Create a mock calculator for testing."""
        calculator = Mock(spec=Calculator)
        return calculator
    
    @pytest.fixture
    def descriptor_calc(self, mock_calculator):
        """Create a PolymerDescriptorCalculator instance."""
        return PolymerDescriptorCalculator(mock_calculator)
    
    @pytest.fixture
    def sample_molecules(self):
        """Sample polymer molecules for testing."""
        return pd.Series([
            'C=C',  # ethylene
            'C=CC',  # propylene
            'C=Cc1ccccc1',  # styrene
            'C=CC(=O)O',  # acrylic acid
            'C=CC(=O)OC',  # methyl acrylate
        ])
    
    def test_initialization(self, mock_calculator):
        """Test PolymerDescriptorCalculator initialization."""
        calc = PolymerDescriptorCalculator(mock_calculator)
        assert calc.calculator == mock_calculator
    
    def test_calculate_polymer_descriptors_structure(self, descriptor_calc, sample_molecules):
        """Test the structure of calculated polymer descriptors."""
        with patch('src.chemi_accel.utils.validation.validate_smiles_series'):
            result = descriptor_calc.calculate_polymer_descriptors(sample_molecules)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_molecules)
            
            # Check that all descriptor categories are present
            expected_columns = [
                # Basic descriptors
                'molecular_weight', 'heavy_atom_count', 'logp', 'tpsa',
                # Polymer-specific descriptors
                'vinyl_groups', 'backbone_flexibility', 'polymerization_sites',
                # Thermal descriptors
                'tg_predictor_1', 'aromatic_content', 'crystallinity_index',
                # Mechanical descriptors
                'modulus_predictor_1', 'backbone_stiffness', 'density_predictor'
            ]
            
            for col in expected_columns:
                assert col in result.columns, f"Missing column: {col}"
    
    def test_calculate_polymer_descriptors_list_input(self, descriptor_calc):
        """Test descriptor calculation with list input."""
        molecules_list = ['C=C', 'C=CC']
        
        with patch('src.chemi_accel.utils.validation.validate_smiles_series'):
            result = descriptor_calc.calculate_polymer_descriptors(molecules_list)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
    
    @patch('src.chemi_accel.polymer.predictive.descriptors.RDKIT_AVAILABLE', True)
    def test_calculate_rdkit_descriptors(self, descriptor_calc):
        """Test RDKit-based descriptor calculation."""
        with patch('rdkit.Chem.MolFromSmiles') as mock_mol_from_smiles:
            # Mock successful molecule creation
            mock_mol = Mock()
            mock_mol.GetNumHeavyAtoms.return_value = 2
            mock_mol.GetNumAtoms.return_value = 4
            mock_mol.GetNumBonds.return_value = 1
            mock_mol_from_smiles.return_value = mock_mol
            
            with patch('rdkit.Chem.Descriptors.MolWt', return_value=28.0):
                with patch('rdkit.Chem.Crippen.MolLogP', return_value=1.0):
                    with patch('rdkit.Chem.rdMolDescriptors.CalcTPSA', return_value=0.0):
                        descriptors = descriptor_calc._calculate_rdkit_descriptors('C=C')
                        
                        assert 'molecular_weight' in descriptors
                        assert 'heavy_atom_count' in descriptors
                        assert 'logp' in descriptors
                        assert 'tpsa' in descriptors
                        assert descriptors['molecular_weight'] == 28.0
                        assert descriptors['heavy_atom_count'] == 2
    
    @patch('src.chemi_accel.polymer.predictive.descriptors.RDKIT_AVAILABLE', True)
    def test_calculate_rdkit_descriptors_invalid_smiles(self, descriptor_calc):
        """Test RDKit descriptor calculation with invalid SMILES."""
        with patch('rdkit.Chem.MolFromSmiles', return_value=None):
            descriptors = descriptor_calc._calculate_rdkit_descriptors('INVALID')
            
            # Should return default descriptors
            assert 'molecular_weight' in descriptors
            assert descriptors['molecular_weight'] == 0.0
    
    @patch('src.chemi_accel.polymer.predictive.descriptors.RDKIT_AVAILABLE', False)
    def test_calculate_simple_descriptors(self, descriptor_calc):
        """Test simple descriptor calculation without RDKit."""
        descriptors = descriptor_calc._calculate_simple_descriptors('C=CC(=O)O')
        
        assert 'smiles_length' in descriptors
        assert 'carbon_count' in descriptors
        assert 'oxygen_count' in descriptors
        assert 'estimated_molecular_weight' in descriptors
        
        assert descriptors['carbon_count'] == 3  # Three carbons in C=CC(=O)O
        assert descriptors['oxygen_count'] == 2  # Two oxygens
    
    def test_count_vinyl_groups(self, descriptor_calc):
        """Test vinyl group counting."""
        assert descriptor_calc._count_vinyl_groups('C=C') == 1
        assert descriptor_calc._count_vinyl_groups('C=CC=C') == 2
        assert descriptor_calc._count_vinyl_groups('CC') == 0
    
    def test_count_epoxy_groups(self, descriptor_calc):
        """Test epoxy group counting."""
        assert descriptor_calc._count_epoxy_groups('C1OC1') == 1
        assert descriptor_calc._count_epoxy_groups('C1OC1C1OC1') == 2
        assert descriptor_calc._count_epoxy_groups('CC') == 0
    
    def test_count_hydroxyl_groups(self, descriptor_calc):
        """Test hydroxyl group counting (simplified)."""
        # Note: This is a simplified implementation
        assert descriptor_calc._count_hydroxyl_groups('CO') == 1
        assert descriptor_calc._count_hydroxyl_groups('COCO') == 2
        assert descriptor_calc._count_hydroxyl_groups('CC') == 0
    
    def test_count_carboxyl_groups(self, descriptor_calc):
        """Test carboxyl group counting."""
        assert descriptor_calc._count_carboxyl_groups('C(=O)O') == 1
        assert descriptor_calc._count_carboxyl_groups('CC(=O)O') == 1
        assert descriptor_calc._count_carboxyl_groups('CC') == 0
    
    def test_count_amine_groups(self, descriptor_calc):
        """Test amine group counting (simplified)."""
        assert descriptor_calc._count_amine_groups('CN') == 1
        assert descriptor_calc._count_amine_groups('CNCN') == 2
        assert descriptor_calc._count_amine_groups('CC') == 0
    
    def test_estimate_backbone_flexibility(self, descriptor_calc):
        """Test backbone flexibility estimation."""
        # More single bonds should increase flexibility
        flex1 = descriptor_calc._estimate_backbone_flexibility('C-C-C')
        flex2 = descriptor_calc._estimate_backbone_flexibility('C=C=C')
        
        assert flex1 > flex2  # Single bonds more flexible than double bonds
        assert 0 <= flex1 <= 1
        assert 0 <= flex2 <= 1
    
    def test_estimate_side_chain_length(self, descriptor_calc):
        """Test side chain length estimation."""
        # No branches
        length1 = descriptor_calc._estimate_side_chain_length('CCC')
        assert length1 == 0.0
        
        # With branches
        length2 = descriptor_calc._estimate_side_chain_length('C(C)C')
        assert length2 > 0
    
    def test_estimate_branching_factor(self, descriptor_calc):
        """Test branching factor estimation."""
        # Linear molecule
        branch1 = descriptor_calc._estimate_branching_factor('CCC')
        
        # Branched molecule
        branch2 = descriptor_calc._estimate_branching_factor('C(C)C')
        
        assert branch2 > branch1
        assert branch1 >= 0
        assert branch2 >= 0
    
    def test_count_chiral_centers(self, descriptor_calc):
        """Test chiral center counting."""
        assert descriptor_calc._count_chiral_centers('C@HCl') == 1
        assert descriptor_calc._count_chiral_centers('C@@HCl') == 1
        assert descriptor_calc._count_chiral_centers('CCl') == 0
    
    def test_count_polymerization_sites(self, descriptor_calc):
        """Test polymerization site counting."""
        sites1 = descriptor_calc._count_polymerization_sites('C=C')  # vinyl
        sites2 = descriptor_calc._count_polymerization_sites('C1OC1')  # epoxy
        sites3 = descriptor_calc._count_polymerization_sites('C=CC1OC1')  # both
        
        assert sites1 == 1
        assert sites2 == 1
        assert sites3 == 2
    
    def test_thermal_descriptors(self, descriptor_calc):
        """Test thermal property descriptors."""
        # Test aromatic content
        aromatic1 = descriptor_calc._calculate_aromatic_content('cccccc')  # benzene
        aromatic2 = descriptor_calc._calculate_aromatic_content('CCCCCC')  # hexane
        
        assert aromatic1 > aromatic2
        assert 0 <= aromatic1 <= 1
        assert 0 <= aromatic2 <= 1
        
        # Test thermal stability index
        stability = descriptor_calc._calculate_thermal_stability_index('c1ccccc1')
        assert stability >= 0
    
    def test_mechanical_descriptors(self, descriptor_calc):
        """Test mechanical property descriptors."""
        # Test backbone stiffness
        stiffness1 = descriptor_calc._calculate_backbone_stiffness('c1ccccc1')  # aromatic
        stiffness2 = descriptor_calc._calculate_backbone_stiffness('CCCCCC')    # aliphatic
        
        assert stiffness1 > stiffness2  # Aromatic should be stiffer
        assert 0 <= stiffness1 <= 1
        assert 0 <= stiffness2 <= 1
        
        # Test density predictor
        density = descriptor_calc._calculate_density_predictor('c1ccccc1')
        assert density > 0
    
    def test_tg_predictors(self, descriptor_calc):
        """Test glass transition temperature predictors."""
        tg1 = descriptor_calc._calculate_tg_predictor_1('c1ccccc1')  # aromatic
        tg2 = descriptor_calc._calculate_tg_predictor_1('CCCCCC')    # aliphatic
        
        # Aromatic should have higher Tg predictor
        assert tg1 > tg2
        
        # Test other Tg predictors
        tg_pred2 = descriptor_calc._calculate_tg_predictor_2('c1ccccc1')
        tg_pred3 = descriptor_calc._calculate_tg_predictor_3('c1ccc(O)cc1')  # phenol
        
        assert isinstance(tg_pred2, (int, float))
        assert isinstance(tg_pred3, (int, float))
    
    def test_modulus_predictors(self, descriptor_calc):
        """Test Young's modulus predictors."""
        mod1 = descriptor_calc._calculate_modulus_predictor_1('c1ccccc1')  # aromatic
        mod2 = descriptor_calc._calculate_modulus_predictor_1('CCCCCC')    # aliphatic
        
        # Aromatic should have higher modulus predictor
        assert mod1 > mod2
        
        # Test other modulus predictors
        mod_pred2 = descriptor_calc._calculate_modulus_predictor_2('C=C')
        mod_pred3 = descriptor_calc._calculate_modulus_predictor_3('c1ccccc1')
        
        assert mod_pred2 >= 0
        assert mod_pred3 >= 0
    
    def test_crystallinity_and_packing(self, descriptor_calc):
        """Test crystallinity and packing descriptors."""
        # Regular structure should have higher crystallinity
        cryst1 = descriptor_calc._calculate_crystallinity_index('CCCCCC')
        cryst2 = descriptor_calc._calculate_crystallinity_index('C(C)(C)C(C)C')
        
        assert cryst1 > cryst2  # More regular structure
        
        # Packing efficiency
        pack1 = descriptor_calc._estimate_packing_efficiency('CCCCCC')
        pack2 = descriptor_calc._estimate_packing_efficiency('C(C)(C)C(C)C')
        
        assert pack1 > pack2  # Less branched = better packing
    
    def test_descriptor_value_ranges(self, descriptor_calc, sample_molecules):
        """Test that descriptor values are in reasonable ranges."""
        with patch('src.chemi_accel.utils.validation.validate_smiles_series'):
            result = descriptor_calc.calculate_polymer_descriptors(sample_molecules)
            
            # Check that all values are finite
            assert result.isna().sum().sum() == 0, "Found NaN values in descriptors"
            assert np.isfinite(result.values).all(), "Found infinite values in descriptors"
            
            # Check some specific ranges
            if 'backbone_flexibility' in result.columns:
                assert (result['backbone_flexibility'] >= 0).all()
                assert (result['backbone_flexibility'] <= 1).all()
            
            if 'aromatic_content' in result.columns:
                assert (result['aromatic_content'] >= 0).all()
                assert (result['aromatic_content'] <= 1).all()
            
            if 'density_predictor' in result.columns:
                assert (result['density_predictor'] > 0).all()
                assert (result['density_predictor'] < 5).all()  # Reasonable density range


if __name__ == '__main__':
    pytest.main([__file__])
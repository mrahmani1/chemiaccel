"""
Unit tests for copolymer and blend property modeling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.chemi_accel.polymer.copolymer import (
    CopolymerType, MixingRule, PhaseState,
    MonomerComponent, CopolymerSystem, CopolymerProperties,
    MicrophaseInfo, BlendMiscibility,
    LinearMixingRule, LogarithmicMixingRule, GordonTaylorMixingRule,
    FoxMixingRule, FloryHugginsMixingRule,
    CopolymerPropertyPredictor, MicrophaseAnalyzer,
    BlendMiscibilityAnalyzer, CopolymerBlendModeler
)
from src.chemi_accel.core.exceptions import ChemiAccelError


class TestMonomerComponent:
    """Test MonomerComponent class."""
    
    def test_monomer_component_creation(self):
        """Test creating a monomer component."""
        component = MonomerComponent(
            smiles="C=C",
            name="Ethylene",
            mole_fraction=0.5,
            weight_fraction=0.5,
            molecular_weight=28.0,
            glass_transition_temp=200.0
        )
        
        assert component.smiles == "C=C"
        assert component.name == "Ethylene"
        assert component.mole_fraction == 0.5
        assert component.weight_fraction == 0.5
        assert component.molecular_weight == 28.0
        assert component.glass_transition_temp == 200.0


class TestCopolymerSystem:
    """Test CopolymerSystem class."""
    
    def test_valid_copolymer_system(self):
        """Test creating a valid copolymer system."""
        components = [
            MonomerComponent("C=C", "Ethylene", 0.5, 0.4, 28.0),
            MonomerComponent("C=C(C)C(=O)OC", "MMA", 0.5, 0.6, 100.0)
        ]
        
        system = CopolymerSystem(
            components=components,
            copolymer_type=CopolymerType.RANDOM
        )
        
        assert len(system.components) == 2
        assert system.copolymer_type == CopolymerType.RANDOM
    
    def test_invalid_single_component(self):
        """Test that single component works for homopolymers but not for other types."""
        components = [
            MonomerComponent("C=C", "Ethylene", 1.0, 1.0, 28.0)
        ]
        
        # Should work for random type (homopolymer)
        system = CopolymerSystem(components=components, copolymer_type=CopolymerType.RANDOM)
        assert len(system.components) == 1
        
        # Should fail for block type
        with pytest.raises(ChemiAccelError, match="requires at least 2 components"):
            CopolymerSystem(components=components, copolymer_type=CopolymerType.BLOCK)
    
    def test_invalid_mole_fractions(self):
        """Test that invalid mole fractions raise error."""
        components = [
            MonomerComponent("C=C", "Ethylene", 0.3, 0.4, 28.0),
            MonomerComponent("C=C(C)C(=O)OC", "MMA", 0.3, 0.6, 100.0)  # Sum = 0.6
        ]
        
        with pytest.raises(ChemiAccelError, match="Mole fractions must sum to 1.0"):
            CopolymerSystem(components=components, copolymer_type=CopolymerType.RANDOM)


class TestMixingRules:
    """Test mixing rule implementations."""
    
    def setup_method(self):
        """Set up test components."""
        self.components = [
            MonomerComponent(
                "C=C", "Ethylene", 0.5, 0.4, 28.0,
                glass_transition_temp=200.0,
                properties={'density': 0.9}
            ),
            MonomerComponent(
                "C=C(C)C(=O)OC", "MMA", 0.5, 0.6, 100.0,
                glass_transition_temp=378.0,
                properties={'density': 1.2}
            )
        ]
    
    def test_linear_mixing_rule(self):
        """Test linear mixing rule."""
        rule = LinearMixingRule()
        
        # Test with properties dict
        density = rule.calculate_property(self.components, 'density')
        expected = 0.4 * 0.9 + 0.6 * 1.2
        assert abs(density - expected) < 1e-6
        
        # Test with direct attribute
        tg = rule.calculate_property(self.components, 'glass_transition_temp')
        expected = 0.4 * 200.0 + 0.6 * 378.0
        assert abs(tg - expected) < 1e-6
    
    def test_logarithmic_mixing_rule(self):
        """Test logarithmic mixing rule."""
        rule = LogarithmicMixingRule()
        
        density = rule.calculate_property(self.components, 'density')
        expected = np.exp(0.4 * np.log(0.9) + 0.6 * np.log(1.2))
        assert abs(density - expected) < 1e-6
    
    def test_gordon_taylor_mixing_rule(self):
        """Test Gordon-Taylor mixing rule."""
        rule = GordonTaylorMixingRule(k_parameter=1.5)
        
        tg = rule.calculate_property(self.components, 'glass_transition_temp')
        expected = (0.4 * 200.0 + 1.5 * 0.6 * 378.0) / (0.4 + 1.5 * 0.6)
        assert abs(tg - expected) < 1e-6
        
        # Test with wrong property
        with pytest.raises(ValueError, match="only applies to glass transition"):
            rule.calculate_property(self.components, 'density')
    
    def test_fox_mixing_rule(self):
        """Test Fox mixing rule."""
        rule = FoxMixingRule()
        
        tg = rule.calculate_property(self.components, 'glass_transition_temp')
        expected = 1.0 / (0.4 / 200.0 + 0.6 / 378.0)
        assert abs(tg - expected) < 1e-6
        
        # Test with wrong property
        with pytest.raises(ValueError, match="only applies to glass transition"):
            rule.calculate_property(self.components, 'density')
    
    def test_flory_huggins_mixing_rule(self):
        """Test Flory-Huggins mixing rule."""
        rule = FloryHugginsMixingRule(chi_parameter=0.3)
        
        density = rule.calculate_property(self.components, 'density')
        linear_value = 0.4 * 0.9 + 0.6 * 1.2
        correction = 1.0 + 0.3 * 0.4 * 0.6
        expected = linear_value * correction
        assert abs(density - expected) < 1e-6


class TestCopolymerPropertyPredictor:
    """Test CopolymerPropertyPredictor class."""
    
    def setup_method(self):
        """Set up test system."""
        self.components = [
            MonomerComponent(
                "C=C", "Ethylene", 0.5, 0.4, 28.0,
                glass_transition_temp=200.0,
                density=0.9,
                properties={'youngs_modulus': 1000.0}
            ),
            MonomerComponent(
                "C=C(C)C(=O)OC", "MMA", 0.5, 0.6, 100.0,
                glass_transition_temp=378.0,
                density=1.2,
                properties={'youngs_modulus': 3000.0}
            )
        ]
        
        self.system = CopolymerSystem(
            components=self.components,
            copolymer_type=CopolymerType.RANDOM
        )
        
        self.predictor = CopolymerPropertyPredictor()
    
    def test_predict_properties(self):
        """Test property prediction."""
        properties = self.predictor.predict_properties(self.system)
        
        assert isinstance(properties, CopolymerProperties)
        assert properties.glass_transition_temp is not None
        assert properties.density is not None
        assert properties.youngs_modulus is not None
    
    def test_architecture_correction(self):
        """Test architecture-specific corrections."""
        # Test block copolymer
        block_system = CopolymerSystem(
            components=self.components,
            copolymer_type=CopolymerType.BLOCK
        )
        
        block_props = self.predictor.predict_properties(block_system)
        random_props = self.predictor.predict_properties(self.system)
        
        # Block copolymers should have enhanced mechanical properties
        assert block_props.youngs_modulus > random_props.youngs_modulus
    
    def test_custom_mixing_rules(self):
        """Test custom mixing rules."""
        custom_rules = {
            'glass_transition_temp': MixingRule.FOX,
            'density': MixingRule.LOGARITHMIC
        }
        
        properties = self.predictor.predict_properties(self.system, custom_rules)
        assert properties.glass_transition_temp is not None
        assert properties.density is not None


class TestMicrophaseAnalyzer:
    """Test MicrophaseAnalyzer class."""
    
    def setup_method(self):
        """Set up test system."""
        self.components = [
            MonomerComponent(
                "CCCCCCCC", "Octane", 0.5, 0.5, 114.0,
                solubility_parameter=15.5
            ),
            MonomerComponent(
                "c1ccccc1", "Benzene", 0.5, 0.5, 78.0,
                solubility_parameter=18.8
            )
        ]
        
        self.block_system = CopolymerSystem(
            components=self.components,
            copolymer_type=CopolymerType.BLOCK,
            sequence_length=1000
        )
        
        self.analyzer = MicrophaseAnalyzer()
    
    def test_microphase_analysis_block_copolymer(self):
        """Test microphase analysis for block copolymer."""
        info = self.analyzer.analyze_microphase_separation(self.block_system)
        
        assert isinstance(info, MicrophaseInfo)
        assert info.chi_parameter is not None
        assert info.segregation_strength is not None
        assert info.morphology is not None
    
    def test_microphase_analysis_random_copolymer(self):
        """Test microphase analysis for random copolymer."""
        random_system = CopolymerSystem(
            components=self.components,
            copolymer_type=CopolymerType.RANDOM
        )
        
        info = self.analyzer.analyze_microphase_separation(random_system)
        assert not info.is_microphase_separated
    
    @patch('src.chemi_accel.polymer.copolymer.RDKIT_AVAILABLE', False)
    def test_microphase_analysis_no_rdkit(self):
        """Test microphase analysis without RDKit."""
        info = self.analyzer.analyze_microphase_separation(self.block_system)
        assert info.chi_parameter == 0.1  # Default value


class TestBlendMiscibilityAnalyzer:
    """Test BlendMiscibilityAnalyzer class."""
    
    def setup_method(self):
        """Set up test systems."""
        self.polymer1 = CopolymerSystem(
            components=[
                MonomerComponent(
                    "C=C", "Ethylene", 1.0, 1.0, 28.0,
                    solubility_parameter=16.0
                )
            ],
            copolymer_type=CopolymerType.RANDOM
        )
        
        self.polymer2 = CopolymerSystem(
            components=[
                MonomerComponent(
                    "C=C(C)C(=O)OC", "MMA", 1.0, 1.0, 100.0,
                    solubility_parameter=19.0
                )
            ],
            copolymer_type=CopolymerType.RANDOM
        )
        
        self.analyzer = BlendMiscibilityAnalyzer()
    
    def test_miscibility_analysis(self):
        """Test blend miscibility analysis."""
        miscibility = self.analyzer.analyze_miscibility(
            self.polymer1, self.polymer2, temperature=298.15, blend_ratio=0.5
        )
        
        assert isinstance(miscibility, BlendMiscibility)
        assert miscibility.interaction_parameter is not None
        assert miscibility.phase_state in PhaseState
        assert miscibility.phase_diagram_data is not None
    
    def test_temperature_effects(self):
        """Test temperature effects on miscibility."""
        low_temp = self.analyzer.analyze_miscibility(
            self.polymer1, self.polymer2, temperature=200.0
        )
        high_temp = self.analyzer.analyze_miscibility(
            self.polymer1, self.polymer2, temperature=400.0
        )
        
        # Higher temperature should generally improve miscibility
        assert low_temp.interaction_parameter >= high_temp.interaction_parameter
    
    @patch('src.chemi_accel.polymer.copolymer.RDKIT_AVAILABLE', False)
    def test_miscibility_analysis_no_rdkit(self):
        """Test miscibility analysis without RDKit."""
        # Create polymers without solubility parameters to test fallback
        polymer1_no_sp = CopolymerSystem(
            components=[
                MonomerComponent("C=C", "Ethylene", 1.0, 1.0, 28.0)
            ],
            copolymer_type=CopolymerType.RANDOM
        )
        
        polymer2_no_sp = CopolymerSystem(
            components=[
                MonomerComponent("C=C(C)C(=O)OC", "MMA", 1.0, 1.0, 100.0)
            ],
            copolymer_type=CopolymerType.RANDOM
        )
        
        miscibility = self.analyzer.analyze_miscibility(
            polymer1_no_sp, polymer2_no_sp
        )
        assert miscibility.interaction_parameter == 0.1  # Default value


class TestCopolymerBlendModeler:
    """Test CopolymerBlendModeler class."""
    
    def setup_method(self):
        """Set up test data."""
        self.monomer1_data = {
            'smiles': 'C=C',
            'name': 'Ethylene',
            'mole_fraction': 0.5,
            'weight_fraction': 0.4,
            'molecular_weight': 28.0,
            'glass_transition_temp': 200.0
        }
        
        self.monomer2_data = {
            'smiles': 'C=C(C)C(=O)OC',
            'name': 'MMA',
            'mole_fraction': 0.5,
            'weight_fraction': 0.6,
            'molecular_weight': 100.0,
            'glass_transition_temp': 378.0
        }
        
        self.modeler = CopolymerBlendModeler()
    
    def test_create_copolymer_system(self):
        """Test creating copolymer system."""
        monomer_data = [self.monomer1_data, self.monomer2_data]
        
        system = self.modeler.create_copolymer_system(
            monomer_data, CopolymerType.RANDOM
        )
        
        assert isinstance(system, CopolymerSystem)
        assert len(system.components) == 2
        assert system.copolymer_type == CopolymerType.RANDOM
    
    @patch('src.chemi_accel.polymer.copolymer.RDKIT_AVAILABLE', False)
    def test_create_copolymer_system_no_rdkit(self):
        """Test creating copolymer system without RDKit."""
        monomer_data = [
            {**self.monomer1_data, 'molecular_weight': None},
            {**self.monomer2_data, 'molecular_weight': None}
        ]
        
        system = self.modeler.create_copolymer_system(
            monomer_data, CopolymerType.RANDOM
        )
        
        # Should use default molecular weight
        assert all(comp.molecular_weight == 100.0 for comp in system.components)
    
    def test_predict_copolymer_properties(self):
        """Test copolymer property prediction."""
        monomer_data = [self.monomer1_data, self.monomer2_data]
        system = self.modeler.create_copolymer_system(
            monomer_data, CopolymerType.RANDOM
        )
        
        results = self.modeler.predict_copolymer_properties(system)
        
        assert 'copolymer_system' in results
        assert 'predicted_properties' in results
        assert isinstance(results['predicted_properties'], CopolymerProperties)
    
    def test_predict_block_copolymer_properties(self):
        """Test block copolymer property prediction with microphase analysis."""
        monomer_data = [self.monomer1_data, self.monomer2_data]
        system = self.modeler.create_copolymer_system(
            monomer_data, CopolymerType.BLOCK
        )
        
        results = self.modeler.predict_copolymer_properties(system)
        
        assert 'microphase_info' in results
        assert isinstance(results['microphase_info'], MicrophaseInfo)
    
    def test_analyze_polymer_blend(self):
        """Test polymer blend analysis."""
        # Create homopolymer data with proper fractions
        polymer1_data = [{
            **self.monomer1_data,
            'mole_fraction': 1.0,
            'weight_fraction': 1.0
        }]
        polymer2_data = [{
            **self.monomer2_data,
            'mole_fraction': 1.0,
            'weight_fraction': 1.0
        }]
        
        results = self.modeler.analyze_polymer_blend(
            polymer1_data, polymer2_data, blend_ratio=0.3
        )
        
        assert 'polymer1' in results
        assert 'polymer2' in results
        assert 'blend_system' in results
        assert 'miscibility_info' in results
        assert 'blend_properties' in results
        assert results['blend_ratio'] == 0.3
    
    def test_generate_composition_study(self):
        """Test composition study generation."""
        results = self.modeler.generate_composition_study(
            self.monomer1_data, self.monomer2_data,
            composition_range=(0.2, 0.8), num_points=5
        )
        
        assert 'compositions' in results
        assert 'properties' in results
        assert 'microphase_info' in results
        assert len(results['compositions']) == 5
        assert len(results['properties']) == 5
        assert len(results['microphase_info']) == 5
    
    @patch('src.chemi_accel.polymer.copolymer.PLOTTING_AVAILABLE', False)
    def test_visualize_phase_diagram_no_plotting(self):
        """Test phase diagram visualization without plotting libraries."""
        blend_results = {
            'miscibility_info': BlendMiscibility(
                is_miscible=True,
                phase_state=PhaseState.MISCIBLE,
                interaction_parameter=0.3,
                phase_diagram_data={'compositions': [0, 1], 'temperatures': [200, 400]}
            )
        }
        
        fig = self.modeler.visualize_phase_diagram(blend_results)
        assert fig is None


class TestIntegration:
    """Integration tests for copolymer modeling."""
    
    def test_complete_copolymer_workflow(self):
        """Test complete copolymer analysis workflow."""
        modeler = CopolymerBlendModeler()
        
        # Define monomers
        styrene_data = {
            'smiles': 'C=Cc1ccccc1',
            'name': 'Styrene',
            'mole_fraction': 0.7,
            'weight_fraction': 0.75,
            'molecular_weight': 104.0,
            'glass_transition_temp': 373.0,
            'solubility_parameter': 18.5
        }
        
        butadiene_data = {
            'smiles': 'C=CC=C',
            'name': 'Butadiene',
            'mole_fraction': 0.3,
            'weight_fraction': 0.25,
            'molecular_weight': 54.0,
            'glass_transition_temp': 165.0,
            'solubility_parameter': 17.0
        }
        
        # Create and analyze copolymer system
        monomer_data = [styrene_data, butadiene_data]
        system = modeler.create_copolymer_system(monomer_data, CopolymerType.BLOCK)
        
        results = modeler.predict_copolymer_properties(system)
        
        # Verify results
        assert results['predicted_properties'].glass_transition_temp is not None
        assert results['predicted_properties'].density is not None
        assert results['microphase_info'].is_microphase_separated is not None
        
        # Test composition study
        comp_study = modeler.generate_composition_study(
            styrene_data, butadiene_data, num_points=3
        )
        
        assert len(comp_study['compositions']) == 3
        assert all(isinstance(prop, CopolymerProperties) for prop in comp_study['properties'])
    
    def test_complete_blend_workflow(self):
        """Test complete polymer blend analysis workflow."""
        modeler = CopolymerBlendModeler()
        
        # Define polymers
        pe_data = [{
            'smiles': 'CC',
            'name': 'Polyethylene',
            'mole_fraction': 1.0,
            'weight_fraction': 1.0,
            'molecular_weight': 28.0,
            'glass_transition_temp': 200.0,
            'solubility_parameter': 16.0
        }]
        
        ps_data = [{
            'smiles': 'CCc1ccccc1',
            'name': 'Polystyrene',
            'mole_fraction': 1.0,
            'weight_fraction': 1.0,
            'molecular_weight': 104.0,
            'glass_transition_temp': 373.0,
            'solubility_parameter': 18.5
        }]
        
        # Analyze blend
        blend_results = modeler.analyze_polymer_blend(
            pe_data, ps_data, blend_ratio=0.5, temperature=298.15
        )
        
        # Verify results
        assert blend_results['miscibility_info'].is_miscible is not None
        assert blend_results['miscibility_info'].phase_state in PhaseState
        assert blend_results['blend_properties'].glass_transition_temp is not None
        
        # Test phase diagram data
        phase_data = blend_results['miscibility_info'].phase_diagram_data
        assert 'compositions' in phase_data
        assert 'temperatures' in phase_data
        assert 'miscibility_map' in phase_data


if __name__ == '__main__':
    pytest.main([__file__])
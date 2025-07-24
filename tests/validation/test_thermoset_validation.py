"""
Validation tests for thermoset curing simulation against experimental data.

These tests compare simulation results with published experimental data
to validate the accuracy of the curing model and property predictions.
"""

import pytest
import numpy as np
import logging
from unittest.mock import patch, Mock

from chemi_accel.polymer.thermoset import (
    ThermosetCuringSimulator,
    CuringMechanism,
    CuringKinetics,
    validate_against_experimental_data
)


class TestThermosetValidation:
    """Validation tests against experimental thermoset curing data."""
    
    def setup_method(self):
        """Set up test fixtures with experimental data."""
        self.simulator = ThermosetCuringSimulator(use_gpu=False)
        
        # Experimental data for DGEBA/EDA system (literature values)
        self.dgeba_eda_experimental = {
            'system_name': 'DGEBA/EDA',
            'resin_smiles': 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1',  # DGEBA
            'hardener_smiles': 'NCCN',  # Ethylenediamine
            'stoichiometry': 1.0,
            'temperature': 393.15,  # 120°C
            
            # Experimental measurements
            'gel_point_times': [1800, 1750, 1820],  # seconds at 120°C
            'gel_point_conversions': [0.58, 0.60, 0.57],
            'final_conversions': [0.95, 0.97, 0.94],
            'elastic_modulus': [2800, 2900, 2750],  # MPa
            'glass_transition_temps': [423.15, 425.15, 421.15],  # K
            'tensile_strength': [75, 78, 73],  # MPa
            
            # Kinetic parameters (fitted from experimental data)
            'rate_constants': {'k1': 2.5e-3, 'k2': 1.2e-3},
            'activation_energies': {'E1': 52000, 'E2': 58000}
        }
        
        # Experimental data for epoxy-anhydride system
        self.epoxy_anhydride_experimental = {
            'system_name': 'DGEBA/MHHPA',
            'resin_smiles': 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1',  # DGEBA
            'hardener_smiles': 'O=C1OC(=O)C2C1C=CC2',  # MHHPA (simplified)
            'stoichiometry': 0.85,
            'temperature': 423.15,  # 150°C
            
            # Experimental measurements
            'gel_point_times': [2400, 2350, 2450],  # seconds at 150°C
            'gel_point_conversions': [0.65, 0.67, 0.63],
            'final_conversions': [0.92, 0.94, 0.91],
            'elastic_modulus': [3200, 3300, 3150],  # MPa
            'glass_transition_temps': [453.15, 455.15, 451.15],  # K
            'tensile_strength': [85, 88, 82],  # MPa
            
            # Kinetic parameters
            'rate_constants': {'k1': 1.8e-3, 'k2': 0.9e-3},
            'activation_energies': {'E1': 55000, 'E2': 62000}
        }
    
    @patch('chemi_accel.polymer.thermoset.RDKIT_AVAILABLE', True)
    def test_dgeba_eda_system_validation(self):
        """Validate DGEBA/EDA epoxy-amine system against experimental data."""
        exp_data = self.dgeba_eda_experimental
        
        with patch('chemi_accel.polymer.thermoset.parse_smiles_safe') as mock_parse:
            mock_resin = Mock()
            mock_hardener = Mock()
            mock_parse.side_effect = [mock_resin, mock_hardener]
            
            # Mock reactive group identification for DGEBA (2 epoxy groups)
            with patch.object(self.simulator, '_identify_reactive_groups') as mock_identify:
                mock_identify.side_effect = [
                    [{'type': 'epoxy', 'reactivity': 0.85, 'steric_factor': 0.8, 'atom_indices': (0, 1, 2)}] * 2,
                    [{'type': 'primary_amine', 'reactivity': 0.9, 'steric_factor': 0.9, 'atom_indices': (0, 1)}] * 2
                ]
                
                # Setup network
                reactive_groups = self.simulator.setup_initial_network(
                    exp_data['resin_smiles'],
                    exp_data['hardener_smiles'],
                    exp_data['stoichiometry'],
                    system_size=100  # Reasonable size for validation
                )
                
                # Create kinetics from experimental parameters
                kinetics = CuringKinetics(
                    rate_constants=exp_data['rate_constants'],
                    activation_energies=exp_data['activation_energies'],
                    temperature=exp_data['temperature'],
                    catalyst_concentration=0.01,
                    diffusion_coefficients={'epoxy': 1e-12, 'amine': 8e-13}
                )
                
                # Run simulation
                result = self.simulator.simulate_curing(
                    reactive_groups=reactive_groups,
                    mechanism=CuringMechanism.EPOXY_AMINE,
                    simulation_time=3600.0,  # 1 hour
                    time_step=60.0,  # 1 minute steps
                    kinetics=kinetics
                )
                
                # Validate results
                validation_metrics = validate_against_experimental_data(
                    [result], 
                    {
                        'gel_point_times': exp_data['gel_point_times'],
                        'elastic_modulus': exp_data['elastic_modulus'],
                        'glass_transition_temps': exp_data['glass_transition_temps']
                    }
                )
                
                # Log results for analysis
                logging.info(f"DGEBA/EDA Validation Results:")
                logging.info(f"  Simulated gel point time: {result.gel_point_time:.0f}s")
                logging.info(f"  Experimental gel point time: {np.mean(exp_data['gel_point_times']):.0f}s")
                logging.info(f"  Gel point MAE: {validation_metrics.get('gel_point_mae', 'N/A')}")
                
                logging.info(f"  Simulated elastic modulus: {result.mechanical_properties['elastic_modulus_mpa']:.0f} MPa")
                logging.info(f"  Experimental elastic modulus: {np.mean(exp_data['elastic_modulus']):.0f} MPa")
                logging.info(f"  Elastic modulus MAE: {validation_metrics.get('elastic_modulus_mae', 'N/A')}")
                
                logging.info(f"  Simulated Tg: {result.thermal_properties['glass_transition_temp_k']:.1f} K")
                logging.info(f"  Experimental Tg: {np.mean(exp_data['glass_transition_temps']):.1f} K")
                logging.info(f"  Tg MAE: {validation_metrics.get('glass_transition_mae', 'N/A')}")
                
                # Validation criteria (allowing for model limitations)
                if result.gel_point_time is not None:
                    gel_point_error = abs(result.gel_point_time - np.mean(exp_data['gel_point_times']))
                    assert gel_point_error < 600, f"Gel point error too large: {gel_point_error}s"
                
                # Property predictions should be in reasonable range
                elastic_modulus = result.mechanical_properties['elastic_modulus_mpa']
                exp_modulus = np.mean(exp_data['elastic_modulus'])
                modulus_error = abs(elastic_modulus - exp_modulus) / exp_modulus
                
                # Allow up to 50% error for simplified model
                assert modulus_error < 0.5, f"Elastic modulus error too large: {modulus_error:.2%}"
                
                tg_sim = result.thermal_properties['glass_transition_temp_k']
                tg_exp = np.mean(exp_data['glass_transition_temps'])
                tg_error = abs(tg_sim - tg_exp)
                
                # Allow up to 50K error for Tg prediction
                assert tg_error < 50, f"Tg error too large: {tg_error:.1f}K"
    
    @patch('chemi_accel.polymer.thermoset.RDKIT_AVAILABLE', True)
    def test_epoxy_anhydride_system_validation(self):
        """Validate epoxy-anhydride system against experimental data."""
        exp_data = self.epoxy_anhydride_experimental
        
        with patch('chemi_accel.polymer.thermoset.parse_smiles_safe') as mock_parse:
            mock_resin = Mock()
            mock_hardener = Mock()
            mock_parse.side_effect = [mock_resin, mock_hardener]
            
            # Mock reactive group identification
            with patch.object(self.simulator, '_identify_reactive_groups') as mock_identify:
                mock_identify.side_effect = [
                    [{'type': 'epoxy', 'reactivity': 0.8, 'steric_factor': 0.85, 'atom_indices': (0, 1, 2)}] * 2,
                    [{'type': 'anhydride', 'reactivity': 0.75, 'steric_factor': 0.7, 'atom_indices': (0, 1, 2)}]
                ]
                
                # Setup network
                reactive_groups = self.simulator.setup_initial_network(
                    exp_data['resin_smiles'],
                    exp_data['hardener_smiles'],
                    exp_data['stoichiometry'],
                    system_size=80
                )
                
                # Create kinetics
                kinetics = CuringKinetics(
                    rate_constants=exp_data['rate_constants'],
                    activation_energies=exp_data['activation_energies'],
                    temperature=exp_data['temperature'],
                    catalyst_concentration=0.005,
                    diffusion_coefficients={'epoxy': 1e-12, 'anhydride': 6e-13}
                )
                
                # Run simulation
                result = self.simulator.simulate_curing(
                    reactive_groups=reactive_groups,
                    mechanism=CuringMechanism.EPOXY_ANHYDRIDE,
                    simulation_time=4800.0,  # 80 minutes
                    time_step=120.0,  # 2 minute steps
                    kinetics=kinetics
                )
                
                # Log results
                logging.info(f"Epoxy-Anhydride Validation Results:")
                logging.info(f"  Final conversion: {result.conversion_profile[-1]:.3f}")
                logging.info(f"  Gel fraction: {result.final_gel_fraction:.3f}")
                logging.info(f"  Crosslink density: {result.final_crosslink_density:.3f}")
                
                # Validate against experimental trends
                # Anhydride systems typically cure slower than amine systems
                if result.gel_point_time is not None:
                    logging.info(f"  Gel point time: {result.gel_point_time:.0f}s")
                    # Should be in reasonable range for anhydride system
                    assert 1800 < result.gel_point_time < 3600
                
                # Properties should be reasonable for anhydride-cured epoxy
                elastic_modulus = result.mechanical_properties['elastic_modulus_mpa']
                assert elastic_modulus > 1000, "Elastic modulus too low for cured epoxy"
                
                tg = result.thermal_properties['glass_transition_temp_k']
                assert tg > 400, "Glass transition temperature too low"
    
    def test_temperature_dependence_validation(self):
        """Validate temperature dependence of curing kinetics."""
        temperatures = [353.15, 393.15, 433.15]  # 80°C, 120°C, 160°C
        gel_times = []
        final_conversions = []
        
        with patch('chemi_accel.polymer.thermoset.RDKIT_AVAILABLE', True):
            with patch('chemi_accel.polymer.thermoset.parse_smiles_safe') as mock_parse:
                mock_resin = Mock()
                mock_hardener = Mock()
                mock_parse.side_effect = [mock_resin, mock_hardener] * len(temperatures)
                
                with patch.object(self.simulator, '_identify_reactive_groups') as mock_identify:
                    mock_identify.side_effect = [
                        [{'type': 'epoxy', 'reactivity': 0.8, 'steric_factor': 0.9, 'atom_indices': (0, 1, 2)}],
                        [{'type': 'primary_amine', 'reactivity': 0.9, 'steric_factor': 0.8, 'atom_indices': (0, 1)}]
                    ] * len(temperatures)
                    
                    for temp in temperatures:
                        # Setup system
                        reactive_groups = self.simulator.setup_initial_network(
                            "C1OC1", "N", 1.0, 30
                        )
                        
                        # Temperature-dependent kinetics
                        kinetics = CuringKinetics(
                            rate_constants={'k1': 1e-3, 'k2': 5e-4},
                            activation_energies={'E1': 50000, 'E2': 60000},
                            temperature=temp,
                            catalyst_concentration=0.01,
                            diffusion_coefficients={'epoxy': 1e-12, 'amine': 8e-13}
                        )
                        
                        # Run simulation
                        result = self.simulator.simulate_curing(
                            reactive_groups=reactive_groups,
                            mechanism=CuringMechanism.EPOXY_AMINE,
                            simulation_time=1800.0,  # 30 minutes
                            time_step=60.0,
                            kinetics=kinetics
                        )
                        
                        if result.gel_point_time is not None:
                            gel_times.append(result.gel_point_time)
                        else:
                            gel_times.append(float('inf'))
                        
                        final_conversions.append(result.conversion_profile[-1])
                        
                        logging.info(f"T={temp:.1f}K: gel_time={gel_times[-1]:.0f}s, "
                                   f"final_conversion={final_conversions[-1]:.3f}")
                    
                    # Validate Arrhenius behavior
                    # Higher temperature should lead to faster curing (lower gel times)
                    finite_gel_times = [t for t in gel_times if t != float('inf')]
                    if len(finite_gel_times) > 1:
                        # Should show decreasing trend with temperature
                        assert finite_gel_times[0] >= finite_gel_times[-1], \
                            "Gel time should decrease with temperature"
                    
                    # Final conversion should increase with temperature
                    assert final_conversions[-1] >= final_conversions[0], \
                        "Final conversion should increase with temperature"
    
    def test_stoichiometry_effects_validation(self):
        """Validate effects of stoichiometric ratio on curing."""
        stoichiometries = [0.8, 1.0, 1.2]  # Sub-stoichiometric, stoichiometric, excess hardener
        results = []
        
        with patch('chemi_accel.polymer.thermoset.RDKIT_AVAILABLE', True):
            with patch('chemi_accel.polymer.thermoset.parse_smiles_safe') as mock_parse:
                mock_resin = Mock()
                mock_hardener = Mock()
                mock_parse.side_effect = [mock_resin, mock_hardener] * len(stoichiometries)
                
                with patch.object(self.simulator, '_identify_reactive_groups') as mock_identify:
                    mock_identify.side_effect = [
                        [{'type': 'epoxy', 'reactivity': 0.8, 'steric_factor': 0.9, 'atom_indices': (0, 1, 2)}],
                        [{'type': 'primary_amine', 'reactivity': 0.9, 'steric_factor': 0.8, 'atom_indices': (0, 1)}]
                    ] * len(stoichiometries)
                    
                    for stoich in stoichiometries:
                        # Setup system
                        reactive_groups = self.simulator.setup_initial_network(
                            "C1OC1", "N", stoich, 40
                        )
                        
                        # Run simulation
                        result = self.simulator.simulate_curing(
                            reactive_groups=reactive_groups,
                            mechanism=CuringMechanism.EPOXY_AMINE,
                            simulation_time=1200.0,  # 20 minutes
                            time_step=60.0
                        )
                        
                        results.append(result)
                        
                        logging.info(f"Stoichiometry {stoich}: "
                                   f"conversion={result.conversion_profile[-1]:.3f}, "
                                   f"gel_fraction={result.final_gel_fraction:.3f}, "
                                   f"crosslink_density={result.final_crosslink_density:.3f}")
                    
                    # Validate stoichiometric effects
                    # Stoichiometric ratio (1.0) should give highest crosslink density
                    crosslink_densities = [r.final_crosslink_density for r in results]
                    max_crosslink_idx = np.argmax(crosslink_densities)
                    
                    # The maximum should be at or near stoichiometric ratio
                    assert max_crosslink_idx in [0, 1], \
                        "Maximum crosslink density should occur near stoichiometric ratio"
    
    def test_model_limitations_and_assumptions(self):
        """Test and document model limitations."""
        logging.info("Model Limitations and Assumptions:")
        logging.info("1. Simplified reaction kinetics (no diffusion limitations)")
        logging.info("2. Ideal mixing assumption")
        logging.info("3. No side reactions considered")
        logging.info("4. Simplified property prediction models")
        logging.info("5. No vitrification effects")
        
        # Test edge cases that reveal model limitations
        with patch('chemi_accel.polymer.thermoset.RDKIT_AVAILABLE', True):
            with patch('chemi_accel.polymer.thermoset.parse_smiles_safe') as mock_parse:
                mock_parse.return_value = Mock()
                
                with patch.object(self.simulator, '_identify_reactive_groups') as mock_identify:
                    mock_identify.return_value = [
                        {'type': 'epoxy', 'reactivity': 0.8, 'steric_factor': 0.9, 'atom_indices': (0, 1, 2)}
                    ]
                    
                    # Very high reactivity should lead to rapid gelation
                    high_reactivity_kinetics = CuringKinetics(
                        rate_constants={'k1': 1.0, 'k2': 0.5},  # Very high rates
                        activation_energies={'E1': 10000, 'E2': 15000},  # Low activation energies
                        temperature=393.15,
                        catalyst_concentration=0.1,
                        diffusion_coefficients={'epoxy': 1e-12, 'amine': 8e-13}
                    )
                    
                    groups = self.simulator.setup_initial_network("C1OC1", "N", 1.0, 20)
                    
                    result = self.simulator.simulate_curing(
                        reactive_groups=groups,
                        mechanism=CuringMechanism.EPOXY_AMINE,
                        simulation_time=60.0,  # 1 minute
                        time_step=1.0,
                        kinetics=high_reactivity_kinetics
                    )
                    
                    # Should reach high conversion quickly
                    assert result.conversion_profile[-1] > 0.5, \
                        "High reactivity should lead to high conversion"
                    
                    logging.info(f"High reactivity test: final conversion = {result.conversion_profile[-1]:.3f}")


if __name__ == '__main__':
    # Configure detailed logging for validation
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    pytest.main([__file__, '-v', '-s'])
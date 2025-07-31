"""
Integration tests for copolymer and blend property modeling.
"""

from typing import Any, Dict

import pytest

from src.chemi_accel.polymer.copolymer import (
    CopolymerBlendModeler,
    CopolymerType,
    MixingRule,
    PhaseState,
)


class TestCopolymerIntegration:
    """Integration tests for copolymer modeling."""

    def setup_method(self):
        """Set up test environment."""
        self.modeler = CopolymerBlendModeler()

        # Common polymer data
        self.polymer_database = {
            "styrene": {
                "smiles": "C=Cc1ccccc1",
                "name": "Styrene",
                "molecular_weight": 104.15,
                "glass_transition_temp": 373.0,
                "density": 1.05,
                "solubility_parameter": 18.5,
                "properties": {
                    "youngs_modulus": 3200.0,
                    "tensile_strength": 40.0,
                    "thermal_conductivity": 0.13,
                },
            },
            "butadiene": {
                "smiles": "C=CC=C",
                "name": "Butadiene",
                "molecular_weight": 54.09,
                "glass_transition_temp": 165.0,
                "density": 0.90,
                "solubility_parameter": 17.0,
                "properties": {
                    "youngs_modulus": 1.5,
                    "tensile_strength": 2.0,
                    "thermal_conductivity": 0.25,
                },
            },
            "methyl_methacrylate": {
                "smiles": "C=C(C)C(=O)OC",
                "name": "Methyl Methacrylate",
                "molecular_weight": 100.12,
                "glass_transition_temp": 378.0,
                "density": 1.19,
                "solubility_parameter": 19.3,
                "properties": {
                    "youngs_modulus": 2800.0,
                    "tensile_strength": 65.0,
                    "thermal_conductivity": 0.19,
                },
            },
            "ethylene": {
                "smiles": "C=C",
                "name": "Ethylene",
                "molecular_weight": 28.05,
                "glass_transition_temp": 200.0,
                "density": 0.92,
                "solubility_parameter": 16.0,
                "properties": {
                    "youngs_modulus": 800.0,
                    "tensile_strength": 25.0,
                    "thermal_conductivity": 0.33,
                },
            },
        }

    def create_monomer_data(
        self, monomer_name: str, mole_fraction: float, weight_fraction: float
    ) -> Dict[str, Any]:
        """Create monomer data for copolymer system."""
        base_data = self.polymer_database[monomer_name].copy()
        base_data.update(
            {"mole_fraction": mole_fraction, "weight_fraction": weight_fraction}
        )
        return base_data

    def test_styrene_butadiene_random_copolymer(self):
        """Test styrene-butadiene random copolymer (SBR)."""
        # Create 70/30 styrene/butadiene random copolymer
        monomer_data = [
            self.create_monomer_data("styrene", 0.7, 0.75),
            self.create_monomer_data("butadiene", 0.3, 0.25),
        ]

        system = self.modeler.create_copolymer_system(
            monomer_data, CopolymerType.RANDOM
        )

        results = self.modeler.predict_copolymer_properties(system)
        properties = results["predicted_properties"]

        # Verify properties are reasonable
        assert (
            200 < properties.glass_transition_temp < 350
        )  # Between component Tg values
        assert 0.9 < properties.density < 1.1  # Between component densities
        assert properties.youngs_modulus > 100  # Should have reasonable modulus

        # Test that random copolymer doesn't show microphase separation
        assert "microphase_info" not in results

    def test_styrene_butadiene_block_copolymer(self):
        """Test styrene-butadiene block copolymer (SBS)."""
        # Create 50/50 styrene/butadiene block copolymer
        monomer_data = [
            self.create_monomer_data("styrene", 0.5, 0.55),
            self.create_monomer_data("butadiene", 0.5, 0.45),
        ]

        system = self.modeler.create_copolymer_system(
            monomer_data,
            CopolymerType.BLOCK,
            sequence_length=1000,
            block_lengths={"styrene": 250, "butadiene": 250},
        )

        results = self.modeler.predict_copolymer_properties(system)
        properties = results["predicted_properties"]
        microphase_info = results["microphase_info"]

        # Verify properties
        assert properties.glass_transition_temp is not None
        assert properties.density is not None
        assert properties.youngs_modulus is not None

        # Verify microphase separation analysis
        assert microphase_info.chi_parameter is not None
        assert microphase_info.segregation_strength is not None
        assert microphase_info.morphology in [
            "spherical",
            "cylindrical",
            "lamellar",
            "disordered",
        ]

        # For 50/50 composition, expect lamellar morphology if phase separated
        if microphase_info.is_microphase_separated:
            assert microphase_info.morphology == "lamellar"

    def test_composition_effects_on_properties(self):
        """Test how composition affects copolymer properties."""
        compositions = [0.1, 0.3, 0.5, 0.7, 0.9]
        tg_values = []
        modulus_values = []

        for styrene_fraction in compositions:
            butadiene_fraction = 1.0 - styrene_fraction

            monomer_data = [
                self.create_monomer_data("styrene", styrene_fraction, styrene_fraction),
                self.create_monomer_data(
                    "butadiene", butadiene_fraction, butadiene_fraction
                ),
            ]

            system = self.modeler.create_copolymer_system(
                monomer_data, CopolymerType.RANDOM
            )

            results = self.modeler.predict_copolymer_properties(system)
            properties = results["predicted_properties"]

            tg_values.append(properties.glass_transition_temp)
            modulus_values.append(properties.youngs_modulus)

        # Verify monotonic trends
        assert all(tg_values[i] <= tg_values[i + 1] for i in range(len(tg_values) - 1))
        assert all(
            modulus_values[i] <= modulus_values[i + 1]
            for i in range(len(modulus_values) - 1)
        )

        # Verify boundary conditions
        assert abs(tg_values[0] - 165.0) < 50  # Close to butadiene Tg
        assert abs(tg_values[-1] - 373.0) < 50  # Close to styrene Tg

    def test_architecture_effects(self):
        """Test effects of different copolymer architectures."""
        # Same composition, different architectures
        monomer_data = [
            self.create_monomer_data("styrene", 0.6, 0.65),
            self.create_monomer_data("butadiene", 0.4, 0.35),
        ]

        architectures = [
            CopolymerType.RANDOM,
            CopolymerType.ALTERNATING,
            CopolymerType.BLOCK,
            CopolymerType.GRAFT,
        ]

        results_by_architecture = {}

        for arch in architectures:
            system = self.modeler.create_copolymer_system(monomer_data, arch)
            results = self.modeler.predict_copolymer_properties(system)
            results_by_architecture[arch] = results["predicted_properties"]

        # Block and graft copolymers should have enhanced mechanical properties
        random_modulus = results_by_architecture[CopolymerType.RANDOM].youngs_modulus
        block_modulus = results_by_architecture[CopolymerType.BLOCK].youngs_modulus
        graft_modulus = results_by_architecture[CopolymerType.GRAFT].youngs_modulus

        assert block_modulus >= random_modulus
        assert graft_modulus >= random_modulus

    def test_polymer_blend_miscibility(self):
        """Test polymer blend miscibility analysis."""
        # Test PE/PS blend (known to be immiscible)
        pe_data = [self.create_monomer_data("ethylene", 1.0, 1.0)]
        ps_data = [self.create_monomer_data("styrene", 1.0, 1.0)]

        blend_results = self.modeler.analyze_polymer_blend(
            pe_data, ps_data, blend_ratio=0.5, temperature=298.15
        )

        miscibility_info = blend_results["miscibility_info"]

        # PE/PS should be immiscible due to large solubility parameter difference
        assert miscibility_info.phase_state in [
            PhaseState.IMMISCIBLE,
            PhaseState.PARTIALLY_MISCIBLE,
        ]
        assert (
            miscibility_info.interaction_parameter > 0.1
        )  # Should be significantly above critical value

        # Test temperature effects
        high_temp_results = self.modeler.analyze_polymer_blend(
            pe_data, ps_data, blend_ratio=0.5, temperature=500.0
        )

        # Higher temperature should improve miscibility
        assert (
            high_temp_results["miscibility_info"].interaction_parameter
            < miscibility_info.interaction_parameter
        )

    def test_compatible_polymer_blend(self):
        """Test blend of more compatible polymers."""
        # Test styrene/MMA blend (more compatible)
        styrene_data = [self.create_monomer_data("styrene", 1.0, 1.0)]
        mma_data = [self.create_monomer_data("methyl_methacrylate", 1.0, 1.0)]

        blend_results = self.modeler.analyze_polymer_blend(
            styrene_data, mma_data, blend_ratio=0.5, temperature=298.15
        )

        miscibility_info = blend_results["miscibility_info"]

        # Should be more miscible than PE/PS due to similar solubility parameters
        assert miscibility_info.interaction_parameter < 1.0

        # Verify blend properties are reasonable
        blend_properties = blend_results["blend_properties"]
        assert 300 < blend_properties.glass_transition_temp < 400
        assert 1.0 < blend_properties.density < 1.2

    def test_composition_study_comprehensive(self):
        """Test comprehensive composition study."""
        styrene_data = self.polymer_database["styrene"].copy()
        mma_data = self.polymer_database["methyl_methacrylate"].copy()

        comp_study = self.modeler.generate_composition_study(
            styrene_data, mma_data, composition_range=(0.0, 1.0), num_points=11
        )

        compositions = comp_study["compositions"]
        properties = comp_study["properties"]
        microphase_info = comp_study["microphase_info"]

        # Verify data structure
        assert len(compositions) == 11
        assert len(properties) == 11
        assert len(microphase_info) == 11

        # Verify property trends
        tg_values = [prop.glass_transition_temp for prop in properties]
        density_values = [prop.density for prop in properties]

        # Should show smooth transitions
        assert all(tg is not None for tg in tg_values)
        assert all(d is not None for d in density_values)

        # Boundary conditions
        assert abs(tg_values[0] - styrene_data["glass_transition_temp"]) < 20
        assert abs(tg_values[-1] - mma_data["glass_transition_temp"]) < 20

    def test_mixing_rule_comparison(self):
        """Test different mixing rules for property prediction."""
        monomer_data = [
            self.create_monomer_data("styrene", 0.5, 0.55),
            self.create_monomer_data("methyl_methacrylate", 0.5, 0.45),
        ]

        system = self.modeler.create_copolymer_system(
            monomer_data, CopolymerType.RANDOM
        )

        # Test different mixing rules for Tg
        mixing_rules = [
            {MixingRule.LINEAR: "glass_transition_temp"},
            {MixingRule.GORDON_TAYLOR: "glass_transition_temp"},
            {MixingRule.FOX: "glass_transition_temp"},
        ]

        tg_values = []

        for rule_dict in mixing_rules:
            rule = list(rule_dict.keys())[0]
            custom_rules = {"glass_transition_temp": rule}

            properties = self.modeler.property_predictor.predict_properties(
                system, custom_rules
            )
            tg_values.append(properties.glass_transition_temp)

        # All should give reasonable values
        assert all(300 < tg < 400 for tg in tg_values)

        # Fox equation typically gives lower values than linear
        linear_tg, gt_tg, fox_tg = tg_values
        assert fox_tg <= linear_tg

    def test_microphase_separation_prediction(self):
        """Test microphase separation prediction accuracy."""
        # Create systems with different segregation strengths
        test_cases = [
            # Low segregation (should not phase separate)
            {
                "monomers": ["styrene", "methyl_methacrylate"],
                "fractions": [0.5, 0.5],
                "expected_separation": False,
            },
            # High segregation (should phase separate)
            {
                "monomers": ["styrene", "ethylene"],
                "fractions": [0.5, 0.5],
                "expected_separation": True,
            },
        ]

        for case in test_cases:
            monomer_data = []
            for i, monomer in enumerate(case["monomers"]):
                fraction = case["fractions"][i]
                monomer_data.append(
                    self.create_monomer_data(monomer, fraction, fraction)
                )

            system = self.modeler.create_copolymer_system(
                monomer_data, CopolymerType.BLOCK, sequence_length=1000
            )

            results = self.modeler.predict_copolymer_properties(system)
            microphase_info = results["microphase_info"]

            # Check prediction matches expectation
            if case["expected_separation"]:
                assert microphase_info.segregation_strength > 10.5
            else:
                # May or may not separate, but segregation strength should be reasonable
                assert microphase_info.segregation_strength >= 0

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        # Test with extreme compositions
        extreme_data = [
            self.create_monomer_data("styrene", 0.99, 0.99),
            self.create_monomer_data("butadiene", 0.01, 0.01),
        ]

        system = self.modeler.create_copolymer_system(
            extreme_data, CopolymerType.RANDOM
        )

        results = self.modeler.predict_copolymer_properties(system)

        # Should still produce reasonable results
        assert results["predicted_properties"].glass_transition_temp is not None

        # Test with missing property data
        incomplete_data = [
            {
                "smiles": "C=C",
                "name": "Ethylene",
                "mole_fraction": 0.5,
                "weight_fraction": 0.5,
                "molecular_weight": 28.0,
                # Missing Tg and other properties
            },
            {
                "smiles": "C=Cc1ccccc1",
                "name": "Styrene",
                "mole_fraction": 0.5,
                "weight_fraction": 0.5,
                "molecular_weight": 104.0,
                # Missing Tg and other properties
            },
        ]

        system = self.modeler.create_copolymer_system(
            incomplete_data, CopolymerType.RANDOM
        )

        # Should handle missing data gracefully
        results = self.modeler.predict_copolymer_properties(system)
        assert results is not None

    def test_performance_with_large_systems(self):
        """Test performance with larger copolymer systems."""
        # Create a multicomponent system
        monomers = ["styrene", "butadiene", "methyl_methacrylate", "ethylene"]
        fractions = [0.4, 0.3, 0.2, 0.1]

        monomer_data = []
        for monomer, fraction in zip(monomers, fractions):
            monomer_data.append(self.create_monomer_data(monomer, fraction, fraction))

        system = self.modeler.create_copolymer_system(
            monomer_data, CopolymerType.RANDOM
        )

        # Should handle multicomponent systems
        results = self.modeler.predict_copolymer_properties(system)
        assert results["predicted_properties"] is not None

        # Test composition study with many points
        comp_study = self.modeler.generate_composition_study(
            self.polymer_database["styrene"],
            self.polymer_database["butadiene"],
            num_points=21,
        )

        assert len(comp_study["compositions"]) == 21
        assert len(comp_study["properties"]) == 21


if __name__ == "__main__":
    pytest.main([__file__])

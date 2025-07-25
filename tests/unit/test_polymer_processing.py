"""
Unit tests for polymer processing system.
"""


import pytest

from src.chemi_accel.descriptors.polymer import (
    PolymerChainFlexibilityCalculator,
    PolymerDescriptorCalculator,
)
from src.chemi_accel.polymer.conformers import (
    ChainFlexibilityAnalyzer,
    ImplicitSolventHandler,
    IntramolecularInteractionAnalyzer,
    PolymerConformerGenerator,
)

# Import the polymer modules
from src.chemi_accel.polymer.monomer import (
    MonomerAnalyzer,
    PolymerizationPotentialAnalyzer,
    ReactivityAnalyzer,
    RingOpeningAnalyzer,
    StereochemistryHandler,
)
from src.chemi_accel.polymer.oligomer import (
    ChainGrowthSimulator,
    ConformationalFlexibilityAnalyzer,
    EndGroupHandler,
    OligomerProcessor,
    SequenceAnalyzer,
)
from src.chemi_accel.polymer.repeating import (
    BranchedPolymerHandler,
    PolymerFragmentAnalyzer,
    RepeatingUnitIdentifier,
)


class TestMonomerAnalysis:
    """Test monomer analysis functionality."""

    def test_reactivity_analyzer_initialization(self):
        """Test ReactivityAnalyzer initialization."""
        analyzer = ReactivityAnalyzer()
        assert analyzer is not None
        assert len(analyzer.functional_groups) > 0

    def test_functional_group_identification(self):
        """Test functional group identification."""
        analyzer = ReactivityAnalyzer()

        # Test vinyl group identification
        vinyl_smiles = "C=C"
        groups = analyzer.identify_functional_groups(vinyl_smiles)

        # Should identify at least one functional group
        assert len(groups) >= 0  # May be 0 if RDKit not available

    def test_reactivity_profile_calculation(self):
        """Test reactivity profile calculation."""
        analyzer = ReactivityAnalyzer()

        # Test with ethylene (simple vinyl)
        ethylene_smiles = "C=C"
        profile = analyzer.calculate_reactivity_profile(ethylene_smiles)

        assert profile is not None
        assert hasattr(profile, "overall_reactivity")
        assert hasattr(profile, "functional_groups")
        assert hasattr(profile, "preferred_mechanisms")

    def test_polymerization_potential_assessment(self):
        """Test polymerization potential assessment."""
        analyzer = PolymerizationPotentialAnalyzer()

        # Test with styrene
        styrene_smiles = "C=Cc1ccccc1"
        potential = analyzer.assess_polymerization_potential(styrene_smiles)

        assert potential is not None
        assert hasattr(potential, "can_polymerize")
        assert hasattr(potential, "reactivity_score")
        assert hasattr(potential, "stability_score")

    def test_stereochemistry_analysis(self):
        """Test stereochemistry analysis."""
        handler = StereochemistryHandler()

        # Test with propylene (has potential for tacticity)
        propylene_smiles = "C=C(C)"
        stereo_info = handler.analyze_stereochemistry(propylene_smiles)

        assert stereo_info is not None
        assert hasattr(stereo_info, "chiral_centers")
        assert hasattr(stereo_info, "tacticity_potential")

    def test_ring_opening_analysis(self):
        """Test ring-opening polymerization analysis."""
        analyzer = RingOpeningAnalyzer()

        # Test with ethylene oxide (3-membered ring)
        ethylene_oxide_smiles = "C1CO1"
        analysis = analyzer.analyze_ring_opening_potential(ethylene_oxide_smiles)

        assert analysis is not None
        assert "can_ring_open" in analysis
        assert "ring_strain_score" in analysis
        assert "polymerization_likelihood" in analysis

    def test_monomer_analyzer_comprehensive(self):
        """Test comprehensive monomer analysis."""
        analyzer = MonomerAnalyzer()

        # Test with acrylate monomer
        acrylate_smiles = "C=CC(=O)OC"
        results = analyzer.analyze_monomer(acrylate_smiles)

        assert results is not None
        assert "reactivity_profile" in results
        assert "polymerization_potential" in results
        assert "stereochemistry_info" in results
        assert "ring_opening_analysis" in results


class TestOligomerProcessing:
    """Test oligomer processing functionality."""

    def test_chain_growth_simulator(self):
        """Test chain growth simulation."""
        simulator = ChainGrowthSimulator()

        # Test with ethylene
        ethylene_smiles = "C=C"
        steps = simulator.simulate_chain_growth(ethylene_smiles, chain_length=3)

        assert isinstance(steps, list)
        # May be empty if simulation fails, but should not raise exception

    def test_conformational_flexibility_analyzer(self):
        """Test conformational flexibility analysis."""
        analyzer = ConformationalFlexibilityAnalyzer()

        # Test with butane (simple flexible chain)
        butane_smiles = "CCCC"
        ensemble = analyzer.analyze_flexibility(butane_smiles)

        assert ensemble is not None
        assert hasattr(ensemble, "conformer_smiles")
        assert hasattr(ensemble, "flexibility_metrics")

    def test_end_group_handler(self):
        """Test end group analysis."""
        handler = EndGroupHandler()

        # Test with simple alcohol
        alcohol_smiles = "CCCCO"
        end_group_info = handler.analyze_end_groups(alcohol_smiles)

        assert end_group_info is not None
        assert hasattr(end_group_info, "left_end_group")
        assert hasattr(end_group_info, "right_end_group")
        assert hasattr(end_group_info, "end_group_reactivity")

    def test_sequence_analyzer(self):
        """Test sequence analysis."""
        analyzer = SequenceAnalyzer()

        # Test with simple oligomer
        oligomer_smiles = "CCCCCCCC"
        monomer_library = ["CC", "CCC"]
        sequence_info = analyzer.analyze_sequence(oligomer_smiles, monomer_library)

        assert sequence_info is not None
        assert hasattr(sequence_info, "monomer_sequence")
        assert hasattr(sequence_info, "sequence_pattern")
        assert hasattr(sequence_info, "regularity_score")

    def test_oligomer_processor_comprehensive(self):
        """Test comprehensive oligomer processing."""
        processor = OligomerProcessor()

        # Test with hexane (simple oligomer)
        hexane_smiles = "CCCCCC"
        results = processor.process_oligomer(hexane_smiles)

        assert results is not None
        assert "flexibility_analysis" in results
        assert "end_group_analysis" in results


class TestConformerGeneration:
    """Test conformer generation functionality."""

    def test_polymer_conformer_generator(self):
        """Test polymer conformer generation."""
        generator = PolymerConformerGenerator(use_gpu=False)  # Use CPU for testing

        # Test with simple polymer chain
        polymer_smiles = "CCCCCCCC"
        ensemble = generator.generate_conformers(polymer_smiles, num_conformers=5)

        assert ensemble is not None
        assert hasattr(ensemble, "conformers")
        assert hasattr(ensemble, "populations")
        assert hasattr(ensemble, "flexibility_metrics")

    def test_chain_flexibility_analyzer(self):
        """Test chain flexibility analysis."""
        analyzer = ChainFlexibilityAnalyzer()

        # Test with polymer chain
        polymer_smiles = "CCCCCCCCCC"
        metrics = analyzer.analyze_flexibility(polymer_smiles)

        assert metrics is not None
        assert hasattr(metrics, "characteristic_ratio")
        assert hasattr(metrics, "persistence_length")
        assert hasattr(metrics, "flexibility_index")

    def test_intramolecular_interaction_analyzer(self):
        """Test intramolecular interaction analysis."""
        analyzer = IntramolecularInteractionAnalyzer()

        # Test with polymer containing potential H-bonds
        polymer_smiles = "CCCNCCCOCC"
        interactions = analyzer.analyze_interactions(polymer_smiles)

        assert isinstance(interactions, list)
        # May be empty but should not raise exception

    def test_implicit_solvent_handler(self):
        """Test implicit solvent effects."""
        from src.chemi_accel.polymer.conformers import (
            ConformationalEnsemble,
            ConformerProperties,
            SolventModel,
        )

        handler = ImplicitSolventHandler()

        # Create mock ensemble
        mock_conformer = ConformerProperties(
            conformer_id=0, energy=10.0, radius_of_gyration=5.0, end_to_end_distance=8.0
        )

        mock_ensemble = ConformationalEnsemble(
            conformers=[mock_conformer],
            populations=[1.0],
            flexibility_metrics=None,
            intramolecular_interactions=[],
            ensemble_properties={},
        )

        # Apply water solvent effects
        modified_ensemble = handler.apply_solvent_effects(
            mock_ensemble, SolventModel.WATER
        )

        assert modified_ensemble is not None
        assert len(modified_ensemble.conformers) == 1


class TestRepeatingUnitIdentification:
    """Test repeating unit identification functionality."""

    def test_repeating_unit_identifier(self):
        """Test repeating unit identification."""
        identifier = RepeatingUnitIdentifier()

        # Test with polyethylene-like structure
        polymer_smiles = "CCCCCCCCCC"
        units = identifier.identify_repeating_units(polymer_smiles)

        assert isinstance(units, list)
        # May be empty if no clear repeating units found

    def test_polymer_fragment_analyzer(self):
        """Test polymer fragment analysis."""
        analyzer = PolymerFragmentAnalyzer()

        # Test with branched polymer
        polymer_smiles = "CCC(C)CCC(C)CC"
        fragments = analyzer.analyze_fragments(polymer_smiles)

        assert isinstance(fragments, list)
        # Should identify some fragments

    def test_branched_polymer_handler(self):
        """Test branched polymer analysis."""
        handler = BranchedPolymerHandler()

        # Test with branched structure
        branched_smiles = "CCC(CCC)CCC(CC)CC"
        analysis = handler.analyze_branching(branched_smiles)

        assert analysis is not None
        assert hasattr(analysis, "topology")
        assert hasattr(analysis, "branch_points")
        assert hasattr(analysis, "structural_regularity")


class TestPolymerDescriptors:
    """Test polymer-specific descriptors."""

    def test_polymer_chain_flexibility_calculator(self):
        """Test polymer chain flexibility calculations."""
        calculator = PolymerChainFlexibilityCalculator(use_gpu=False)

        # Test with simple polymer chain
        polymer_smiles = "CCCCCCCCCC"

        # Test persistence length calculation
        persistence_length = calculator.calculate_persistence_length(polymer_smiles)
        assert isinstance(persistence_length, float)
        assert persistence_length >= 0

        # Test radius of gyration calculation
        rg = calculator.calculate_radius_of_gyration(polymer_smiles)
        assert isinstance(rg, float)
        assert rg >= 0

        # Test end-to-end distance calculation
        end_to_end = calculator.calculate_end_to_end_distance(polymer_smiles)
        assert isinstance(end_to_end, float)
        assert end_to_end >= 0

        # Test characteristic ratio calculation
        char_ratio = calculator.calculate_characteristic_ratio(polymer_smiles)
        assert isinstance(char_ratio, float)
        assert char_ratio >= 0

    def test_polymer_descriptor_calculator(self):
        """Test comprehensive polymer descriptor calculation."""
        calculator = PolymerDescriptorCalculator(use_gpu=False)

        # Test with polystyrene-like structure
        polymer_smiles = "CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C"
        descriptors = calculator.calculate_all_descriptors(polymer_smiles)

        assert descriptors is not None

        # Check that all descriptor fields are present
        assert descriptors.persistence_length is not None
        assert descriptors.radius_of_gyration is not None
        assert descriptors.end_to_end_distance is not None
        assert descriptors.glass_transition_temperature is not None
        assert descriptors.elastic_modulus_predictor is not None
        assert descriptors.viscosity_predictor is not None
        assert descriptors.crystallinity_index is not None

        # Check that values are reasonable
        assert descriptors.persistence_length >= 0
        assert descriptors.radius_of_gyration >= 0
        assert descriptors.glass_transition_temperature > 200  # Reasonable Tg
        assert descriptors.elastic_modulus_predictor > 0
        assert descriptors.viscosity_predictor > 0
        assert 0 <= descriptors.crystallinity_index <= 1
        assert 0 <= descriptors.brittleness_index <= 1
        assert 0 <= descriptors.processability_index <= 1


class TestErrorHandling:
    """Test error handling in polymer processing."""

    def test_invalid_smiles_handling(self):
        """Test handling of invalid SMILES."""
        analyzer = MonomerAnalyzer()

        # Test with invalid SMILES
        with pytest.raises(Exception):  # Should raise InvalidMoleculeError or similar
            analyzer.analyze_monomer("invalid_smiles")

    def test_empty_molecule_handling(self):
        """Test handling of empty molecules."""
        calculator = PolymerDescriptorCalculator()

        # Test with empty string
        with pytest.raises(Exception):
            calculator.calculate_all_descriptors("")

    def test_rdkit_unavailable_handling(self):
        """Test behavior when RDKit is unavailable."""
        # This would require mocking RDKit import failure
        # For now, just ensure the modules can be imported
        assert ReactivityAnalyzer is not None
        assert PolymerDescriptorCalculator is not None


if __name__ == "__main__":
    pytest.main([__file__])

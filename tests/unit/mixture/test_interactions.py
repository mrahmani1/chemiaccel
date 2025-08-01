"""
Tests for inter-component interaction modeling.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.core.exceptions import ChemiAccelError
from chemi_accel.mixture.core import ComponentData, MixtureData
from chemi_accel.mixture.interactions import (
    ElectrostaticModel,
    HydrogenBondingModel,
    InteractionModeler,
    MolecularInteraction,
    PiStackingModel,
    PolymerSolventModel,
    VanDerWaalsModel,
)


class TestMolecularInteraction:
    """Tests for MolecularInteraction data structure."""

    def test_interaction_creation(self):
        """Test basic interaction creation."""
        interaction = MolecularInteraction(
            component_i=0,
            component_j=1,
            interaction_type="hydrogen_bonding",
            strength=5.2,
            energy=-10.5,
        )

        assert interaction.component_i == 0
        assert interaction.component_j == 1
        assert interaction.interaction_type == "hydrogen_bonding"
        assert interaction.strength == 5.2
        assert interaction.energy == -10.5
        assert not interaction.is_self_interaction

    def test_self_interaction(self):
        """Test self-interaction detection."""
        interaction = MolecularInteraction(
            component_i=0, component_j=0, interaction_type="van_der_waals", strength=1.0
        )

        assert interaction.is_self_interaction


class TestHydrogenBondingModel:
    """Tests for hydrogen bonding model."""

    def test_model_initialization(self):
        """Test model initialization."""
        backend = Mock()
        model = HydrogenBondingModel(backend)

        assert model.backend is backend
        assert model.donor_acceptor_cutoff == 3.5
        assert model.angle_cutoff == 120.0

    @patch("rdkit.Chem.rdMolDescriptors.CalcNumHBD")
    @patch("rdkit.Chem.rdMolDescriptors.CalcNumHBA")
    def test_energy_calculation(self, mock_hba, mock_hbd):
        """Test hydrogen bonding energy calculation."""
        backend = Mock()
        model = HydrogenBondingModel(backend)

        # Mock RDKit functions
        mock_hbd.side_effect = [1, 0]  # mol_i has 1 donor, mol_j has 0
        mock_hba.side_effect = [0, 1]  # mol_i has 0 acceptors, mol_j has 1

        mol_i = Mock()
        mol_j = Mock()

        energy = model.calculate_interaction_energy(mol_i, mol_j, distance=3.0)

        assert energy < 0  # Hydrogen bonds are attractive
        mock_hbd.assert_any_call(mol_i)
        mock_hbd.assert_any_call(mol_j)
        mock_hba.assert_any_call(mol_i)
        mock_hba.assert_any_call(mol_j)

    @patch("rdkit.Chem.rdMolDescriptors.CalcNumHBD")
    @patch("rdkit.Chem.rdMolDescriptors.CalcNumHBA")
    def test_no_hydrogen_bonding(self, mock_hba, mock_hbd):
        """Test case with no hydrogen bonding."""
        backend = Mock()
        model = HydrogenBondingModel(backend)

        # No donors or acceptors
        mock_hbd.return_value = 0
        mock_hba.return_value = 0

        mol_i = Mock()
        mol_j = Mock()

        energy = model.calculate_interaction_energy(mol_i, mol_j)

        assert energy == 0.0

    @patch("rdkit.Chem.rdMolDescriptors.CalcNumHBD")
    @patch("rdkit.Chem.rdMolDescriptors.CalcNumHBA")
    def test_descriptors(self, mock_hba, mock_hbd):
        """Test hydrogen bonding descriptors."""
        backend = Mock()
        model = HydrogenBondingModel(backend)

        mock_hbd.side_effect = [2, 1]
        mock_hba.side_effect = [1, 2]

        mol_i = Mock()
        mol_j = Mock()

        descriptors = model.get_interaction_descriptors(mol_i, mol_j)

        assert "hbd_i" in descriptors
        assert "hba_i" in descriptors
        assert "hbd_j" in descriptors
        assert "hba_j" in descriptors
        assert "h_bond_potential" in descriptors
        assert descriptors["hbd_i"] == 2
        assert descriptors["hba_j"] == 2


class TestPiStackingModel:
    """Tests for π-π stacking model."""

    @patch("rdkit.Chem.rdMolDescriptors.CalcNumAromaticRings")
    def test_stacking_energy(self, mock_aromatic_rings):
        """Test π-π stacking energy calculation."""
        backend = Mock()
        model = PiStackingModel(backend)

        # Both molecules have aromatic rings
        mock_aromatic_rings.side_effect = [2, 1]

        mol_i = Mock()
        mol_j = Mock()

        energy = model.calculate_interaction_energy(mol_i, mol_j, distance=3.4)

        assert energy < 0  # π-π stacking is attractive
        mock_aromatic_rings.assert_any_call(mol_i)
        mock_aromatic_rings.assert_any_call(mol_j)

    @patch("rdkit.Chem.rdMolDescriptors.CalcNumAromaticRings")
    def test_no_stacking(self, mock_aromatic_rings):
        """Test case with no aromatic rings."""
        backend = Mock()
        model = PiStackingModel(backend)

        # No aromatic rings
        mock_aromatic_rings.return_value = 0

        mol_i = Mock()
        mol_j = Mock()

        energy = model.calculate_interaction_energy(mol_i, mol_j)

        assert energy == 0.0


class TestVanDerWaalsModel:
    """Tests for van der Waals model."""

    def test_vdw_energy_calculation(self):
        """Test van der Waals energy calculation."""
        backend = Mock()
        model = VanDerWaalsModel(backend)

        mol_i = Mock()
        mol_j = Mock()

        # Mock molecular volume calculation
        with patch.object(
            model, "_calculate_molecular_volume", side_effect=[100.0, 120.0]
        ):
            energy = model.calculate_interaction_energy(mol_i, mol_j, distance=4.0)

        assert isinstance(energy, float)
        # At large distances, should be small and attractive
        assert energy < 0

    def test_repulsive_at_short_distance(self):
        """Test repulsive interaction at very short distances."""
        backend = Mock()
        model = VanDerWaalsModel(backend)

        mol_i = Mock()
        mol_j = Mock()

        with patch.object(
            model, "_calculate_molecular_volume", side_effect=[100.0, 100.0]
        ):
            energy = model.calculate_interaction_energy(mol_i, mol_j, distance=2.0)

        # At very short distances, should be repulsive (positive)
        assert energy > 0


class TestElectrostaticModel:
    """Tests for electrostatic model."""

    def test_electrostatic_energy(self):
        """Test electrostatic energy calculation."""
        backend = Mock()
        model = ElectrostaticModel(backend)

        mol_i = Mock()
        mol_j = Mock()

        # Mock partial charge calculation
        with patch.object(model, "_estimate_partial_charge", side_effect=[1.0, -1.0]):
            energy = model.calculate_interaction_energy(mol_i, mol_j, distance=5.0)

        assert energy < 0  # Opposite charges attract

    def test_neutral_molecules(self):
        """Test neutral molecules with no electrostatic interaction."""
        backend = Mock()
        model = ElectrostaticModel(backend)

        mol_i = Mock()
        mol_j = Mock()

        # Both molecules are neutral
        with patch.object(model, "_estimate_partial_charge", return_value=0.0):
            energy = model.calculate_interaction_energy(mol_i, mol_j)

        assert energy == 0.0


class TestPolymerSolventModel:
    """Tests for polymer-solvent interaction model."""

    def test_polymer_identification(self):
        """Test polymer identification."""
        backend = Mock()
        model = PolymerSolventModel(backend)

        # Mock molecule with high molecular weight (polymer)
        mol_polymer = Mock()
        with patch("rdkit.Chem.Descriptors.MolWt", return_value=2000.0):
            assert model._is_polymer(mol_polymer)

        # Mock molecule with low molecular weight (not polymer)
        mol_small = Mock()
        with patch("rdkit.Chem.Descriptors.MolWt", return_value=100.0):
            assert not model._is_polymer(mol_small)

    def test_flory_huggins_calculation(self):
        """Test Flory-Huggins interaction parameter calculation."""
        backend = Mock()
        model = PolymerSolventModel(backend)

        mol_i = Mock()
        mol_j = Mock()

        with patch.object(
            model, "_estimate_solubility_parameter", side_effect=[20.0, 25.0]
        ):
            chi = model._get_flory_huggins_parameter(mol_i, mol_j)

        assert isinstance(chi, float)
        assert chi > 0  # Should be positive for immiscible systems


class TestInteractionModeler:
    """Tests for main interaction modeling system."""

    def test_modeler_initialization(self):
        """Test interaction modeler initialization."""
        backend = Mock()
        modeler = InteractionModeler(backend)

        assert modeler.backend is backend
        assert "hydrogen_bonding" in modeler.models
        assert "pi_stacking" in modeler.models
        assert "van_der_waals" in modeler.models
        assert "electrostatic" in modeler.models
        assert "polymer_solvent" in modeler.models

    @patch("rdkit.Chem.MolFromSmiles")
    def test_calculate_interactions(self, mock_mol_from_smiles):
        """Test interaction calculation for mixture."""
        backend = Mock()
        modeler = InteractionModeler(backend)

        # Mock RDKit molecules
        mock_mol1 = Mock()
        mock_mol2 = Mock()
        mock_mol_from_smiles.side_effect = [mock_mol1, mock_mol2]

        # Create test mixture
        components = [
            ComponentData(smiles="CCO", mol_fraction=0.6),
            ComponentData(smiles="O", mol_fraction=0.4),
        ]
        mixture = MixtureData(components=components)

        # Mock interaction models to return specific energies
        for model in modeler.models.values():
            model.calculate_interaction_energy = Mock(return_value=-5.0)
            model.get_interaction_descriptors = Mock(return_value={"test": 1.0})

        interactions = modeler.calculate_interactions(
            mixture, interaction_types=["hydrogen_bonding", "van_der_waals"]
        )

        assert isinstance(interactions, list)
        # Should have interactions for each pair and each interaction type
        assert len(interactions) > 0

        for interaction in interactions:
            assert isinstance(interaction, MolecularInteraction)
            assert interaction.interaction_type in ["hydrogen_bonding", "van_der_waals"]

    def test_build_interaction_matrix(self):
        """Test interaction matrix construction."""
        backend = Mock()
        modeler = InteractionModeler(backend)

        interactions = [
            MolecularInteraction(0, 1, "test", 1.0, energy=-2.0),
            MolecularInteraction(1, 2, "test", 2.0, energy=-3.0),
        ]

        matrix = modeler.build_interaction_matrix(interactions, n_components=3)

        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == -2.0
        assert matrix[1, 0] == -2.0  # Should be symmetric
        assert matrix[1, 2] == -3.0

    def test_mixture_interaction_energy(self):
        """Test total mixture interaction energy calculation."""
        backend = Mock()
        modeler = InteractionModeler(backend)

        components = [
            ComponentData(smiles="CCO", mol_fraction=0.5),
            ComponentData(smiles="O", mol_fraction=0.5),
        ]
        mixture = MixtureData(components=components)

        interactions = [
            MolecularInteraction(0, 1, "test", 1.0, energy=-2.0),
            MolecularInteraction(0, 0, "test", 0.5, energy=-1.0),
        ]

        total_energy = modeler.calculate_mixture_interaction_energy(
            mixture, interactions
        )

        assert isinstance(total_energy, float)
        # Should be weighted by composition
        expected = 2 * 0.5 * 0.5 * (-2.0) + 0.5 * 0.5 * (-1.0)  # Cross + self
        assert abs(total_energy - expected) < 1e-6

    def test_predict_mixing_behavior(self):
        """Test mixing behavior prediction."""
        backend = Mock()
        modeler = InteractionModeler(backend)

        components = [ComponentData(smiles="CCO", mol_fraction=0.5)]
        mixture = MixtureData(components=components)

        interactions = [MolecularInteraction(0, 0, "test", 1.0, energy=-1.0)]

        predictions = modeler.predict_mixing_behavior(mixture, interactions)

        assert isinstance(predictions, dict)
        assert "total_interaction_energy" in predictions
        assert "mixing_enthalpy" in predictions
        assert "miscibility_score" in predictions
        assert "dominant_interactions" in predictions

    def test_invalid_smiles_error(self):
        """Test error handling for invalid SMILES."""
        backend = Mock()
        modeler = InteractionModeler(backend)

        components = [ComponentData(smiles="INVALID_SMILES", mol_fraction=1.0)]
        mixture = MixtureData(components=components)

        with patch("rdkit.Chem.MolFromSmiles", return_value=None):
            with pytest.raises(ChemiAccelError, match="Invalid SMILES"):
                modeler.calculate_interactions(mixture)

    def test_miscibility_score_calculation(self):
        """Test miscibility score calculation."""
        backend = Mock()
        modeler = InteractionModeler(backend)

        # Favorable interactions (negative) should increase miscibility
        favorable_matrix = np.array([[0, -2.0], [-2.0, 0]])
        miscibility_favorable = modeler._calculate_miscibility_score(favorable_matrix)

        # Unfavorable interactions (positive) should decrease miscibility
        unfavorable_matrix = np.array([[0, 2.0], [2.0, 0]])
        miscibility_unfavorable = modeler._calculate_miscibility_score(
            unfavorable_matrix
        )

        assert miscibility_favorable > miscibility_unfavorable
        assert 0 <= miscibility_favorable <= 1
        assert 0 <= miscibility_unfavorable <= 1

    def test_visualization_data_generation(self):
        """Test generation of visualization data."""
        backend = Mock()
        modeler = InteractionModeler(backend)

        components = [
            ComponentData(smiles="CCO", mol_fraction=0.6),
            ComponentData(smiles="O", mol_fraction=0.4),
        ]
        mixture = MixtureData(components=components)

        interactions = [
            MolecularInteraction(0, 1, "hydrogen_bonding", 2.0, energy=-3.0)
        ]

        viz_data = modeler.visualize_interactions(interactions, mixture)

        assert isinstance(viz_data, dict)
        assert "interaction_network" in viz_data
        assert "energy_landscape" in viz_data
        assert "interaction_heatmap" in viz_data

        # Check network data structure
        network_data = viz_data["interaction_network"]
        assert "nodes" in network_data
        assert "edges" in network_data
        assert len(network_data["nodes"]) == 2  # Two components
        assert len(network_data["edges"]) == 1  # One interaction

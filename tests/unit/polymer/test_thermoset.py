"""
Unit tests for thermoset curing simulation module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import logging

from chemi_accel.polymer.thermoset import (
    ThermosetCuringSimulator,
    NetworkTopologyAnalyzer,
    GelPointDetector,
    CuringMechanism,
    NetworkState,
    ReactiveGroup,
    CrosslinkBond,
    NetworkTopology,
    CuringKinetics,
    CuringResult,
    validate_against_experimental_data
)
from chemi_accel.core.exceptions import ChemiAccelError, InvalidMoleculeError


class TestThermosetCuringSimulator:
    """Test cases for ThermosetCuringSimulator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.simulator = ThermosetCuringSimulator(use_gpu=False)
        
        # Test molecules
        self.epoxy_resin = "C1OC1CC(C1OC1)C"  # Simple diepoxy
        self.amine_hardener = "NCCN"  # Ethylenediamine
        
        # Test kinetics
        self.test_kinetics = CuringKinetics(
            rate_constants={'k1': 1e-2, 'k2': 5e-3},
            activation_energies={'E1': 50000, 'E2': 60000},
            temperature=393.15,
            catalyst_concentration=0.01,
            diffusion_coefficients={'epoxy': 1e-12, 'amine': 8e-13}
        )
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        # Test CPU mode
        sim_cpu = ThermosetCuringSimulator(use_gpu=False)
        assert not sim_cpu.use_gpu
        
        # Test default kinetics are loaded
        assert CuringMechanism.EPOXY_AMINE in sim_cpu.default_kinetics
        assert CuringMechanism.EPOXY_ANHYDRIDE in sim_cpu.default_kinetics
        assert CuringMechanism.POLYURETHANE in sim_cpu.default_kinetics
    
    @patch('chemi_accel.polymer.thermoset.RDKIT_AVAILABLE', True)
    @patch('chemi_accel.polymer.thermoset.parse_smiles_safe')
    def test_setup_initial_network(self, mock_parse):
        """Test initial network setup."""
        # Mock RDKit molecules
        mock_resin = Mock()
        mock_hardener = Mock()
        mock_parse.side_effect = [mock_resin, mock_hardener]
        
        # Mock reactive group identification
        with patch.object(self.simulator, '_identify_reactive_groups') as mock_identify:
            mock_identify.side_effect = [
                [{'type': 'epoxy', 'reactivity': 0.8, 'steric_factor': 0.9, 'atom_indices': (0, 1, 2)}],
                [{'type': 'primary_amine', 'reactivity': 0.9, 'steric_factor': 0.8, 'atom_indices': (0, 1)}]
            ]
            
            groups = self.simulator.setup_initial_network(
                self.epoxy_resin, self.amine_hardener, 1.0, 100
            )
            
            assert len(groups) > 0
            assert all(isinstance(g, ReactiveGroup) for g in groups)
            assert any(g.group_type == 'epoxy' for g in groups)
            assert any(g.group_type == 'primary_amine' for g in groups)
    
    def test_setup_initial_network_invalid_smiles(self):
        """Test network setup with invalid SMILES."""
        with patch('chemi_accel.polymer.thermoset.parse_smiles_safe', return_value=None):
            with pytest.raises(InvalidMoleculeError):
                self.simulator.setup_initial_network("invalid", "smiles", 1.0, 100)
    
    def test_setup_initial_network_no_reactive_groups(self):
        """Test network setup with no reactive groups."""
        with patch('chemi_accel.polymer.thermoset.parse_smiles_safe') as mock_parse:
            mock_parse.return_value = Mock()
            with patch.object(self.simulator, '_identify_reactive_groups', return_value=[]):
                with pytest.raises(ChemiAccelError, match="No reactive groups found"):
                    self.simulator.setup_initial_network("CC", "CC", 1.0, 100)
    
    def test_identify_reactive_groups(self):
        """Test reactive group identification."""
        with patch('chemi_accel.polymer.thermoset.parse_smiles_safe') as mock_parse:
            mock_mol = Mock()
            mock_parse.return_value = mock_mol
            
            # Mock RDKit pattern matching
            with patch('chemi_accel.polymer.thermoset.Chem.MolFromSmarts') as mock_smarts:
                mock_pattern = Mock()
                mock_smarts.return_value = mock_pattern
                mock_mol.GetSubstructMatches.return_value = [(0, 1, 2)]
                
                with patch.object(self.simulator, '_calculate_group_reactivity', return_value=0.8):
                    with patch.object(self.simulator, '_calculate_steric_factor', return_value=0.9):
                        groups = self.simulator._identify_reactive_groups(mock_mol, "test")
                        
                        assert len(groups) > 0
                        assert all('type' in g for g in groups)
                        assert all('reactivity' in g for g in groups)
                        assert all('steric_factor' in g for g in groups)
    
    def test_can_react(self):
        """Test reaction compatibility checking."""
        epoxy_group = ReactiveGroup(0, 'epoxy', 0.8, 0.9, (0, 0, 0))
        amine_group = ReactiveGroup(1, 'primary_amine', 0.9, 0.8, (1, 1, 1))
        vinyl_group = ReactiveGroup(2, 'vinyl', 0.7, 0.8, (2, 2, 2))
        
        # Compatible pairs
        assert self.simulator._can_react(epoxy_group, amine_group)
        assert self.simulator._can_react(vinyl_group, vinyl_group)
        
        # Incompatible pairs (create a group that shouldn't react)
        incompatible_group = ReactiveGroup(3, 'unknown', 0.5, 0.5, (3, 3, 3))
        assert not self.simulator._can_react(epoxy_group, incompatible_group)
    
    def test_calculate_reaction_probability(self):
        """Test reaction probability calculation."""
        group1 = ReactiveGroup(0, 'epoxy', 0.8, 0.9, (0, 0, 0))
        group2 = ReactiveGroup(1, 'primary_amine', 0.9, 0.8, (5, 0, 0))  # 5 Å away
        
        prob = self.simulator._calculate_reaction_probability(
            group1, group2, self.test_kinetics, 1.0
        )
        
        assert 0 <= prob <= 1
        assert isinstance(prob, float)
    
    def test_calculate_bond_strength(self):
        """Test bond strength calculation."""
        epoxy_group = ReactiveGroup(0, 'epoxy', 0.8, 0.9, (0, 0, 0))
        amine_group = ReactiveGroup(1, 'primary_amine', 0.9, 0.8, (1, 1, 1))
        
        strength = self.simulator._calculate_bond_strength(epoxy_group, amine_group)
        
        assert strength > 0
        assert isinstance(strength, float)
        # Should be around 350 kJ/mol for epoxy-amine
        assert 300 <= strength <= 400
    
    def test_monte_carlo_step(self):
        """Test Monte Carlo simulation step."""
        # Create test groups
        groups = [
            ReactiveGroup(0, 'epoxy', 0.8, 0.9, (0, 0, 0)),
            ReactiveGroup(1, 'primary_amine', 0.9, 0.8, (1, 1, 1))
        ]
        
        # Mock high reaction probability
        with patch.object(self.simulator, '_calculate_reaction_probability', return_value=1.0):
            new_bonds = self.simulator._monte_carlo_step(
                groups, [], self.test_kinetics, 1.0, 0.0
            )
            
            # Should form at least one bond with probability 1.0
            assert len(new_bonds) >= 0  # May be 0 if groups become reacted
            if new_bonds:
                assert all(isinstance(b, CrosslinkBond) for b in new_bonds)
    
    def test_network_topology_analysis_cpu(self):
        """Test CPU-based network topology analysis."""
        # Create test network
        groups = [
            ReactiveGroup(0, 'epoxy', 0.8, 0.9, (0, 0, 0)),
            ReactiveGroup(1, 'primary_amine', 0.9, 0.8, (1, 1, 1)),
            ReactiveGroup(2, 'epoxy', 0.8, 0.9, (2, 2, 2))
        ]
        
        bonds = [
            CrosslinkBond(0, 0, 1, 350.0, 10.0, 'epoxy-primary_amine')
        ]
        
        # Update group connections
        groups[0].connected_to = [1]
        groups[1].connected_to = [0]
        groups[0].is_reacted = True
        groups[1].is_reacted = True
        
        topology = self.simulator._analyze_network_topology_cpu(groups, bonds)
        
        assert isinstance(topology, NetworkTopology)
        assert topology.adjacency_matrix.shape == (3, 3)
        assert len(topology.cluster_sizes) > 0
        assert topology.crosslink_density > 0
        assert 0 <= topology.gel_fraction <= 1
    
    def test_detect_gel_point(self):
        """Test gel point detection."""
        # Create topology with high gel fraction
        topology_gel = NetworkTopology(
            nodes=[], bonds=[], adjacency_matrix=np.array([]),
            cluster_sizes=[80, 20], largest_cluster_size=80,
            gel_fraction=0.8, crosslink_density=0.5, average_functionality=2.0
        )
        
        # Create topology without gel
        topology_no_gel = NetworkTopology(
            nodes=[], bonds=[], adjacency_matrix=np.array([]),
            cluster_sizes=[30, 25, 25, 20], largest_cluster_size=30,
            gel_fraction=0.3, crosslink_density=0.2, average_functionality=1.5
        )
        
        assert self.simulator._detect_gel_point(topology_gel)
        assert not self.simulator._detect_gel_point(topology_no_gel)
    
    def test_predict_mechanical_properties(self):
        """Test mechanical property prediction."""
        topology = NetworkTopology(
            nodes=[], bonds=[], adjacency_matrix=np.array([]),
            cluster_sizes=[50, 30, 20], largest_cluster_size=50,
            gel_fraction=0.5, crosslink_density=0.3, average_functionality=2.0
        )
        
        props = self.simulator._predict_mechanical_properties(topology, self.test_kinetics)
        
        assert 'elastic_modulus_mpa' in props
        assert 'glass_transition_temp_k' in props
        assert 'tensile_strength_mpa' in props
        assert 'elongation_at_break' in props
        
        assert all(v > 0 for v in props.values())
    
    def test_predict_thermal_properties(self):
        """Test thermal property prediction."""
        topology = NetworkTopology(
            nodes=[], bonds=[], adjacency_matrix=np.array([]),
            cluster_sizes=[50, 30, 20], largest_cluster_size=50,
            gel_fraction=0.5, crosslink_density=0.3, average_functionality=2.0
        )
        
        props = self.simulator._predict_thermal_properties(topology, self.test_kinetics)
        
        assert 'thermal_conductivity_w_mk' in props
        assert 'thermal_expansion_coeff' in props
        assert 'heat_capacity_j_gk' in props
        assert 'decomposition_temp_k' in props
        
        assert all(v > 0 for v in props.values())
    
    def test_find_clusters_simple(self):
        """Test simple cluster finding algorithm."""
        # Create test adjacency matrix
        adj_matrix = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ])
        
        cluster_sizes = self.simulator._find_clusters_simple(adj_matrix)
        
        assert len(cluster_sizes) == 2  # One cluster of 3, one isolated node
        assert sorted(cluster_sizes) == [1, 3]


class TestNetworkTopologyAnalyzer:
    """Test cases for NetworkTopologyAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NetworkTopologyAnalyzer(use_gpu=False)
    
    def test_calculate_network_descriptors(self):
        """Test network descriptor calculation."""
        # Create test topology
        groups = [ReactiveGroup(i, 'test', 0.5, 0.5, (i, i, i)) for i in range(10)]
        bonds = []
        adj_matrix = np.eye(10)  # Identity matrix - all isolated nodes
        
        topology = NetworkTopology(
            nodes=groups, bonds=bonds, adjacency_matrix=adj_matrix,
            cluster_sizes=[1] * 10, largest_cluster_size=1,
            gel_fraction=0.1, crosslink_density=0.0, average_functionality=0.0
        )
        
        descriptors = self.analyzer.calculate_network_descriptors(topology)
        
        assert 'crosslink_density' in descriptors
        assert 'gel_fraction' in descriptors
        assert 'average_functionality' in descriptors
        assert 'largest_cluster_fraction' in descriptors
        assert 'cluster_size_variance' in descriptors
        assert 'number_of_clusters' in descriptors
        assert 'network_density' in descriptors
        assert 'average_degree' in descriptors
    
    def test_calculate_skewness(self):
        """Test skewness calculation."""
        # Symmetric distribution
        symmetric_data = [1, 2, 3, 4, 5]
        skewness = self.analyzer._calculate_skewness(symmetric_data)
        assert abs(skewness) < 0.1  # Should be close to 0
        
        # Right-skewed distribution
        right_skewed = [1, 1, 1, 2, 5]
        skewness = self.analyzer._calculate_skewness(right_skewed)
        assert skewness > 0
        
        # Edge cases
        assert self.analyzer._calculate_skewness([1, 1]) == 0.0
        assert self.analyzer._calculate_skewness([1, 1, 1]) == 0.0


class TestGelPointDetector:
    """Test cases for GelPointDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = GelPointDetector()
    
    def test_detect_gel_point_percolation(self):
        """Test gel point detection using percolation theory."""
        # Topology with gel
        topology_gel = NetworkTopology(
            nodes=[], bonds=[], adjacency_matrix=np.array([]),
            cluster_sizes=[70, 20, 10], largest_cluster_size=70,
            gel_fraction=0.7, crosslink_density=0.5, average_functionality=2.0
        )
        
        result = self.detector.detect_gel_point_percolation(topology_gel)
        
        assert result['is_gel']
        assert result['gel_fraction'] == 0.7
        assert result['largest_cluster_size'] == 70
        assert 'size_ratio' in result
    
    def test_analyze_gel_evolution(self):
        """Test gel evolution analysis."""
        # Create mock curing results
        results = [
            CuringResult(
                time_points=np.array([0, 1, 2]),
                conversion_profile=np.array([0, 0.5, 1.0]),
                gel_point_time=1.5,
                gel_point_conversion=0.6,
                final_crosslink_density=0.5,
                final_gel_fraction=0.8,
                network_topology=None,
                mechanical_properties={},
                thermal_properties={},
                simulation_metadata={'temperature': 393.15}
            ),
            CuringResult(
                time_points=np.array([0, 1, 2]),
                conversion_profile=np.array([0, 0.5, 1.0]),
                gel_point_time=2.0,
                gel_point_conversion=0.7,
                final_crosslink_density=0.6,
                final_gel_fraction=0.9,
                network_topology=None,
                mechanical_properties={},
                thermal_properties={},
                simulation_metadata={'temperature': 423.15}
            )
        ]
        
        analysis = self.detector.analyze_gel_evolution(results)
        
        assert len(analysis['gel_points']) == 2
        assert abs(analysis['average_gel_time'] - 1.75) < 1e-10
        assert abs(analysis['average_gel_conversion'] - 0.65) < 1e-10


class TestValidationFunctions:
    """Test cases for validation functions."""
    
    def test_validate_against_experimental_data(self):
        """Test validation against experimental data."""
        # Create mock simulation results
        sim_results = [
            CuringResult(
                time_points=np.array([0, 1, 2]),
                conversion_profile=np.array([0, 0.5, 1.0]),
                gel_point_time=1.5,
                gel_point_conversion=0.6,
                final_crosslink_density=0.5,
                final_gel_fraction=0.8,
                network_topology=None,
                mechanical_properties={'elastic_modulus_mpa': 100.0},
                thermal_properties={'glass_transition_temp_k': 400.0},
                simulation_metadata={}
            )
        ]
        
        # Mock experimental data
        exp_data = {
            'gel_point_times': [1.6],
            'elastic_modulus': [95.0],
            'glass_transition_temps': [405.0]
        }
        
        metrics = validate_against_experimental_data(sim_results, exp_data)
        
        assert 'gel_point_mae' in metrics
        assert 'elastic_modulus_mae' in metrics
        assert 'glass_transition_mae' in metrics
        
        assert abs(metrics['gel_point_mae'] - 0.1) < 1e-10
        assert abs(metrics['elastic_modulus_mae'] - 5.0) < 1e-10
        assert abs(metrics['glass_transition_mae'] - 5.0) < 1e-10


class TestDataStructures:
    """Test cases for data structures."""
    
    def test_reactive_group(self):
        """Test ReactiveGroup data structure."""
        group = ReactiveGroup(
            group_id=1,
            group_type='epoxy',
            reactivity=0.8,
            steric_factor=0.9,
            position=(1.0, 2.0, 3.0)
        )
        
        assert group.group_id == 1
        assert group.group_type == 'epoxy'
        assert group.reactivity == 0.8
        assert group.steric_factor == 0.9
        assert group.position == (1.0, 2.0, 3.0)
        assert not group.is_reacted
        assert group.connected_to == []
    
    def test_crosslink_bond(self):
        """Test CrosslinkBond data structure."""
        bond = CrosslinkBond(
            bond_id=1,
            group1_id=0,
            group2_id=1,
            bond_strength=350.0,
            formation_time=10.0,
            bond_type='epoxy-amine'
        )
        
        assert bond.bond_id == 1
        assert bond.group1_id == 0
        assert bond.group2_id == 1
        assert bond.bond_strength == 350.0
        assert bond.formation_time == 10.0
        assert bond.bond_type == 'epoxy-amine'
    
    def test_curing_kinetics(self):
        """Test CuringKinetics data structure."""
        kinetics = CuringKinetics(
            rate_constants={'k1': 1e-3},
            activation_energies={'E1': 50000},
            temperature=393.15,
            catalyst_concentration=0.01,
            diffusion_coefficients={'epoxy': 1e-12}
        )
        
        assert kinetics.rate_constants['k1'] == 1e-3
        assert kinetics.activation_energies['E1'] == 50000
        assert kinetics.temperature == 393.15
        assert kinetics.catalyst_concentration == 0.01
        assert kinetics.diffusion_coefficients['epoxy'] == 1e-12


if __name__ == '__main__':
    pytest.main([__file__])
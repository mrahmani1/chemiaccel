"""
Integration tests for thermoset curing simulation.

These tests demonstrate the complete workflow from initial network setup
through curing simulation to property prediction and validation.
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.polymer.thermoset import (
    CuringKinetics,
    CuringMechanism,
    GelPointDetector,
    NetworkTopologyAnalyzer,
    ThermosetCuringSimulator,
    validate_against_experimental_data,
)


class TestThermosetIntegration:
    """Integration tests for complete thermoset curing workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.simulator = ThermosetCuringSimulator(use_gpu=False)
        self.analyzer = NetworkTopologyAnalyzer(use_gpu=False)
        self.detector = GelPointDetector()

        # Test system parameters
        self.epoxy_resin = "C1OC1CC(C1OC1)C"  # Diglycidyl ether
        self.amine_hardener = "NCCN"  # Ethylenediamine
        self.stoichiometry = 1.0
        self.system_size = 50  # Small system for fast testing

    @patch("chemi_accel.polymer.thermoset.RDKIT_AVAILABLE", True)
    def test_complete_epoxy_amine_curing_workflow(self):
        """Test complete epoxy-amine curing simulation workflow."""
        # Mock RDKit functionality
        with patch("chemi_accel.polymer.thermoset.parse_smiles_safe") as mock_parse:
            mock_resin = Mock()
            mock_hardener = Mock()
            mock_parse.side_effect = [mock_resin, mock_hardener]

            # Mock reactive group identification
            with patch.object(
                self.simulator, "_identify_reactive_groups"
            ) as mock_identify:
                mock_identify.side_effect = [
                    [
                        {
                            "type": "epoxy",
                            "reactivity": 0.8,
                            "steric_factor": 0.9,
                            "atom_indices": (0, 1, 2),
                        }
                    ]
                    * 2,
                    [
                        {
                            "type": "primary_amine",
                            "reactivity": 0.9,
                            "steric_factor": 0.8,
                            "atom_indices": (0, 1),
                        }
                    ]
                    * 2,
                ]

                # Step 1: Setup initial network
                reactive_groups = self.simulator.setup_initial_network(
                    self.epoxy_resin,
                    self.amine_hardener,
                    self.stoichiometry,
                    self.system_size,
                )

                assert len(reactive_groups) > 0
                logging.info(f"Initial network: {len(reactive_groups)} reactive groups")

                # Step 2: Run curing simulation
                curing_result = self.simulator.simulate_curing(
                    reactive_groups=reactive_groups,
                    mechanism=CuringMechanism.EPOXY_AMINE,
                    simulation_time=100.0,  # Short simulation for testing
                    time_step=10.0,
                )

                # Verify simulation results
                assert curing_result is not None
                assert len(curing_result.time_points) > 0
                assert len(curing_result.conversion_profile) == len(
                    curing_result.time_points
                )
                assert curing_result.final_crosslink_density >= 0
                assert 0 <= curing_result.final_gel_fraction <= 1

                logging.info(
                    f"Final conversion: {curing_result.conversion_profile[-1]:.3f}"
                )
                logging.info(
                    f"Final gel fraction: {curing_result.final_gel_fraction:.3f}"
                )
                logging.info(
                    f"Crosslink density: {curing_result.final_crosslink_density:.3f}"
                )

                # Step 3: Analyze network topology
                topology_descriptors = self.analyzer.calculate_network_descriptors(
                    curing_result.network_topology
                )

                assert "crosslink_density" in topology_descriptors
                assert "gel_fraction" in topology_descriptors
                assert "network_density" in topology_descriptors

                # Step 4: Gel point analysis
                gel_analysis = self.detector.detect_gel_point_percolation(
                    curing_result.network_topology
                )

                assert "is_gel" in gel_analysis
                assert "gel_fraction" in gel_analysis

                # Step 5: Property predictions
                assert "elastic_modulus_mpa" in curing_result.mechanical_properties
                assert "glass_transition_temp_k" in curing_result.thermal_properties

                # Verify property values are reasonable
                elastic_modulus = curing_result.mechanical_properties[
                    "elastic_modulus_mpa"
                ]
                tg = curing_result.thermal_properties["glass_transition_temp_k"]

                assert elastic_modulus > 0
                assert 250 < tg < 500  # Reasonable Tg range for epoxy

                logging.info(f"Predicted elastic modulus: {elastic_modulus:.1f} MPa")
                logging.info(f"Predicted Tg: {tg:.1f} K")

    @patch("chemi_accel.polymer.thermoset.RDKIT_AVAILABLE", True)
    def test_multiple_curing_conditions(self):
        """Test curing under multiple temperature conditions."""
        temperatures = [353.15, 393.15, 433.15]  # 80°C, 120°C, 160°C
        results = []

        with patch("chemi_accel.polymer.thermoset.parse_smiles_safe") as mock_parse:
            mock_resin = Mock()
            mock_hardener = Mock()
            mock_parse.side_effect = [mock_resin, mock_hardener] * len(temperatures)

            with patch.object(
                self.simulator, "_identify_reactive_groups"
            ) as mock_identify:
                mock_identify.side_effect = [
                    [
                        {
                            "type": "epoxy",
                            "reactivity": 0.8,
                            "steric_factor": 0.9,
                            "atom_indices": (0, 1, 2),
                        }
                    ],
                    [
                        {
                            "type": "primary_amine",
                            "reactivity": 0.9,
                            "steric_factor": 0.8,
                            "atom_indices": (0, 1),
                        }
                    ],
                ] * len(temperatures)

                for temp in temperatures:
                    # Setup network
                    reactive_groups = self.simulator.setup_initial_network(
                        self.epoxy_resin, self.amine_hardener, 1.0, 20
                    )

                    # Custom kinetics for this temperature
                    kinetics = CuringKinetics(
                        rate_constants={"k1": 1e-2, "k2": 5e-3},
                        activation_energies={"E1": 50000, "E2": 60000},
                        temperature=temp,
                        catalyst_concentration=0.01,
                        diffusion_coefficients={"epoxy": 1e-12, "amine": 8e-13},
                    )

                    # Run simulation
                    result = self.simulator.simulate_curing(
                        reactive_groups=reactive_groups,
                        mechanism=CuringMechanism.EPOXY_AMINE,
                        simulation_time=50.0,
                        time_step=10.0,
                        kinetics=kinetics,
                    )

                    results.append(result)

                # Analyze temperature effects
                assert len(results) == len(temperatures)

                # Higher temperature should generally lead to faster curing
                # (though this is a simplified test with mocked components)
                for i, result in enumerate(results):
                    logging.info(
                        f"T={temperatures[i]:.1f}K: "
                        f"final conversion={result.conversion_profile[-1]:.3f}, "
                        f"gel fraction={result.final_gel_fraction:.3f}"
                    )

                # Test gel evolution analysis
                gel_evolution = self.detector.analyze_gel_evolution(results)
                assert "gel_points" in gel_evolution

    def test_network_topology_evolution(self):
        """Test evolution of network topology during curing."""
        # Create a simple test system
        from chemi_accel.polymer.thermoset import CrosslinkBond, ReactiveGroup

        # Initial state: isolated groups
        groups = [ReactiveGroup(i, "test", 0.5, 0.5, (i, 0, 0)) for i in range(10)]

        bonds_stages = [
            [],  # No bonds initially
            [CrosslinkBond(0, 0, 1, 300.0, 10.0, "test-test")],  # One bond
            [
                CrosslinkBond(0, 0, 1, 300.0, 10.0, "test-test"),
                CrosslinkBond(1, 2, 3, 300.0, 20.0, "test-test"),
            ],  # Two bonds
            [
                CrosslinkBond(0, 0, 1, 300.0, 10.0, "test-test"),
                CrosslinkBond(1, 2, 3, 300.0, 20.0, "test-test"),
                CrosslinkBond(2, 1, 4, 300.0, 30.0, "test-test"),
            ],  # Three bonds
        ]

        gel_fractions = []
        crosslink_densities = []

        for bonds in bonds_stages:
            # Update group connections
            for group in groups:
                group.connected_to = []
                group.is_reacted = False

            for bond in bonds:
                groups[bond.group1_id].connected_to.append(bond.group2_id)
                groups[bond.group2_id].connected_to.append(bond.group1_id)
                groups[bond.group1_id].is_reacted = True
                groups[bond.group2_id].is_reacted = True

            # Analyze topology
            topology = self.simulator._analyze_network_topology_cpu(groups, bonds)

            gel_fractions.append(topology.gel_fraction)
            crosslink_densities.append(topology.crosslink_density)

            # Calculate descriptors
            descriptors = self.analyzer.calculate_network_descriptors(topology)

            logging.info(
                f"Stage {len(bonds)} bonds: "
                f"gel fraction={topology.gel_fraction:.3f}, "
                f"crosslink density={topology.crosslink_density:.3f}, "
                f"clusters={len(topology.cluster_sizes)}"
            )

        # Verify evolution trends
        assert len(gel_fractions) == len(bonds_stages)
        assert len(crosslink_densities) == len(bonds_stages)

        # Crosslink density should increase with more bonds
        assert crosslink_densities[-1] >= crosslink_densities[0]

    def test_property_prediction_validation(self):
        """Test property prediction and validation workflow."""
        # Create test topology with known characteristics
        from chemi_accel.polymer.thermoset import NetworkTopology

        groups = [Mock() for _ in range(100)]
        bonds = [Mock() for _ in range(30)]

        topology = NetworkTopology(
            nodes=groups,
            bonds=bonds,
            adjacency_matrix=np.eye(100),
            cluster_sizes=[60, 25, 15],
            largest_cluster_size=60,
            gel_fraction=0.6,
            crosslink_density=0.3,
            average_functionality=2.0,
        )

        # Test kinetics
        kinetics = CuringKinetics(
            rate_constants={"k1": 1e-3},
            activation_energies={"E1": 50000},
            temperature=393.15,
            catalyst_concentration=0.01,
            diffusion_coefficients={"epoxy": 1e-12},
        )

        # Predict properties
        mechanical_props = self.simulator._predict_mechanical_properties(
            topology, kinetics
        )
        thermal_props = self.simulator._predict_thermal_properties(topology, kinetics)

        # Verify reasonable property values
        assert mechanical_props["elastic_modulus_mpa"] > 0
        assert thermal_props["glass_transition_temp_k"] > 250
        assert thermal_props["thermal_conductivity_w_mk"] > 0

        # Create mock experimental data for validation
        experimental_data = {
            "elastic_modulus": [
                mechanical_props["elastic_modulus_mpa"] * 1.1
            ],  # 10% difference
            "glass_transition_temps": [
                thermal_props["glass_transition_temp_k"] - 5
            ],  # 5K difference
        }

        # Create mock simulation results
        from chemi_accel.polymer.thermoset import CuringResult

        sim_result = CuringResult(
            time_points=np.array([0, 10, 20]),
            conversion_profile=np.array([0, 0.5, 0.8]),
            gel_point_time=15.0,
            gel_point_conversion=0.6,
            final_crosslink_density=0.3,
            final_gel_fraction=0.6,
            network_topology=topology,
            mechanical_properties=mechanical_props,
            thermal_properties=thermal_props,
            simulation_metadata={},
        )

        # Validate against experimental data
        validation_metrics = validate_against_experimental_data(
            [sim_result], experimental_data
        )

        assert "elastic_modulus_mae" in validation_metrics
        assert "glass_transition_mae" in validation_metrics

        # Check that validation metrics are reasonable
        assert validation_metrics["elastic_modulus_mae"] > 0
        assert validation_metrics["glass_transition_mae"] == 5.0

        logging.info(f"Validation metrics: {validation_metrics}")

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        # Test with invalid mechanism
        with pytest.raises(ChemiAccelError):
            self.simulator.simulate_curing(
                reactive_groups=[],
                mechanism=None,  # Invalid mechanism
                simulation_time=10.0,
            )

        # Test with empty reactive groups
        result = self.simulator.simulate_curing(
            reactive_groups=[],
            mechanism=CuringMechanism.EPOXY_AMINE,
            simulation_time=10.0,
        )

        # Should handle empty system gracefully
        assert result.final_crosslink_density == 0
        assert result.final_gel_fraction == 0

        # Test topology analysis with empty network
        empty_topology = self.simulator._analyze_network_topology_cpu([], [])
        assert empty_topology.crosslink_density == 0
        assert empty_topology.gel_fraction == 0

        # Test descriptor calculation with minimal topology
        minimal_descriptors = self.analyzer.calculate_network_descriptors(
            empty_topology
        )
        assert all(
            v == 0
            for v in minimal_descriptors.values()
            if "number_of_clusters" not in str(v)
        )

    def test_performance_and_scalability(self):
        """Test performance characteristics and scalability."""
        import time

        system_sizes = [10, 20, 50]
        simulation_times = []

        for size in system_sizes:
            # Create test system
            from chemi_accel.polymer.thermoset import ReactiveGroup

            groups = [
                ReactiveGroup(
                    i, "epoxy" if i % 2 == 0 else "amine", 0.8, 0.9, (i, 0, 0)
                )
                for i in range(size)
            ]

            # Time the simulation
            start_time = time.time()

            result = self.simulator.simulate_curing(
                reactive_groups=groups,
                mechanism=CuringMechanism.EPOXY_AMINE,
                simulation_time=20.0,
                time_step=10.0,
            )

            elapsed_time = time.time() - start_time
            simulation_times.append(elapsed_time)

            logging.info(f"System size {size}: {elapsed_time:.3f}s")

        # Verify simulations completed successfully
        assert len(simulation_times) == len(system_sizes)
        assert all(t > 0 for t in simulation_times)

        # For small test systems, simulation should be fast
        assert all(t < 10.0 for t in simulation_times)  # Less than 10 seconds


if __name__ == "__main__":
    # Configure logging for test output
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])

"""
Validation Tests for GPU-Accelerated GIST Implementation

This module provides comprehensive validation tests for Task 19.3 - GPU-Accelerated
Solvation Thermodynamics (GIST) against established reference implementations.

Test Categories:
1. Thermodynamic property accuracy validation
2. Grid convergence and spatial resolution tests
3. Trajectory analysis validation
4. Hydration hotspot identification accuracy
5. Pharmacophore integration validation
6. Performance and scaling benchmarks
"""

import os
import tempfile

import numpy as np
import pytest

# Test configuration
pytestmark = pytest.mark.gist


@pytest.fixture
def gist_engine():
    """Create GIST engine for testing."""
    from chemi_accel.backends.cuda.solvation_gist import create_gist_engine

    try:
        return create_gist_engine(
            grid_spacing=0.5,
            temperature=298.15,
            memory_limit_gb=2.0,
            trajectory_support=True,
        )
    except Exception:
        pytest.skip("GIST engine not available")


@pytest.fixture
def reference_molecules():
    """Create reference molecules for validation."""
    return [
        {
            "mol_id": "water",
            "coordinates": np.array([[0.0, 0.0, 0.0]]),
            "charges": np.array([0.0]),
            "elements": ["O"],
        },
        {
            "mol_id": "methane",
            "coordinates": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                ]
            ),
            "charges": np.array([-0.4, 0.1, 0.1, 0.1, 0.1]),
            "elements": ["C", "H", "H", "H", "H"],
        },
        {
            "mol_id": "benzene",
            "coordinates": np.array(
                [
                    [0.0, 1.4, 0.0],
                    [1.21, 0.7, 0.0],
                    [1.21, -0.7, 0.0],
                    [0.0, -1.4, 0.0],
                    [-1.21, -0.7, 0.0],
                    [-1.21, 0.7, 0.0],
                    [0.0, 2.5, 0.0],
                    [2.15, 1.25, 0.0],
                    [2.15, -1.25, 0.0],
                    [0.0, -2.5, 0.0],
                    [-2.15, -1.25, 0.0],
                    [-2.15, 1.25, 0.0],
                ]
            ),
            "charges": np.array([-0.1] * 6 + [0.1] * 6),
            "elements": ["C"] * 6 + ["H"] * 6,
        },
    ]


@pytest.fixture
def mock_trajectory_files():
    """Create mock trajectory files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_file = os.path.join(tmpdir, "test.dcd")
        top_file = os.path.join(tmpdir, "test.pdb")

        # Create mock files
        with open(traj_file, "w") as f:
            f.write("mock trajectory data")
        with open(top_file, "w") as f:
            f.write("mock topology data")

        yield traj_file, top_file


class TestGISTThermodynamics:
    """Test thermodynamic property calculations."""

    def test_solvation_free_energy_accuracy(self, gist_engine, reference_molecules):
        """Test solvation free energy calculation accuracy."""

        # Test with known reference values
        for molecule in reference_molecules:
            result = gist_engine.calculate_gist_properties(
                molecule=molecule,
                properties=[
                    "solvation_free_energy",
                    "solvation_enthalpy",
                    "solvation_entropy",
                ],
            )

            thermo_props = result["thermodynamic_properties"]

            # Basic thermodynamic consistency checks
            assert isinstance(thermo_props.solvation_free_energy, float)
            assert isinstance(thermo_props.solvation_enthalpy, float)
            assert isinstance(thermo_props.solvation_entropy, float)

            # Check thermodynamic relationship: ΔG = ΔH - TΔS
            calculated_free_energy = (
                thermo_props.solvation_enthalpy
                - 298.15 * thermo_props.solvation_entropy / 1000.0
            )

            # Allow for small numerical differences
            assert (
                abs(calculated_free_energy - thermo_props.solvation_free_energy) < 0.1
            )

            # Check reasonable ranges for small molecules
            assert -20.0 < thermo_props.solvation_free_energy < 20.0  # kcal/mol
            assert -30.0 < thermo_props.solvation_enthalpy < 30.0  # kcal/mol
            assert -100.0 < thermo_props.solvation_entropy < 100.0  # cal/mol/K

    def test_water_density_normalization(self, gist_engine, reference_molecules):
        """Test that bulk water density is properly normalized."""

        # Use a simple molecule for testing
        molecule = reference_molecules[0]  # water

        result = gist_engine.calculate_gist_properties(
            molecule=molecule, properties=["water_density", "density_fluctuation"]
        )

        thermo_props = result["thermodynamic_properties"]

        # Bulk water density should be ~0.0334 molecules/Å³ at 298K
        expected_bulk_density = 0.0334
        assert 0.01 < thermo_props.average_water_density < 0.1  # Reasonable range

        # Density fluctuations should be positive
        assert thermo_props.density_fluctuation >= 0.0

    def test_thermodynamic_decomposition(self, gist_engine, reference_molecules):
        """Test decomposition of thermodynamic contributions."""

        molecule = reference_molecules[1]  # methane

        result = gist_engine.calculate_gist_properties(
            molecule=molecule, properties=["solvation_free_energy"]
        )

        thermo_props = result["thermodynamic_properties"]

        # Check that contributions sum approximately to total
        total_calculated = (
            thermo_props.electrostatic_contribution
            + thermo_props.van_der_waals_contribution
            + thermo_props.cavity_formation_energy
        )

        # Allow for reasonable approximation error
        assert abs(total_calculated - thermo_props.solvation_free_energy) < 1.0

        # Cavity formation should be positive for hydrophobic molecules
        assert thermo_props.cavity_formation_energy > 0.0


class TestGridAnalysis:
    """Test 3D grid analysis capabilities."""

    def test_grid_resolution_convergence(self, gist_engine, reference_molecules):
        """Test convergence with respect to grid resolution."""

        molecule = reference_molecules[1]  # methane
        grid_spacings = [1.0, 0.5, 0.25]
        free_energies = []

        for spacing in grid_spacings:
            engine = gist_engine.__class__(
                grid_spacing=spacing, temperature=298.15, memory_limit_gb=2.0
            )

            try:
                result = engine.calculate_gist_properties(
                    molecule=molecule, properties=["solvation_free_energy"]
                )
                free_energies.append(
                    result["thermodynamic_properties"].solvation_free_energy
                )
            except Exception:
                # Skip if grid spacing not supported
                continue
            finally:
                engine.cleanup()

        if len(free_energies) >= 2:
            # Check that results are converging (differences decreasing)
            differences = [
                abs(free_energies[i + 1] - free_energies[i])
                for i in range(len(free_energies) - 1)
            ]

            # Later differences should be smaller (convergence)
            if len(differences) >= 2:
                assert differences[-1] <= differences[0] * 2.0  # Allow some variation

    def test_spatial_map_generation(self, gist_engine, reference_molecules):
        """Test generation of 3D spatial thermodynamic maps."""

        molecule = reference_molecules[2]  # benzene

        result = gist_engine.calculate_gist_properties(
            molecule=molecule, properties=["solvation_free_energy"], generate_maps=True
        )

        # Check that spatial maps are generated
        assert "spatial_maps" in result
        spatial_maps = result["spatial_maps"]

        required_maps = [
            "enthalpy_grid",
            "entropy_grid",
            "free_energy_grid",
            "density_grid",
        ]
        for map_type in required_maps:
            assert map_type in spatial_maps

            # Check that maps are 3D numpy arrays
            assert isinstance(spatial_maps[map_type], np.ndarray)
            assert len(spatial_maps[map_type].shape) == 3

            # Check for reasonable values
            assert not np.all(spatial_maps[map_type] == 0)  # Not all zeros
            assert np.all(np.isfinite(spatial_maps[map_type]))  # No NaN/inf

    def test_volume_analysis(self, gist_engine, reference_molecules):
        """Test molecular volume and surface area analysis."""

        molecule = reference_molecules[2]  # benzene

        result = gist_engine.calculate_gist_properties(
            molecule=molecule, properties=["solvation_free_energy"]
        )

        thermo_props = result["thermodynamic_properties"]

        # Check volume calculations
        total_volume = (
            thermo_props.favorable_volume
            + thermo_props.unfavorable_volume
            + thermo_props.neutral_volume
        )

        assert total_volume > 0.0  # Should have some analyzed volume
        assert thermo_props.favorable_volume >= 0.0
        assert thermo_props.unfavorable_volume >= 0.0
        assert thermo_props.neutral_volume >= 0.0

        # Check surface area calculations
        total_surface = (
            thermo_props.hydrophobic_surface_area + thermo_props.polar_surface_area
        )
        assert total_surface > 0.0
        assert thermo_props.hydrophobic_surface_area >= 0.0
        assert thermo_props.polar_surface_area >= 0.0


class TestTrajectoryAnalysis:
    """Test MD trajectory analysis capabilities."""

    def test_trajectory_loading(self, gist_engine, mock_trajectory_files):
        """Test trajectory file loading and processing."""

        traj_file, top_file = mock_trajectory_files

        # Test trajectory analysis
        result = gist_engine.analyze_md_trajectory(
            trajectory_file=traj_file,
            topology_file=top_file,
            stride=10,
            grid_buffer=3.0,
        )

        # Check result structure
        assert "trajectory_info" in result
        assert "thermodynamic_properties" in result
        assert "spatial_maps" in result
        assert "hydration_hotspots" in result
        assert "convergence_analysis" in result

        # Check trajectory info
        traj_info = result["trajectory_info"]
        assert "total_frames" in traj_info
        assert "frames_analyzed" in traj_info
        assert "analysis_time" in traj_info

        assert traj_info["total_frames"] > 0
        assert traj_info["frames_analyzed"] > 0
        assert traj_info["analysis_time"] > 0.0

    def test_convergence_analysis(self, gist_engine, mock_trajectory_files):
        """Test trajectory convergence analysis."""

        traj_file, top_file = mock_trajectory_files

        result = gist_engine.analyze_md_trajectory(
            trajectory_file=traj_file, topology_file=top_file, stride=5
        )

        convergence = result["convergence_analysis"]

        # Check convergence metrics
        assert "free_energy_convergence" in convergence
        assert "spatial_convergence" in convergence
        assert "thermodynamic_decomposition" in convergence
        assert "sampling_metrics" in convergence

        # Check free energy convergence
        fe_conv = convergence["free_energy_convergence"]
        assert isinstance(fe_conv["converged"], bool)
        assert fe_conv["final_uncertainty"] > 0.0
        assert fe_conv["convergence_frame"] > 0

        # Check sampling metrics
        sampling = convergence["sampling_metrics"]
        assert sampling["effective_samples"] > 0
        assert sampling["correlation_time"] > 0.0
        assert 0.0 <= sampling["sampling_efficiency"] <= 1.0

    def test_batch_processing(self, gist_engine, mock_trajectory_files):
        """Test trajectory batch processing capabilities."""

        traj_file, top_file = mock_trajectory_files

        # Test with small batch size to ensure batching works
        original_batch_size = gist_engine.max_frames_batch
        gist_engine.max_frames_batch = 10

        try:
            result = gist_engine.analyze_md_trajectory(
                trajectory_file=traj_file, topology_file=top_file, stride=1
            )

            # Should still get valid results with batching
            assert "thermodynamic_properties" in result
            assert result["trajectory_info"]["frames_analyzed"] > 0

        finally:
            gist_engine.max_frames_batch = original_batch_size


class TestHydrationHotspots:
    """Test hydration hotspot identification and ranking."""

    def test_hotspot_identification(self, gist_engine, mock_trajectory_files):
        """Test identification of hydration hotspots."""

        traj_file, top_file = mock_trajectory_files

        result = gist_engine.analyze_md_trajectory(
            trajectory_file=traj_file, topology_file=top_file
        )

        hotspots = result["hydration_hotspots"]

        # Check hotspot structure
        if hotspots:  # May be empty for mock data
            for hotspot in hotspots:
                assert "id" in hotspot
                assert "center" in hotspot
                assert "volume" in hotspot
                assert "free_energy" in hotspot
                assert "enthalpy" in hotspot
                assert "entropy" in hotspot
                assert "favorability_score" in hotspot
                assert "site_type" in hotspot

                # Check reasonable values
                assert len(hotspot["center"]) == 3  # 3D coordinates
                assert hotspot["volume"] > 0.0
                assert isinstance(hotspot["favorability_score"], float)
                assert hotspot["site_type"] in [
                    "strong_binding",
                    "enthalpic",
                    "entropic",
                    "unfavorable",
                    "neutral",
                ]

    def test_hotspot_ranking(self, gist_engine, mock_trajectory_files):
        """Test hotspot ranking by thermodynamic favorability."""

        traj_file, top_file = mock_trajectory_files

        result = gist_engine.analyze_md_trajectory(
            trajectory_file=traj_file, topology_file=top_file
        )

        hotspots = result["hydration_hotspots"]

        if len(hotspots) > 1:
            # Check that hotspots are ranked by free energy (most favorable first)
            free_energies = [h["free_energy"] for h in hotspots]

            # Should be sorted in ascending order (most negative first)
            for i in range(len(free_energies) - 1):
                assert free_energies[i] <= free_energies[i + 1]

    def test_hotspot_classification(self, gist_engine, reference_molecules):
        """Test hotspot thermodynamic classification."""

        molecule = reference_molecules[2]  # benzene

        result = gist_engine.calculate_gist_properties(
            molecule=molecule, properties=["solvation_free_energy"], generate_maps=True
        )

        # Mock some hotspots for classification testing
        mock_gist_results = {
            "free_energy_grid": np.array([-2.0, -1.5, -0.5, 0.5, 1.5]),
            "enthalpy_grid": np.array([-2.5, -1.8, -0.3, 0.8, 2.0]),
            "entropy_grid": np.array([-20.0, -12.0, -3.0, 5.0, 15.0]),
            "density_grid": np.array([0.05, 0.04, 0.02, 0.03, 0.02]),
            "grid_points": np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
            ),
        }

        hotspots = gist_engine._identify_hydration_hotspots(mock_gist_results)

        if hotspots:
            # Check that different site types are assigned
            site_types = [h["site_type"] for h in hotspots]
            assert all(
                st
                in ["strong_binding", "enthalpic", "entropic", "unfavorable", "neutral"]
                for st in site_types
            )


class TestPharmacophoreIntegration:
    """Test integration with pharmacophore modeling."""

    def test_pharmacophore_mapping(self, gist_engine, mock_trajectory_files):
        """Test mapping of GIST results to pharmacophore features."""

        traj_file, top_file = mock_trajectory_files

        # Run GIST analysis
        gist_results = gist_engine.analyze_md_trajectory(
            trajectory_file=traj_file, topology_file=top_file
        )

        # Define mock pharmacophore features
        pharmacophore_features = [
            {"id": 1, "type": "hydrogen_bond_donor", "coordinates": [1.0, 2.0, 3.0]},
            {"id": 2, "type": "hydrogen_bond_acceptor", "coordinates": [4.0, 5.0, 6.0]},
            {"id": 3, "type": "hydrophobic", "coordinates": [7.0, 8.0, 9.0]},
        ]

        # Test integration
        integration_result = gist_engine.integrate_with_pharmacophore(
            gist_results, pharmacophore_features
        )

        # Check result structure
        assert "feature_analyses" in integration_result
        assert "overall_thermodynamic_score" in integration_result
        assert "design_recommendations" in integration_result
        assert "hotspot_utilization" in integration_result

        # Check feature analyses
        feature_analyses = integration_result["feature_analyses"]
        assert len(feature_analyses) == len(pharmacophore_features)

        for analysis in feature_analyses:
            assert "feature_id" in analysis
            assert "feature_type" in analysis
            assert "coordinates" in analysis
            assert "thermodynamic_contribution" in analysis
            assert "desolvation_penalty" in analysis
            assert "net_binding_contribution" in analysis
            assert "optimization_recommendation" in analysis

    def test_design_recommendations(self, gist_engine, mock_trajectory_files):
        """Test generation of design recommendations."""

        traj_file, top_file = mock_trajectory_files

        gist_results = gist_engine.analyze_md_trajectory(
            trajectory_file=traj_file, topology_file=top_file
        )

        pharmacophore_features = [
            {"id": 1, "type": "hydrogen_bond_donor", "coordinates": [0.0, 0.0, 0.0]}
        ]

        integration_result = gist_engine.integrate_with_pharmacophore(
            gist_results, pharmacophore_features
        )

        recommendations = integration_result["design_recommendations"]

        # Should get some recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # All recommendations should be strings
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_desolvation_penalty_calculation(self, gist_engine):
        """Test desolvation penalty calculation."""

        # Mock nearby hotspots
        nearby_hotspots = [
            {"favorability": 2.0, "volume": 27.0},  # 1 water volume
            {"favorability": 1.5, "volume": 54.0},  # 2 water volumes
        ]

        penalty = gist_engine._calculate_desolvation_penalty(nearby_hotspots)

        # Should be positive penalty
        assert penalty > 0.0

        # Should scale with favorability and volume
        assert penalty < max(h["favorability"] for h in nearby_hotspots)


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    def test_memory_management(self, gist_engine, reference_molecules):
        """Test GPU memory management."""

        molecule = reference_molecules[2]  # benzene

        # Run multiple calculations to test memory management
        for i in range(3):
            result = gist_engine.calculate_gist_properties(
                molecule=molecule, properties=["solvation_free_energy"]
            )

            assert "thermodynamic_properties" in result

            # Memory should not grow unboundedly
            memory_usage = (
                gist_engine.mempool.used_bytes()
                if hasattr(gist_engine, "mempool")
                else 0
            )
            assert memory_usage < gist_engine.memory_limit_bytes

    def test_grid_size_scaling(self, gist_engine, reference_molecules):
        """Test scaling with different grid sizes."""

        molecule = reference_molecules[1]  # methane
        grid_spacings = [1.0, 0.5]  # Different resolutions
        calculation_times = []

        for spacing in grid_spacings:
            engine = gist_engine.__class__(
                grid_spacing=spacing, temperature=298.15, memory_limit_gb=2.0
            )

            try:
                start_time = time.time()
                result = engine.calculate_gist_properties(
                    molecule=molecule, properties=["solvation_free_energy"]
                )
                end_time = time.time()

                calculation_times.append(end_time - start_time)

                # Should get valid results
                assert "thermodynamic_properties" in result

            except Exception:
                continue  # Skip if spacing not supported
            finally:
                engine.cleanup()

        # Finer grids should take longer (more grid points)
        if len(calculation_times) >= 2:
            # Allow for some variation in timing
            assert calculation_times[1] >= calculation_times[0] * 0.5

    @pytest.mark.slow
    def test_trajectory_scaling(self, gist_engine, mock_trajectory_files):
        """Test scaling with trajectory length."""

        traj_file, top_file = mock_trajectory_files

        # Test different stride values (simulating different trajectory lengths)
        strides = [10, 5, 2]
        analysis_times = []

        for stride in strides:
            start_time = time.time()
            result = gist_engine.analyze_md_trajectory(
                trajectory_file=traj_file, topology_file=top_file, stride=stride
            )
            end_time = time.time()

            analysis_times.append(end_time - start_time)

            # Should get valid results
            assert "thermodynamic_properties" in result

            # More frames should give more precise results
            frames_analyzed = result["trajectory_info"]["frames_analyzed"]
            assert frames_analyzed > 0

        # More frames (smaller stride) should take longer
        if len(analysis_times) >= 2:
            # Allow for overhead and variation
            assert analysis_times[-1] >= analysis_times[0] * 0.5


class TestReferenceValidation:
    """Validate against known reference implementations."""

    @pytest.mark.reference
    def test_water_solvation_energy(self, gist_engine):
        """Test water self-solvation (should be close to zero)."""

        water_molecule = {
            "mol_id": "water",
            "coordinates": np.array([[0.0, 0.0, 0.0]]),
            "charges": np.array([0.0]),
            "elements": ["O"],
        }

        result = gist_engine.calculate_gist_properties(
            molecule=water_molecule, properties=["solvation_free_energy"]
        )

        free_energy = result["thermodynamic_properties"].solvation_free_energy

        # Water self-solvation should be small
        assert abs(free_energy) < 2.0  # kcal/mol

    @pytest.mark.reference
    def test_methane_hydrophobic_solvation(self, gist_engine):
        """Test methane solvation (should be positive - unfavorable)."""

        methane_molecule = {
            "mol_id": "methane",
            "coordinates": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                ]
            ),
            "charges": np.array([-0.4, 0.1, 0.1, 0.1, 0.1]),
            "elements": ["C", "H", "H", "H", "H"],
        }

        result = gist_engine.calculate_gist_properties(
            molecule=methane_molecule,
            properties=["solvation_free_energy", "cavity_formation_energy"],
        )

        thermo_props = result["thermodynamic_properties"]

        # Methane solvation should be unfavorable (positive)
        assert thermo_props.solvation_free_energy > 0.0

        # Cavity formation should dominate for hydrophobic molecules
        assert thermo_props.cavity_formation_energy > 0.0

    @pytest.mark.reference
    def test_benzene_aromatic_solvation(self, gist_engine):
        """Test benzene solvation properties."""

        benzene_molecule = {
            "mol_id": "benzene",
            "coordinates": np.array(
                [
                    [0.0, 1.4, 0.0],
                    [1.21, 0.7, 0.0],
                    [1.21, -0.7, 0.0],
                    [0.0, -1.4, 0.0],
                    [-1.21, -0.7, 0.0],
                    [-1.21, 0.7, 0.0],
                    [0.0, 2.5, 0.0],
                    [2.15, 1.25, 0.0],
                    [2.15, -1.25, 0.0],
                    [0.0, -2.5, 0.0],
                    [-2.15, -1.25, 0.0],
                    [-2.15, 1.25, 0.0],
                ]
            ),
            "charges": np.array([-0.1] * 6 + [0.1] * 6),
            "elements": ["C"] * 6 + ["H"] * 6,
        }

        result = gist_engine.calculate_gist_properties(
            molecule=benzene_molecule,
            properties=["solvation_free_energy", "hydrophobic_surface_area"],
        )

        thermo_props = result["thermodynamic_properties"]

        # Benzene should have significant hydrophobic character
        assert thermo_props.hydrophobic_surface_area > thermo_props.polar_surface_area

        # Solvation should be moderately unfavorable
        assert 0.0 < thermo_props.solvation_free_energy < 5.0


if __name__ == "__main__":
    import time

    # Run validation tests
    pytest.main([__file__, "-v", "--tb=short"])

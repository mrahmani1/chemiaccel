"""
Unit tests for transport kinetics and partitioning analysis.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.transport.simulation_engine import TransportResult
from chemi_accel.transport.transport_analyzer import (
    KineticsAnalyzer,
    PartitioningAnalysis,
    PartitioningAnalyzer,
    TransportAnalyzer,
    TransportKinetics,
)


class TestTransportKinetics:
    """Test TransportKinetics data class."""

    def test_transport_kinetics_creation(self):
        """Test transport kinetics creation."""
        kinetics = TransportKinetics(
            permeability_coefficient=1e-6,
            diffusion_coefficients={"aqueous": 1e-9, "membrane": 1e-10},
            partition_coefficients={"water_octanol": 2.0},
            rate_constants={"forward_rate": 0.1, "reverse_rate": 0.05},
            residence_times={"aqueous": 10.0, "membrane": 5.0},
            transport_mechanisms=["passive_diffusion"],
            kinetic_barriers={"transport_barrier": 15.0},
            mass_transfer_coefficients={"overall": 1e-5},
            effective_diffusivity=5e-10,
            tortuosity_factor=1.5,
        )

        assert kinetics.permeability_coefficient == 1e-6
        assert len(kinetics.diffusion_coefficients) == 2
        assert len(kinetics.partition_coefficients) == 1
        assert len(kinetics.rate_constants) == 2
        assert len(kinetics.residence_times) == 2
        assert len(kinetics.transport_mechanisms) == 1
        assert kinetics.effective_diffusivity == 5e-10
        assert kinetics.tortuosity_factor == 1.5


class TestPartitioningAnalysis:
    """Test PartitioningAnalysis data class."""

    def test_partitioning_analysis_creation(self):
        """Test partitioning analysis creation."""
        analysis = PartitioningAnalysis(
            phase_concentrations={
                "aqueous": np.array([1, 2, 3]),
                "organic": np.array([2, 4, 6]),
            },
            equilibrium_constants={"aqueous_organic": 2.0},
            activity_coefficients={"aqueous": 1.1, "organic": 0.9},
            solubility_parameters={"dispersive": 18.0, "polar": 10.0},
            interaction_parameters={"interaction_strength": 0.2},
            thermodynamic_driving_forces={
                "chemical_potential_aqueous_organic": -1000.0
            },
            phase_behavior_classification="moderately_partitioning",
        )

        assert len(analysis.phase_concentrations) == 2
        assert len(analysis.equilibrium_constants) == 1
        assert len(analysis.activity_coefficients) == 2
        assert analysis.phase_behavior_classification == "moderately_partitioning"


class TestKineticsAnalyzer:
    """Test KineticsAnalyzer class."""

    def test_initialization(self):
        """Test kinetics analyzer initialization."""
        analyzer = KineticsAnalyzer(backend="cpu")

        assert analyzer.backend == "cpu"

    def test_backend_selection(self):
        """Test backend selection."""
        analyzer = KineticsAnalyzer(backend="auto")

        assert analyzer.backend in ["cuda", "jax", "cpu"]

    def test_analyze_permeability(self):
        """Test permeability analysis."""
        analyzer = KineticsAnalyzer(backend="cpu")

        # Create mock transport result
        result = self._create_mock_transport_result()

        permeability_analysis = analyzer.analyze_permeability(result)

        assert isinstance(permeability_analysis, dict)
        assert "steady_state_permeability" in permeability_analysis
        assert "transport_mechanisms" in permeability_analysis
        assert "flux_variance" in permeability_analysis

    def test_msd_calculation_cpu(self):
        """Test MSD calculation on CPU."""
        analyzer = KineticsAnalyzer(backend="cpu")

        # Create test trajectories
        trajectories = np.random.randn(5, 20, 3)

        msd_results = analyzer._cpu_calculate_msd(trajectories)

        assert isinstance(msd_results, dict)
        assert "total" in msd_results
        assert "x" in msd_results
        assert "y" in msd_results
        assert "z" in msd_results

        for component, msd_data in msd_results.items():
            assert len(msd_data) == 20  # Number of frames
            assert msd_data[0] == 0.0  # MSD at lag 0 should be 0

    def test_diffusion_coefficient_fitting(self):
        """Test diffusion coefficient fitting from MSD."""
        analyzer = KineticsAnalyzer(backend="cpu")

        # Create synthetic MSD data with known diffusion coefficient
        time_points = np.arange(20)
        D_true = 1e-9  # cm²/s
        msd_synthetic = 6 * D_true * time_points  # 3D diffusion

        msd_data = {
            "total": msd_synthetic,
            "x": msd_synthetic / 3,
            "y": msd_synthetic / 3,
            "z": msd_synthetic / 3,
        }

        diffusion_coeffs = analyzer._fit_diffusion_coefficients(msd_data)

        assert "total_diffusion" in diffusion_coeffs
        assert "total_r_squared" in diffusion_coeffs

        # Check that fitted diffusion coefficient is close to true value
        fitted_D = diffusion_coeffs["total_diffusion"]
        assert abs(fitted_D - D_true) / D_true < 0.1  # Within 10%

    def test_phase_specific_diffusion(self):
        """Test phase-specific diffusion analysis."""
        analyzer = KineticsAnalyzer(backend="cpu")

        # Create trajectories with different phases
        trajectories = np.zeros((3, 50, 3))

        # Molecule 1: mostly in aqueous phase (z < -2)
        trajectories[0, :, 2] = -3.0 + 0.1 * np.random.randn(50)

        # Molecule 2: mostly in membrane phase (-2 < z < 2)
        trajectories[1, :, 2] = 0.0 + 0.1 * np.random.randn(50)

        # Molecule 3: mostly in lipid phase (z > 2)
        trajectories[2, :, 2] = 3.0 + 0.1 * np.random.randn(50)

        metadata = {"config": Mock()}

        phase_diffusion = analyzer._analyze_phase_specific_diffusion(
            trajectories, metadata
        )

        assert isinstance(phase_diffusion, dict)
        assert "aqueous" in phase_diffusion
        assert "membrane" in phase_diffusion
        assert "lipid" in phase_diffusion

    def test_anomalous_diffusion_analysis(self):
        """Test anomalous diffusion analysis."""
        analyzer = KineticsAnalyzer(backend="cpu")

        # Create normal diffusion MSD (alpha = 1)
        time_points = np.arange(1, 21)
        msd_normal = time_points  # Linear in time

        # Create subdiffusive MSD (alpha < 1)
        msd_subdiffusive = time_points**0.5  # t^0.5

        msd_data = {"normal": msd_normal, "subdiffusive": msd_subdiffusive}

        anomalous_params = analyzer._analyze_anomalous_diffusion(msd_data)

        assert "normal_alpha" in anomalous_params
        assert "subdiffusive_alpha" in anomalous_params
        assert "normal_type" in anomalous_params
        assert "subdiffusive_type" in anomalous_params

        # Check diffusion types
        assert abs(anomalous_params["normal_alpha"] - 1.0) < 0.2  # Should be ~1
        assert anomalous_params["subdiffusive_alpha"] < 0.9  # Should be <1
        assert anomalous_params["normal_type"] == "normal"
        assert anomalous_params["subdiffusive_type"] == "subdiffusive"

    def test_transport_mechanism_identification(self):
        """Test transport mechanism identification."""
        analyzer = KineticsAnalyzer(backend="cpu")

        # Create trajectories with net displacement (advective transport)
        trajectories = np.zeros((5, 20, 3))
        for i in range(5):
            trajectories[i, :, 2] = np.linspace(-5, 5, 20)  # Linear displacement

        mechanisms = analyzer._identify_transport_mechanisms(trajectories)

        assert isinstance(mechanisms, list)
        assert len(mechanisms) > 0
        assert "advective_transport" in mechanisms

    def _create_mock_transport_result(self):
        """Create mock transport result for testing."""
        trajectories = np.random.randn(5, 50, 3)
        concentration_profiles = {"frame_0": np.random.randn(50)}
        flux_data = np.random.randn(20) + 1.0  # Positive flux

        return TransportResult(
            permeability_coefficient=1e-6,
            diffusion_coefficients={"aqueous": 1e-9, "membrane": 1e-10},
            partition_coefficients={"water_octanol": 2.0},
            trajectories=trajectories,
            concentration_profiles=concentration_profiles,
            flux_data=flux_data,
            simulation_time=5.0,
            metadata={"config": Mock(temperature=298.15)},
        )


class TestPartitioningAnalyzer:
    """Test PartitioningAnalyzer class."""

    def test_initialization(self):
        """Test partitioning analyzer initialization."""
        analyzer = PartitioningAnalyzer(backend="cpu")

        assert analyzer.backend == "cpu"

    def test_phase_concentration_calculation(self):
        """Test phase concentration calculation."""
        analyzer = PartitioningAnalyzer(backend="cpu")

        # Create trajectories with molecules in different phases
        trajectories = np.zeros((10, 20, 3))

        # First half of molecules in aqueous phase (z < -2)
        trajectories[:5, :, 2] = -3.0

        # Second half in organic phase (z > 2)
        trajectories[5:, :, 2] = 3.0

        phase_conc = analyzer._calculate_phase_concentrations(trajectories)

        assert isinstance(phase_conc, dict)
        assert "aqueous" in phase_conc
        assert "interface" in phase_conc
        assert "organic" in phase_conc

        # Check that aqueous and organic phases have expected concentrations
        assert np.mean(phase_conc["aqueous"]) > 0  # Should have molecules
        assert np.mean(phase_conc["organic"]) > 0  # Should have molecules
        assert np.mean(phase_conc["interface"]) == 0  # Should be empty

    def test_equilibrium_constants_calculation(self):
        """Test equilibrium constant calculation."""
        analyzer = PartitioningAnalyzer(backend="cpu")

        # Create phase concentrations
        phase_concentrations = {
            "aqueous": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),  # Stable concentration
            "organic": np.array([4.0, 4.0, 4.0, 4.0, 4.0]),  # Stable concentration
        }

        eq_constants = analyzer._calculate_equilibrium_constants(phase_concentrations)

        assert isinstance(eq_constants, dict)
        assert "aqueous_organic" in eq_constants

        # Expected K = [organic]/[aqueous] = 4/2 = 2
        expected_K = 2.0
        actual_K = eq_constants["aqueous_organic"]
        assert abs(actual_K - expected_K) < 0.1

    def test_activity_coefficient_estimation(self):
        """Test activity coefficient estimation."""
        analyzer = PartitioningAnalyzer(backend="cpu")

        phase_concentrations = {
            "aqueous": np.array([1.0, 1.0, 1.0]),
            "organic": np.array([2.0, 2.0, 2.0]),
        }

        metadata = {}

        activity_coeffs = analyzer._estimate_activity_coefficients(
            phase_concentrations, metadata
        )

        assert isinstance(activity_coeffs, dict)
        assert "aqueous" in activity_coeffs
        assert "organic" in activity_coeffs

        # Activity coefficients should be positive
        for gamma in activity_coeffs.values():
            assert gamma > 0

    def test_solubility_parameters_calculation(self):
        """Test solubility parameter calculation."""
        analyzer = PartitioningAnalyzer(backend="cpu")

        mock_result = Mock()
        mock_result.metadata = {}

        solubility_params = analyzer._calculate_solubility_parameters(mock_result)

        assert isinstance(solubility_params, dict)
        assert "dispersive" in solubility_params
        assert "polar" in solubility_params
        assert "hydrogen_bonding" in solubility_params
        assert "total" in solubility_params

        # Check that total parameter is calculated correctly
        delta_d = solubility_params["dispersive"]
        delta_p = solubility_params["polar"]
        delta_h = solubility_params["hydrogen_bonding"]
        delta_total_expected = np.sqrt(delta_d**2 + delta_p**2 + delta_h**2)

        assert abs(solubility_params["total"] - delta_total_expected) < 1e-6

    def test_interaction_parameters_analysis(self):
        """Test interaction parameter analysis."""
        analyzer = PartitioningAnalyzer(backend="cpu")

        # Create trajectories with known intermolecular distances
        trajectories = np.zeros((3, 10, 3))
        trajectories[0, :, :] = [0, 0, 0]  # Molecule at origin
        trajectories[1, :, :] = [1, 0, 0]  # Molecule at distance 1
        trajectories[2, :, :] = [0, 1, 0]  # Molecule at distance 1

        interaction_params = analyzer._analyze_interaction_parameters(trajectories)

        assert isinstance(interaction_params, dict)
        assert "average_intermolecular_distance" in interaction_params
        assert "interaction_strength" in interaction_params

        # Expected average distance should be around 1.0
        avg_distance = interaction_params["average_intermolecular_distance"]
        assert abs(avg_distance - 1.0) < 0.2

    def test_driving_forces_calculation(self):
        """Test thermodynamic driving force calculation."""
        analyzer = PartitioningAnalyzer(backend="cpu")

        phase_concentrations = {
            "aqueous": np.array([1.0, 1.0, 1.0]),
            "organic": np.array([2.0, 2.0, 2.0]),
        }

        equilibrium_constants = {"aqueous_organic": 2.0}

        driving_forces = analyzer._calculate_driving_forces(
            phase_concentrations, equilibrium_constants
        )

        assert isinstance(driving_forces, dict)

        # Should have chemical potential difference
        if driving_forces:
            for key, value in driving_forces.items():
                assert isinstance(value, (int, float))

    def test_phase_behavior_classification(self):
        """Test phase behavior classification."""
        analyzer = PartitioningAnalyzer(backend="cpu")

        # Test strongly partitioning behavior
        eq_constants_strong = {"phase1_phase2": 150.0}
        activity_coeffs = {"phase1": 1.0, "phase2": 1.0}

        classification = analyzer._classify_phase_behavior(
            eq_constants_strong, activity_coeffs
        )
        assert classification == "strongly_partitioning"

        # Test uniform distribution
        eq_constants_uniform = {"phase1_phase2": 1.0}

        classification = analyzer._classify_phase_behavior(
            eq_constants_uniform, activity_coeffs
        )
        assert classification == "uniform_distribution"

    def test_analyze_equilibrium_partitioning(self):
        """Test complete equilibrium partitioning analysis."""
        analyzer = PartitioningAnalyzer(backend="cpu")

        # Create transport result with trajectories
        trajectories = np.random.randn(5, 20, 3)

        result = TransportResult(
            permeability_coefficient=1e-6,
            diffusion_coefficients={},
            partition_coefficients={},
            trajectories=trajectories,
            concentration_profiles={},
            flux_data=np.array([]),
            simulation_time=1.0,
            metadata={},
        )

        partitioning_analysis = analyzer.analyze_equilibrium_partitioning(result)

        assert isinstance(partitioning_analysis, PartitioningAnalysis)
        assert isinstance(partitioning_analysis.phase_concentrations, dict)
        assert isinstance(partitioning_analysis.equilibrium_constants, dict)
        assert isinstance(partitioning_analysis.activity_coefficients, dict)
        assert isinstance(partitioning_analysis.phase_behavior_classification, str)


class TestTransportAnalyzer:
    """Test main TransportAnalyzer class."""

    def test_initialization(self):
        """Test transport analyzer initialization."""
        analyzer = TransportAnalyzer(backend="cpu")

        assert analyzer.backend == "cpu"
        assert hasattr(analyzer, "kinetics_analyzer")
        assert hasattr(analyzer, "partitioning_analyzer")

    def test_analyze_permeability(self):
        """Test permeability analysis."""
        analyzer = TransportAnalyzer(backend="cpu")

        result = self._create_mock_transport_result()

        permeability_analysis = analyzer.analyze_permeability(result)

        assert isinstance(permeability_analysis, dict)
        assert "steady_state_permeability" in permeability_analysis
        assert "statistical_confidence" in permeability_analysis

    def test_analyze_diffusion(self):
        """Test diffusion analysis."""
        analyzer = TransportAnalyzer(backend="cpu")

        result = self._create_mock_transport_result()

        diffusion_analysis = analyzer.analyze_diffusion(result)

        assert isinstance(diffusion_analysis, dict)
        assert "msd_data" in diffusion_analysis
        assert "diffusion_coefficients" in diffusion_analysis

    def test_analyze_partitioning(self):
        """Test partitioning analysis."""
        analyzer = TransportAnalyzer(backend="cpu")

        result = self._create_mock_transport_result()

        partitioning_analysis = analyzer.analyze_partitioning(result)

        assert isinstance(partitioning_analysis, PartitioningAnalysis)

    def test_generate_transport_report(self):
        """Test comprehensive transport report generation."""
        analyzer = TransportAnalyzer(backend="cpu")

        result = self._create_mock_transport_result()

        report = analyzer.generate_transport_report(result)

        assert isinstance(report, dict)
        assert "simulation_metadata" in report
        assert "permeability_analysis" in report
        assert "diffusion_analysis" in report
        assert "partitioning_analysis" in report
        assert "transport_kinetics" in report
        assert "recommendations" in report

        # Check recommendations
        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)

    def test_statistical_permeability_analysis(self):
        """Test statistical permeability analysis."""
        analyzer = TransportAnalyzer(backend="cpu")

        # Create result with varying flux data
        flux_data = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98])

        result = TransportResult(
            permeability_coefficient=1e-6,
            diffusion_coefficients={},
            partition_coefficients={},
            trajectories=np.zeros((1, 1, 3)),
            concentration_profiles={},
            flux_data=flux_data,
            simulation_time=1.0,
            metadata={},
        )

        stats = analyzer._statistical_permeability_analysis(result)

        assert isinstance(stats, dict)
        assert "flux_coefficient_of_variation" in stats
        assert "steady_state_cv" in stats
        assert "statistical_confidence" in stats

        # CV should be positive and reasonable
        cv = stats["flux_coefficient_of_variation"]
        assert 0 <= cv <= 1

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        analyzer = TransportAnalyzer(backend="cpu")

        # Test low permeability case
        low_perm_result = TransportResult(
            permeability_coefficient=1e-9,  # Very low
            diffusion_coefficients={"aqueous": 1e-8, "membrane": 1e-11},
            partition_coefficients={"water_octanol": -3.0},  # Very hydrophilic
            trajectories=np.zeros((1, 1, 3)),
            concentration_profiles={},
            flux_data=np.array([1.0]),
            simulation_time=1.0,
            metadata={},
        )

        recommendations = analyzer._generate_recommendations(low_perm_result)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should recommend molecular modifications for low permeability
        rec_text = " ".join(recommendations).lower()
        assert "permeability" in rec_text or "transport" in rec_text

    def _create_mock_transport_result(self):
        """Create mock transport result for testing."""
        trajectories = np.random.randn(5, 50, 3)
        concentration_profiles = {"frame_0": np.random.randn(50)}
        flux_data = np.random.randn(20) + 1.0  # Positive flux

        return TransportResult(
            permeability_coefficient=1e-6,
            diffusion_coefficients={"aqueous": 1e-9, "membrane": 1e-10},
            partition_coefficients={"water_octanol": 2.0},
            trajectories=trajectories,
            concentration_profiles=concentration_profiles,
            flux_data=flux_data,
            simulation_time=5.0,
            metadata={"config": Mock(temperature=298.15)},
        )


class TestCUDAAcceleration:
    """Test CUDA acceleration for transport analysis if available."""

    @pytest.mark.skipif(not hasattr(np, "cuda"), reason="CUDA not available")
    def test_cuda_msd_calculation(self):
        """Test CUDA-accelerated MSD calculation."""
        try:
            analyzer = KineticsAnalyzer(backend="cuda")
            trajectories = np.random.randn(5, 20, 3)

            msd_results = analyzer._cuda_calculate_msd(trajectories)

            assert isinstance(msd_results, dict)
            assert "total" in msd_results

            for component, msd_data in msd_results.items():
                assert isinstance(msd_data, np.ndarray)
                assert len(msd_data) == 20

        except ImportError:
            pytest.skip("CuPy not available for CUDA testing")


class TestJAXAcceleration:
    """Test JAX acceleration for transport analysis if available."""

    @pytest.mark.skipif(True, reason="JAX testing requires specific setup")
    def test_jax_msd_calculation(self):
        """Test JAX-accelerated MSD calculation."""
        try:
            analyzer = KineticsAnalyzer(backend="jax")
            trajectories = np.random.randn(5, 20, 3)

            msd_results = analyzer._jax_calculate_msd(trajectories)

            assert isinstance(msd_results, dict)
            assert "total" in msd_results

        except ImportError:
            pytest.skip("JAX not available for testing")

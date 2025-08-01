"""
Unit tests for multi-phase transport simulation engine.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.core.exceptions import ChemiAccelError
from chemi_accel.transport.simulation_engine import (
    AISubrogateSimulation,
    AtomisticMDSimulation,
    CoarseGrainedDPDSimulation,
    MultiPhaseSimulator,
    SimulationConfig,
    TransportResult,
)


class TestSimulationConfig:
    """Test SimulationConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = SimulationConfig()

        assert config.method == "ai_surrogate"
        assert config.temperature == 298.15
        assert config.pressure == 1.0
        assert config.timestep == 1e-15
        assert config.n_steps == 100000
        assert config.output_frequency == 1000
        assert config.interface_type == "lipid_bilayer"
        assert config.boundary_conditions == "periodic"
        assert config.phase_definitions == {}

    def test_custom_config(self):
        """Test custom configuration."""
        custom_phases = {"aqueous": "water", "organic": "octanol"}

        config = SimulationConfig(
            method="md",
            temperature=310.0,
            n_steps=50000,
            phase_definitions=custom_phases,
        )

        assert config.method == "md"
        assert config.temperature == 310.0
        assert config.n_steps == 50000
        assert config.phase_definitions == custom_phases


class TestTransportResult:
    """Test TransportResult class."""

    def test_transport_result_creation(self):
        """Test transport result creation."""
        trajectories = np.random.randn(10, 100, 3)
        concentration_profiles = {"frame_0": np.random.randn(50)}
        flux_data = np.random.randn(20)

        result = TransportResult(
            permeability_coefficient=1e-6,
            diffusion_coefficients={"aqueous": 1e-9, "membrane": 1e-10},
            partition_coefficients={"water_octanol": 2.0},
            trajectories=trajectories,
            concentration_profiles=concentration_profiles,
            flux_data=flux_data,
            simulation_time=5.0,
            metadata={"method": "ai_surrogate"},
        )

        assert result.permeability_coefficient == 1e-6
        assert len(result.diffusion_coefficients) == 2
        assert len(result.partition_coefficients) == 1
        assert result.trajectories.shape == (10, 100, 3)
        assert result.simulation_time == 5.0


class TestAISubrogateSimulation:
    """Test AI surrogate simulation method."""

    def test_initialization(self):
        """Test AI surrogate initialization."""
        sim = AISubrogateSimulation(backend="cpu")

        assert sim.backend == "cpu"
        assert hasattr(sim, "permeability_model")
        assert hasattr(sim, "diffusion_model")
        assert hasattr(sim, "partition_model")

    def test_backend_selection(self):
        """Test backend selection."""
        sim = AISubrogateSimulation(backend="auto")

        assert sim.backend in ["cuda", "jax", "cpu"]

    def test_feature_extraction(self):
        """Test molecular feature extraction."""
        sim = AISubrogateSimulation(backend="cpu")

        molecules = ["CCO", "CC(C)C", "c1ccccc1"]
        features = sim._extract_molecular_features(molecules)

        assert features.shape == (3, 5)  # 3 molecules, 5 features
        assert features.dtype == np.float32

    def test_cpu_models(self):
        """Test CPU-based surrogate models."""
        sim = AISubrogateSimulation(backend="cpu")

        features = np.random.randn(2, 5).astype(np.float32)

        # Test permeability model
        permeabilities = sim._cpu_permeability_model(features)
        assert len(permeabilities) == 2
        assert np.all(permeabilities > 0)

        # Test diffusion model
        diffusion_coeffs = sim._cpu_diffusion_model(features)
        assert diffusion_coeffs.shape == (2, 3)
        assert np.all(diffusion_coeffs > 0)

        # Test partition model
        partition_coeffs = sim._cpu_partition_model(features)
        assert partition_coeffs.shape == (2, 2)

    def test_trajectory_generation(self):
        """Test trajectory generation."""
        sim = AISubrogateSimulation(backend="cpu")
        config = SimulationConfig(n_steps=10000, output_frequency=1000)

        molecules = ["CCO", "CC(C)C"]
        trajectories = sim._generate_trajectories(molecules, config)

        expected_frames = config.n_steps // config.output_frequency
        assert trajectories.shape == (2, expected_frames, 3)

    def test_concentration_profiles(self):
        """Test concentration profile calculation."""
        sim = AISubrogateSimulation(backend="cpu")
        config = SimulationConfig()

        trajectories = np.random.randn(5, 20, 3)
        profiles = sim._calculate_concentration_profiles(trajectories, config)

        assert isinstance(profiles, dict)
        assert len(profiles) > 0

        for profile in profiles.values():
            assert isinstance(profile, np.ndarray)
            assert len(profile) == 50  # Number of bins

    def test_flux_calculation(self):
        """Test flux calculation."""
        sim = AISubrogateSimulation(backend="cpu")
        config = SimulationConfig()

        # Create trajectories with membrane crossing
        trajectories = np.zeros((3, 20, 3))
        trajectories[:, :10, 2] = -2.0  # Start below membrane
        trajectories[:, 10:, 2] = 2.0  # End above membrane

        flux = sim._calculate_flux(trajectories, config)

        assert isinstance(flux, np.ndarray)
        assert len(flux) == 19  # Number of time bins

    def test_run_simulation(self):
        """Test complete simulation run."""
        sim = AISubrogateSimulation(backend="cpu")
        config = SimulationConfig(n_steps=5000, output_frequency=500)

        molecules = ["CCO", "c1ccccc1"]
        result = sim.run_simulation(molecules, config)

        assert isinstance(result, TransportResult)
        assert result.permeability_coefficient > 0
        assert len(result.diffusion_coefficients) == 3
        assert len(result.partition_coefficients) == 2
        assert result.trajectories.shape == (2, 10, 3)
        assert result.simulation_time >= 0


class TestAtomisticMDSimulation:
    """Test atomistic MD simulation method."""

    def test_initialization(self):
        """Test MD simulation initialization."""
        sim = AtomisticMDSimulation(backend="cpu")

        assert sim.backend == "cpu"

    def test_run_simulation(self):
        """Test MD simulation run (placeholder)."""
        sim = AtomisticMDSimulation(backend="cpu")
        config = SimulationConfig()

        molecules = ["CCO"]
        result = sim.run_simulation(molecules, config)

        assert isinstance(result, TransportResult)
        assert result.permeability_coefficient > 0
        assert "method" in result.metadata
        assert result.metadata["method"] == "atomistic_md"


class TestCoarseGrainedDPDSimulation:
    """Test coarse-grained DPD simulation method."""

    def test_initialization(self):
        """Test DPD simulation initialization."""
        sim = CoarseGrainedDPDSimulation(backend="cpu")

        assert sim.backend == "cpu"

    def test_run_simulation(self):
        """Test DPD simulation run (placeholder)."""
        sim = CoarseGrainedDPDSimulation(backend="cpu")
        config = SimulationConfig()

        molecules = ["CC(C)C"]
        result = sim.run_simulation(molecules, config)

        assert isinstance(result, TransportResult)
        assert result.permeability_coefficient > 0
        assert "method" in result.metadata
        assert result.metadata["method"] == "coarse_grained_dpd"


class TestMultiPhaseSimulator:
    """Test MultiPhaseSimulator class."""

    def test_initialization(self):
        """Test simulator initialization."""
        simulator = MultiPhaseSimulator(backend="cpu", method="ai_surrogate")

        assert simulator.backend == "cpu"
        assert simulator.method == "ai_surrogate"
        assert hasattr(simulator, "simulation_method")

    def test_invalid_method(self):
        """Test invalid simulation method."""
        with pytest.raises(ChemiAccelError):
            MultiPhaseSimulator(method="invalid_method")

    def test_membrane_transport_simulation(self):
        """Test membrane transport simulation."""
        simulator = MultiPhaseSimulator(backend="cpu", method="ai_surrogate")

        molecules = ["CCO", "c1ccccc1"]
        result = simulator.simulate_membrane_transport(
            molecules=molecules,
            membrane_type="dppc_bilayer",
            conditions={"temperature": 310.0, "pH": 7.4},
            n_steps=1000,
            output_frequency=100,
        )

        assert isinstance(result, TransportResult)
        assert result.permeability_coefficient > 0
        assert len(result.diffusion_coefficients) > 0
        assert len(result.partition_coefficients) > 0

    def test_liquid_liquid_extraction(self):
        """Test liquid-liquid extraction simulation."""
        simulator = MultiPhaseSimulator(backend="cpu", method="ai_surrogate")

        molecules = ["CCO"]
        result = simulator.simulate_liquid_liquid_extraction(
            molecules=molecules,
            solvent_system=("water", "octanol"),
            conditions={"temperature": 298.15},
            n_steps=1000,
        )

        assert isinstance(result, TransportResult)
        assert result.metadata["config"].interface_type == "liquid_liquid"

    def test_polymer_membrane_transport(self):
        """Test polymer membrane transport simulation."""
        simulator = MultiPhaseSimulator(backend="cpu", method="ai_surrogate")

        molecules = ["CC(C)C"]
        result = simulator.simulate_polymer_membrane_transport(
            molecules=molecules,
            polymer_membrane="pvdf",
            conditions={"temperature": 298.15},
            n_steps=1000,
        )

        assert isinstance(result, TransportResult)
        assert result.metadata["config"].interface_type == "polymer_membrane"

    def test_membrane_definitions(self):
        """Test membrane definition retrieval."""
        simulator = MultiPhaseSimulator(backend="cpu")

        dppc_def = simulator._get_membrane_definition("dppc_bilayer")

        assert "lipid_type" in dppc_def
        assert dppc_def["lipid_type"] == "DPPC"
        assert "thickness" in dppc_def
        assert "area_per_lipid" in dppc_def

    def test_polymer_definitions(self):
        """Test polymer membrane definition retrieval."""
        simulator = MultiPhaseSimulator(backend="cpu")

        pvdf_def = simulator._get_polymer_membrane_definition("pvdf")

        assert "polymer_type" in pvdf_def
        assert pvdf_def["polymer_type"] == "PVDF"
        assert "porosity" in pvdf_def
        assert "thickness" in pvdf_def

    def test_benchmark_against_experimental(self):
        """Test benchmarking against experimental data."""
        simulator = MultiPhaseSimulator(backend="cpu")

        # Mock simulation result
        result = TransportResult(
            permeability_coefficient=1e-6,
            diffusion_coefficients={},
            partition_coefficients={"water_octanol": 2.0},
            trajectories=np.zeros((1, 1, 3)),
            concentration_profiles={},
            flux_data=np.array([1.0]),
            simulation_time=1.0,
            metadata={},
        )

        experimental_data = {"permeability": 1.2e-6, "partition_coefficient": 1.8}

        metrics = simulator.benchmark_against_experimental(experimental_data, result)

        assert "permeability_error" in metrics
        assert "partition_error" in metrics
        assert metrics["permeability_error"] >= 0
        assert metrics["partition_error"] >= 0


class TestCUDAAcceleration:
    """Test CUDA acceleration if available."""

    @pytest.mark.skipif(not hasattr(np, "cuda"), reason="CUDA not available")
    def test_cuda_simulation(self):
        """Test CUDA-accelerated simulation."""
        try:
            sim = AISubrogateSimulation(backend="cuda")
            config = SimulationConfig(n_steps=1000, output_frequency=100)

            molecules = ["CCO"]
            result = sim.run_simulation(molecules, config)

            assert isinstance(result, TransportResult)
            assert result.metadata["backend"] == "cuda"

        except ImportError:
            pytest.skip("CuPy not available for CUDA testing")


class TestJAXAcceleration:
    """Test JAX acceleration if available."""

    @pytest.mark.skipif(True, reason="JAX testing requires specific setup")
    def test_jax_simulation(self):
        """Test JAX-accelerated simulation."""
        try:
            sim = AISubrogateSimulation(backend="jax")
            config = SimulationConfig(n_steps=1000, output_frequency=100)

            molecules = ["CCO"]
            result = sim.run_simulation(molecules, config)

            assert isinstance(result, TransportResult)
            assert result.metadata["backend"] == "jax"

        except ImportError:
            pytest.skip("JAX not available for testing")

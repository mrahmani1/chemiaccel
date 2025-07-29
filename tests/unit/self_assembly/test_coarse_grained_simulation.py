"""
Comprehensive tests for the coarse-grained simulation engine.

Tests cover GPU-accelerated particle interactions, DPD simulations,
and various simulation components with different hardware configurations.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.self_assembly.coarse_grained_simulation import (  # Main classes; Data structures; Enums; Utility functions; Exceptions
    Bond,
    BoundaryCondition,
    CoarseGrainedSimulationEngine,
    CoarseGrainedSimulationError,
    GPUMemoryError,
    IntegrationError,
    IntegratorType,
    InteractionParameters,
    InteractionType,
    MDIntegrator,
    Particle,
    ParticleInteractionCalculator,
    ParticleType,
    SimulationBox,
    SimulationMethod,
    SimulationParameters,
    SimulationState,
    TrajectoryFrame,
    create_amphiphilic_system,
    run_dpd_simulation,
)


class TestParticleType:
    """Test ParticleType dataclass."""

    def test_particle_type_creation(self):
        """Test basic particle type creation."""
        ptype = ParticleType(
            name="TEST",
            mass=1.5,
            radius=0.8,
            charge=-0.5,
            epsilon=2.0,
            sigma=1.2,
            hydrophobic=True,
        )

        assert ptype.name == "TEST"
        assert ptype.mass == 1.5
        assert ptype.radius == 0.8
        assert ptype.charge == -0.5
        assert ptype.epsilon == 2.0
        assert ptype.sigma == 1.2
        assert ptype.hydrophobic is True

    def test_particle_type_defaults(self):
        """Test particle type with default values."""
        ptype = ParticleType(name="DEFAULT", mass=1.0, radius=0.5)

        assert ptype.charge == 0.0
        assert ptype.epsilon == 1.0
        assert ptype.sigma == 1.0
        assert ptype.color == "blue"
        assert ptype.hydrophobic is False

    def test_particle_type_validation(self):
        """Test particle type validation."""
        with pytest.raises(ValueError, match="Particle mass must be positive"):
            ParticleType(name="INVALID", mass=0.0, radius=0.5)

        with pytest.raises(ValueError, match="Particle radius must be positive"):
            ParticleType(name="INVALID", mass=1.0, radius=-0.5)


class TestParticle:
    """Test Particle dataclass."""

    def test_particle_creation(self):
        """Test basic particle creation."""
        ptype = ParticleType(name="TEST", mass=1.0, radius=0.5)
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.1, 0.2, 0.3])

        particle = Particle(
            particle_id=42, particle_type=ptype, position=position, velocity=velocity
        )

        assert particle.particle_id == 42
        assert particle.particle_type == ptype
        np.testing.assert_array_equal(particle.position, position)
        np.testing.assert_array_equal(particle.velocity, velocity)
        np.testing.assert_array_equal(particle.force, np.zeros(3))
        assert particle.bonds == []

    def test_particle_validation(self):
        """Test particle validation."""
        ptype = ParticleType(name="TEST", mass=1.0, radius=0.5)

        with pytest.raises(ValueError, match="Position must be 3D vector"):
            Particle(
                particle_id=1,
                particle_type=ptype,
                position=np.array([1.0, 2.0]),  # 2D instead of 3D
                velocity=np.array([0.1, 0.2, 0.3]),
            )

        with pytest.raises(ValueError, match="Velocity must be 3D vector"):
            Particle(
                particle_id=1,
                particle_type=ptype,
                position=np.array([1.0, 2.0, 3.0]),
                velocity=np.array([0.1, 0.2]),  # 2D instead of 3D
            )


class TestSimulationBox:
    """Test SimulationBox dataclass."""

    def test_simulation_box_creation(self):
        """Test simulation box creation."""
        dimensions = np.array([10.0, 10.0, 10.0])
        boundary_conditions = [BoundaryCondition.PERIODIC] * 3

        box = SimulationBox(
            dimensions=dimensions,
            boundary_conditions=boundary_conditions,
            periodic_box=True,
        )

        np.testing.assert_array_equal(box.dimensions, dimensions)
        assert box.boundary_conditions == boundary_conditions
        assert box.periodic_box is True

    def test_simulation_box_defaults(self):
        """Test simulation box with default boundary conditions."""
        dimensions = np.array([5.0, 5.0, 5.0])

        box = SimulationBox(dimensions=dimensions)

        expected_bc = [BoundaryCondition.PERIODIC] * 3
        assert box.boundary_conditions == expected_bc

    def test_simulation_box_validation(self):
        """Test simulation box validation."""
        with pytest.raises(ValueError, match="Box dimensions must be 3D"):
            SimulationBox(dimensions=np.array([10.0, 10.0]))  # 2D instead of 3D


class TestSimulationParameters:
    """Test SimulationParameters dataclass."""

    def test_simulation_parameters_creation(self):
        """Test simulation parameters creation."""
        params = SimulationParameters(
            method=SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS,
            integrator=IntegratorType.DPD_INTEGRATOR,
            timestep=0.005,
            total_time=50.0,
            temperature=300.0,
            use_gpu=False,
        )

        assert params.method == SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS
        assert params.integrator == IntegratorType.DPD_INTEGRATOR
        assert params.timestep == 0.005
        assert params.total_time == 50.0
        assert params.temperature == 300.0
        assert params.use_gpu is False

    def test_simulation_parameters_defaults(self):
        """Test simulation parameters with defaults."""
        params = SimulationParameters()

        assert params.method == SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS
        assert params.integrator == IntegratorType.DPD_INTEGRATOR
        assert params.timestep == 0.01
        assert params.total_time == 100.0
        assert params.temperature == 298.15
        assert params.use_gpu is True

    def test_simulation_parameters_validation(self):
        """Test simulation parameters validation."""
        with pytest.raises(ValueError, match="Timestep must be positive"):
            SimulationParameters(timestep=0.0)

        with pytest.raises(ValueError, match="Total simulation time must be positive"):
            SimulationParameters(total_time=-10.0)

        with pytest.raises(ValueError, match="Temperature must be positive"):
            SimulationParameters(temperature=0.0)


class TestInteractionParameters:
    """Test InteractionParameters dataclass."""

    def test_interaction_parameters_creation(self):
        """Test interaction parameters creation."""
        params = InteractionParameters(dpd_gamma=5.0, dpd_sigma=2.5)

        assert params.dpd_gamma == 5.0
        assert params.dpd_sigma == 2.5
        assert params.interaction_matrix == {}

    def test_add_interaction(self):
        """Test adding interaction parameters."""
        params = InteractionParameters()

        params.add_interaction(
            "A", "B", InteractionType.LENNARD_JONES, {"epsilon": 2.0, "sigma": 1.5}
        )

        interaction = params.get_interaction("A", "B", InteractionType.LENNARD_JONES)
        assert interaction == {"epsilon": 2.0, "sigma": 1.5}

        # Should work in reverse order too
        interaction = params.get_interaction("B", "A", InteractionType.LENNARD_JONES)
        assert interaction == {"epsilon": 2.0, "sigma": 1.5}

    def test_get_nonexistent_interaction(self):
        """Test getting nonexistent interaction."""
        params = InteractionParameters()

        interaction = params.get_interaction("A", "B", InteractionType.LENNARD_JONES)
        assert interaction is None


class TestParticleInteractionCalculator:
    """Test ParticleInteractionCalculator class."""

    def test_calculator_initialization_cpu(self):
        """Test calculator initialization for CPU."""
        calculator = ParticleInteractionCalculator(use_gpu=False)

        assert calculator.use_gpu is False
        assert calculator.device_id == 0

    @pytest.mark.skipif(
        not hasattr(pytest, "importorskip"), reason="CuPy not available"
    )
    def test_calculator_initialization_gpu(self):
        """Test calculator initialization for GPU (if available)."""
        try:
            import cupy as cp

            calculator = ParticleInteractionCalculator(use_gpu=True, device_id=0)
            # If we get here, GPU is available
            assert calculator.device_id == 0
        except ImportError:
            pytest.skip("CuPy not available for GPU testing")

    def test_lennard_jones_forces_cpu(self):
        """Test Lennard-Jones force calculation on CPU."""
        calculator = ParticleInteractionCalculator(use_gpu=False)

        # Create simple test system
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        particle_types = ["A", "A", "A"]

        # Setup interactions
        interaction_params = InteractionParameters()
        interaction_params.add_interaction(
            "A", "A", InteractionType.LENNARD_JONES, {"epsilon": 1.0, "sigma": 1.0}
        )

        box_dimensions = np.array([10.0, 10.0, 10.0])
        cutoff = 2.5

        forces = calculator.calculate_lennard_jones_forces(
            positions, particle_types, interaction_params, box_dimensions, cutoff
        )

        assert forces.shape == (3, 3)
        # Forces should sum to zero (conservation)
        np.testing.assert_allclose(np.sum(forces, axis=0), [0.0, 0.0, 0.0], atol=1e-10)

    def test_dpd_forces_cpu(self):
        """Test DPD force calculation on CPU."""
        calculator = ParticleInteractionCalculator(use_gpu=False)

        # Create test system
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        particle_types = ["A", "A"]

        interaction_params = InteractionParameters(dpd_gamma=4.5, dpd_sigma=3.0)
        interaction_params.add_interaction(
            "A", "A", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
        )

        box_dimensions = np.array([10.0, 10.0, 10.0])
        cutoff = 2.0
        temperature = 298.15
        dt = 0.01

        forces = calculator.calculate_dpd_forces(
            positions,
            velocities,
            particle_types,
            interaction_params,
            box_dimensions,
            cutoff,
            temperature,
            dt,
        )

        assert forces.shape == (2, 3)
        # Forces should be finite
        assert np.all(np.isfinite(forces))


class TestMDIntegrator:
    """Test MD integration schemes."""

    def test_velocity_verlet_step(self):
        """Test velocity-Verlet integration step."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        forces = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])
        dt = 0.01
        box_dimensions = np.array([10.0, 10.0, 10.0])

        new_pos, new_vel = MDIntegrator.velocity_verlet_step(
            positions, velocities, forces, masses, dt, box_dimensions
        )

        assert new_pos.shape == (2, 3)
        assert new_vel.shape == (2, 3)
        assert np.all(np.isfinite(new_pos))
        assert np.all(np.isfinite(new_vel))

    def test_dpd_integrator_step(self):
        """Test DPD-specific integration step."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        forces = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])
        dt = 0.01
        box_dimensions = np.array([10.0, 10.0, 10.0])
        temperature = 298.15

        new_pos, new_vel = MDIntegrator.dpd_integrator_step(
            positions, velocities, forces, masses, dt, box_dimensions, temperature
        )

        assert new_pos.shape == (2, 3)
        assert new_vel.shape == (2, 3)
        assert np.all(np.isfinite(new_pos))
        assert np.all(np.isfinite(new_vel))


class TestCoarseGrainedSimulationEngine:
    """Test main simulation engine."""

    def create_simple_system(self):
        """Create a simple test system."""
        # Create particle types
        ptype = ParticleType(name="TEST", mass=1.0, radius=0.5)

        # Create particles
        particles = []
        for i in range(5):
            position = np.random.uniform(0, 5, 3)
            velocity = np.random.normal(0, 1, 3)
            particle = Particle(
                particle_id=i, particle_type=ptype, position=position, velocity=velocity
            )
            particles.append(particle)

        # Create simulation box
        simulation_box = SimulationBox(
            dimensions=np.array([5.0, 5.0, 5.0]),
            boundary_conditions=[BoundaryCondition.PERIODIC] * 3,
        )

        # Create simulation parameters
        simulation_params = SimulationParameters(
            method=SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS,
            timestep=0.01,
            total_time=0.1,  # Very short for testing
            use_gpu=False,  # Use CPU for testing
            output_frequency=5,
        )

        # Create interaction parameters
        interaction_params = InteractionParameters()
        interaction_params.add_interaction(
            "TEST", "TEST", InteractionType.DPD_CONSERVATIVE, {"a": 25.0}
        )

        return particles, simulation_box, simulation_params, interaction_params

    def test_engine_initialization(self):
        """Test simulation engine initialization."""
        (
            particles,
            simulation_box,
            simulation_params,
            interaction_params,
        ) = self.create_simple_system()

        engine = CoarseGrainedSimulationEngine(
            particles, simulation_box, simulation_params, interaction_params
        )

        assert len(engine.particles) == 5
        assert engine.positions.shape == (5, 3)
        assert engine.velocities.shape == (5, 3)
        assert engine.forces.shape == (5, 3)
        assert len(engine.masses) == 5
        assert len(engine.particle_types) == 5

    def test_engine_run_simulation(self):
        """Test running a short simulation."""
        (
            particles,
            simulation_box,
            simulation_params,
            interaction_params,
        ) = self.create_simple_system()

        engine = CoarseGrainedSimulationEngine(
            particles, simulation_box, simulation_params, interaction_params
        )

        trajectory = engine.run_simulation()

        assert len(trajectory) > 0
        assert all(isinstance(frame, TrajectoryFrame) for frame in trajectory)

        # Check trajectory frame contents
        frame = trajectory[0]
        assert frame.positions.shape == (5, 3)
        assert frame.velocities.shape == (5, 3)
        assert frame.forces.shape == (5, 3)
        assert isinstance(frame.energies, dict)
        assert "kinetic" in frame.energies
        assert "potential" in frame.energies
        assert "total" in frame.energies

    def test_engine_state_tracking(self):
        """Test simulation state tracking."""
        (
            particles,
            simulation_box,
            simulation_params,
            interaction_params,
        ) = self.create_simple_system()

        engine = CoarseGrainedSimulationEngine(
            particles, simulation_box, simulation_params, interaction_params
        )

        initial_state = engine.get_current_state()
        assert initial_state.step == 0
        assert initial_state.time == 0.0

        # Run one step manually
        engine._simulation_step(1)

        updated_state = engine.get_current_state()
        assert updated_state.step == 1
        assert updated_state.time == simulation_params.timestep


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_amphiphilic_system(self):
        """Test creation of amphiphilic system."""
        particles, simulation_box, interaction_params = create_amphiphilic_system(
            n_molecules=10, head_size=1, tail_size=3, box_size=10.0
        )

        # Check system composition
        assert len(particles) > 10  # Should have molecules + water
        assert simulation_box.dimensions.shape == (3,)
        assert np.allclose(simulation_box.dimensions, [10.0, 10.0, 10.0])

        # Check particle types
        particle_types = set(p.particle_type.name for p in particles)
        assert "HEAD" in particle_types
        assert "TAIL" in particle_types
        assert "WATER" in particle_types

        # Check interaction parameters
        assert len(interaction_params.interaction_matrix) > 0

    def test_run_dpd_simulation(self):
        """Test high-level DPD simulation function."""
        particles, simulation_box, interaction_params = create_amphiphilic_system(
            n_molecules=5, head_size=1, tail_size=2, box_size=5.0
        )

        trajectory = run_dpd_simulation(
            particles=particles,
            simulation_box=simulation_box,
            interaction_params=interaction_params,
            total_time=0.05,  # Very short simulation
            timestep=0.01,
            use_gpu=False,
        )

        assert len(trajectory) > 0
        assert all(isinstance(frame, TrajectoryFrame) for frame in trajectory)


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_simulation_error_on_invalid_parameters(self):
        """Test that simulation raises error with invalid parameters."""
        (
            particles,
            simulation_box,
            _,
            interaction_params,
        ) = TestCoarseGrainedSimulationEngine().create_simple_system()

        # Invalid simulation parameters
        invalid_params = SimulationParameters(
            timestep=-0.01
        )  # Should raise during creation

        with pytest.raises(ValueError):
            invalid_params = SimulationParameters(timestep=-0.01)

    def test_gpu_memory_error_handling(self):
        """Test GPU memory error handling."""
        # This test would require mocking GPU operations
        # For now, just test that the exception class exists
        assert issubclass(GPUMemoryError, CoarseGrainedSimulationError)

    def test_integration_error_handling(self):
        """Test integration error handling."""
        assert issubclass(IntegrationError, CoarseGrainedSimulationError)


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for simulation components."""

    def test_force_calculation_performance(self, benchmark):
        """Benchmark force calculation performance."""
        calculator = ParticleInteractionCalculator(use_gpu=False)

        # Create larger test system
        n_particles = 100
        positions = np.random.uniform(0, 10, (n_particles, 3))
        particle_types = ["A"] * n_particles

        interaction_params = InteractionParameters()
        interaction_params.add_interaction(
            "A", "A", InteractionType.LENNARD_JONES, {"epsilon": 1.0, "sigma": 1.0}
        )

        box_dimensions = np.array([10.0, 10.0, 10.0])
        cutoff = 2.5

        result = benchmark(
            calculator.calculate_lennard_jones_forces,
            positions,
            particle_types,
            interaction_params,
            box_dimensions,
            cutoff,
        )

        assert result.shape == (n_particles, 3)

    def test_simulation_step_performance(self, benchmark):
        """Benchmark single simulation step performance."""
        (
            particles,
            simulation_box,
            simulation_params,
            interaction_params,
        ) = TestCoarseGrainedSimulationEngine().create_simple_system()

        engine = CoarseGrainedSimulationEngine(
            particles, simulation_box, simulation_params, interaction_params
        )

        result = benchmark(engine._simulation_step, 0)

        # Check that step completed
        assert engine.state.step == 0


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete simulation scenarios."""

    def test_micelle_formation_simulation(self):
        """Test simulation of micelle formation."""
        # Create amphiphilic system
        particles, simulation_box, interaction_params = create_amphiphilic_system(
            n_molecules=20, head_size=1, tail_size=4, box_size=15.0
        )

        # Run DPD simulation
        trajectory = run_dpd_simulation(
            particles=particles,
            simulation_box=simulation_box,
            interaction_params=interaction_params,
            total_time=1.0,
            timestep=0.01,
            temperature=298.15,
            use_gpu=False,
        )

        # Check that simulation completed
        assert len(trajectory) > 0

        # Check energy conservation (approximately)
        energies = [frame.energies["total"] for frame in trajectory]
        energy_drift = abs(energies[-1] - energies[0]) / abs(energies[0])
        assert energy_drift < 0.5  # Allow some drift due to thermostat

    def test_different_simulation_methods(self):
        """Test different simulation methods."""
        (
            particles,
            simulation_box,
            simulation_params,
            interaction_params,
        ) = TestCoarseGrainedSimulationEngine().create_simple_system()

        # Test DPD method
        simulation_params.method = SimulationMethod.DISSIPATIVE_PARTICLE_DYNAMICS
        simulation_params.total_time = 0.05

        engine_dpd = CoarseGrainedSimulationEngine(
            particles, simulation_box, simulation_params, interaction_params
        )
        trajectory_dpd = engine_dpd.run_simulation()

        # Test Coarse-Grained MD method
        simulation_params.method = SimulationMethod.COARSE_GRAINED_MD
        simulation_params.integrator = IntegratorType.VELOCITY_VERLET

        engine_cgmd = CoarseGrainedSimulationEngine(
            particles, simulation_box, simulation_params, interaction_params
        )
        trajectory_cgmd = engine_cgmd.run_simulation()

        # Both should complete successfully
        assert len(trajectory_dpd) > 0
        assert len(trajectory_cgmd) > 0

    def test_boundary_condition_effects(self):
        """Test different boundary conditions."""
        (
            particles,
            _,
            simulation_params,
            interaction_params,
        ) = TestCoarseGrainedSimulationEngine().create_simple_system()

        # Test periodic boundaries
        periodic_box = SimulationBox(
            dimensions=np.array([5.0, 5.0, 5.0]),
            boundary_conditions=[BoundaryCondition.PERIODIC] * 3,
        )

        simulation_params.total_time = 0.05

        engine = CoarseGrainedSimulationEngine(
            particles, periodic_box, simulation_params, interaction_params
        )
        trajectory = engine.run_simulation()

        # Check that particles stay within bounds (due to PBC)
        final_positions = trajectory[-1].positions
        assert np.all(final_positions >= 0)
        assert np.all(final_positions < 5.0)


if __name__ == "__main__":
    pytest.main([__file__])

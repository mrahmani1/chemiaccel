"""
Unit tests for fluid dynamics coupling module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.chemi_accel.core.exceptions import ChemiAccelError
from src.chemi_accel.digital_twin.fluid_dynamics_coupling import (
    AdaptiveMeshRefinement,
    CouplingScheme,
    FiniteDifferenceHeatSolver,
    FluidDomain,
    FluidDynamicsCouplingManager,
    FluidProperties,
    FluidState,
    FluidThermalCoupledSolver,
    MaterialPropertyCoupling,
    SimpleFDFluidSolver,
)
from src.chemi_accel.digital_twin.manufacturing_integration import ProcessConditions
from src.chemi_accel.digital_twin.multiscale_framework import MultiScaleFramework


class TestFluidProperties:
    """Test FluidProperties dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        props = FluidProperties()

        assert props.density == 1000.0
        assert props.viscosity == 0.001
        assert props.thermal_conductivity == 0.6
        assert props.specific_heat == 4180.0
        assert props.thermal_expansion == 2.1e-4
        assert props.density_func is None
        assert props.viscosity_func is None
        assert props.thermal_conductivity_func is None

    def test_full_initialization(self):
        """Test full initialization with custom values."""

        def custom_viscosity_func(T):
            return 0.002 * np.exp(-1000 / T)

        props = FluidProperties(
            density=1200.0,
            viscosity=0.002,
            thermal_conductivity=0.8,
            specific_heat=3000.0,
            thermal_expansion=3e-4,
            viscosity_func=custom_viscosity_func,
        )

        assert props.density == 1200.0
        assert props.viscosity == 0.002
        assert props.thermal_conductivity == 0.8
        assert props.specific_heat == 3000.0
        assert props.thermal_expansion == 3e-4
        assert props.viscosity_func == custom_viscosity_func

    def test_update_with_temperature_default(self):
        """Test temperature update with default functions."""
        props = FluidProperties(density=1000.0, viscosity=0.001, thermal_expansion=2e-4)

        # Test at high temperature
        updated_props = props.update_with_temperature(373.15)  # 100°C

        # Density should decrease due to thermal expansion
        assert updated_props.density < props.density

        # Viscosity should decrease due to temperature (Arrhenius)
        assert updated_props.viscosity < props.viscosity

        # Other properties should remain unchanged
        assert updated_props.thermal_conductivity == props.thermal_conductivity
        assert updated_props.specific_heat == props.specific_heat

    def test_update_with_temperature_custom_functions(self):
        """Test temperature update with custom functions."""

        def custom_density_func(T):
            return 1000.0 + (T - 293.15) * (-0.5)  # Linear decrease

        def custom_viscosity_func(T):
            return 0.001 * (293.15 / T)  # Inverse proportional

        props = FluidProperties(
            density=1000.0,
            viscosity=0.001,
            density_func=custom_density_func,
            viscosity_func=custom_viscosity_func,
        )

        updated_props = props.update_with_temperature(373.15)

        # Should use custom functions
        expected_density = custom_density_func(373.15)
        expected_viscosity = custom_viscosity_func(373.15)

        assert updated_props.density == expected_density
        assert updated_props.viscosity == expected_viscosity


class TestFluidDomain:
    """Test FluidDomain dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        domain = FluidDomain(dimensions=(1.0, 0.5, 0.2), mesh_resolution=(20, 10, 5))

        assert domain.dimensions == (1.0, 0.5, 0.2)
        assert domain.mesh_resolution == (20, 10, 5)

        # Check default boundary types
        assert domain.boundary_types["inlet"] == "velocity_inlet"
        assert domain.boundary_types["outlet"] == "pressure_outlet"
        assert domain.boundary_types["walls"] == "no_slip"
        assert domain.boundary_types["top"] == "symmetry"
        assert domain.boundary_types["bottom"] == "wall"

    def test_custom_boundary_types(self):
        """Test custom boundary types."""
        custom_boundaries = {
            "inlet": "mass_inlet",
            "outlet": "outflow",
            "walls": "slip",
        }

        domain = FluidDomain(
            dimensions=(1.0, 1.0, 1.0),
            mesh_resolution=(10, 10, 10),
            boundary_types=custom_boundaries,
        )

        assert domain.boundary_types == custom_boundaries


class TestFluidState:
    """Test FluidState dataclass."""

    @pytest.fixture
    def sample_fluid_state(self):
        """Create sample fluid state."""
        shape = (5, 5, 5)
        velocity_field = np.random.random(shape + (3,))
        pressure_field = np.random.random(shape) * 1000
        temperature_field = np.random.random(shape) * 50 + 300
        density_field = np.full(shape, 1000.0)
        viscosity_field = np.full(shape, 0.001)

        return FluidState(
            velocity_field=velocity_field,
            pressure_field=pressure_field,
            temperature_field=temperature_field,
            density_field=density_field,
            viscosity_field=viscosity_field,
            time=10.0,
        )

    def test_initialization(self, sample_fluid_state):
        """Test proper initialization."""
        state = sample_fluid_state

        assert state.velocity_field.shape == (5, 5, 5, 3)
        assert state.pressure_field.shape == (5, 5, 5)
        assert state.temperature_field.shape == (5, 5, 5)
        assert state.density_field.shape == (5, 5, 5)
        assert state.viscosity_field.shape == (5, 5, 5)
        assert state.time == 10.0
        assert state.convergence_residuals == {}

    def test_calculate_flow_metrics(self, sample_fluid_state):
        """Test flow metrics calculation."""
        state = sample_fluid_state

        metrics = state.calculate_flow_metrics()

        assert "velocity_uniformity" in metrics
        assert "temperature_uniformity" in metrics
        assert "pressure_drop" in metrics
        assert "max_velocity" in metrics
        assert "mean_temperature" in metrics

        # Check value ranges
        assert 0 <= metrics["velocity_uniformity"] <= 1
        assert 0 <= metrics["temperature_uniformity"] <= 1
        assert metrics["pressure_drop"] >= 0
        assert metrics["max_velocity"] >= 0
        assert metrics["mean_temperature"] > 0


class TestSimpleFDFluidSolver:
    """Test SimpleFDFluidSolver class."""

    @pytest.fixture
    def solver(self):
        """Create SimpleFDFluidSolver instance."""
        return SimpleFDFluidSolver()

    @pytest.fixture
    def domain(self):
        """Create FluidDomain instance."""
        return FluidDomain(dimensions=(1.0, 0.5, 0.5), mesh_resolution=(10, 10, 10))

    @pytest.fixture
    def properties(self):
        """Create FluidProperties instance."""
        return FluidProperties(
            density=1000.0, viscosity=0.001, thermal_conductivity=0.6
        )

    def test_initialization(self, solver, domain, properties):
        """Test solver initialization."""
        solver.initialize(domain, properties)

        assert solver.domain == domain
        assert solver.properties == properties
        assert solver.velocity_field.shape == (10, 10, 10, 3)
        assert solver.pressure_field.shape == (10, 10, 10)
        assert solver.density_field.shape == (10, 10, 10)
        assert solver.viscosity_field.shape == (10, 10, 10)
        assert np.all(solver.density_field == properties.density)
        assert np.all(solver.viscosity_field == properties.viscosity)

    def test_set_boundary_conditions(self, solver, domain, properties):
        """Test setting boundary conditions."""
        solver.initialize(domain, properties)

        boundary_conditions = {"inlet_velocity": 1.0, "outlet_pressure": 101325.0}

        solver.set_boundary_conditions(boundary_conditions)

        assert solver.boundary_conditions == boundary_conditions
        # Check that inlet velocity is applied
        assert solver.velocity_field[0, :, :, 0].mean() == 1.0

    def test_solve_momentum(self, solver, domain, properties):
        """Test momentum equation solving."""
        solver.initialize(domain, properties)
        solver.set_boundary_conditions({"inlet_velocity": 1.0})

        dt = 0.001
        velocity_field = solver.solve_momentum(dt)

        assert isinstance(velocity_field, np.ndarray)
        assert velocity_field.shape == (10, 10, 10, 3)
        # Velocity should remain finite
        assert np.all(np.isfinite(velocity_field))

    def test_solve_continuity(self, solver, domain, properties):
        """Test continuity equation solving."""
        solver.initialize(domain, properties)
        solver.set_boundary_conditions({"inlet_velocity": 1.0})

        # First solve momentum to set up velocity field
        solver.solve_momentum(0.001)

        pressure_field = solver.solve_continuity()

        assert isinstance(pressure_field, np.ndarray)
        assert pressure_field.shape == (10, 10, 10)
        assert np.all(np.isfinite(pressure_field))

    def test_get_state(self, solver, domain, properties):
        """Test getting solver state."""
        solver.initialize(domain, properties)

        state = solver.get_state()

        assert isinstance(state, FluidState)
        assert state.velocity_field.shape == (10, 10, 10, 3)
        assert state.pressure_field.shape == (10, 10, 10)
        assert state.temperature_field.shape == (10, 10, 10)
        assert state.density_field.shape == (10, 10, 10)
        assert state.viscosity_field.shape == (10, 10, 10)

    def test_update_properties(self, solver, domain, properties):
        """Test updating fluid properties."""
        solver.initialize(domain, properties)

        new_properties = FluidProperties(density=1200.0, viscosity=0.002)

        solver.update_properties(new_properties)

        assert solver.properties == new_properties
        assert np.all(solver.density_field == 1200.0)
        assert np.all(solver.viscosity_field == 0.002)


class TestFiniteDifferenceHeatSolver:
    """Test FiniteDifferenceHeatSolver class."""

    @pytest.fixture
    def solver(self):
        """Create FiniteDifferenceHeatSolver instance."""
        return FiniteDifferenceHeatSolver()

    @pytest.fixture
    def domain(self):
        """Create FluidDomain instance."""
        return FluidDomain(dimensions=(0.1, 0.1, 0.1), mesh_resolution=(8, 8, 8))

    @pytest.fixture
    def properties(self):
        """Create FluidProperties instance."""
        return FluidProperties(
            density=1000.0, thermal_conductivity=0.6, specific_heat=4180.0
        )

    def test_initialization(self, solver, domain, properties):
        """Test heat solver initialization."""
        solver.initialize(domain, properties)

        assert solver.domain == domain
        assert solver.properties == properties
        assert solver.temperature_field.shape == (8, 8, 8)
        assert solver.heat_flux_field.shape == (8, 8, 8, 3)
        assert np.all(solver.temperature_field == 300.0)  # Initial room temperature

    def test_set_thermal_boundary_conditions(self, solver, domain, properties):
        """Test setting thermal boundary conditions."""
        solver.initialize(domain, properties)

        thermal_conditions = {"inlet_temperature": 373.15, "wall_temperature": 293.15}

        solver.set_thermal_boundary_conditions(thermal_conditions)

        assert solver.thermal_boundary_conditions == thermal_conditions
        # Check boundary application
        assert np.all(solver.temperature_field[0, :, :] == 373.15)  # Inlet
        assert np.all(solver.temperature_field[:, 0, :] == 293.15)  # Wall

    def test_solve_energy(self, solver, domain, properties):
        """Test energy equation solving."""
        solver.initialize(domain, properties)
        solver.set_thermal_boundary_conditions(
            {"inlet_temperature": 373.15, "wall_temperature": 293.15}
        )

        # Create sample velocity field
        velocity_field = np.zeros((8, 8, 8, 3))
        velocity_field[:, :, :, 0] = 0.1  # x-velocity

        dt = 0.001
        temperature_field = solver.solve_energy(dt, velocity_field)

        assert isinstance(temperature_field, np.ndarray)
        assert temperature_field.shape == (8, 8, 8)
        assert np.all(np.isfinite(temperature_field))

    def test_calculate_heat_flux(self, solver, domain, properties):
        """Test heat flux calculation."""
        solver.initialize(domain, properties)
        solver.set_thermal_boundary_conditions(
            {"inlet_temperature": 373.15, "wall_temperature": 293.15}
        )

        heat_flux_field = solver.calculate_heat_flux()

        assert isinstance(heat_flux_field, np.ndarray)
        assert heat_flux_field.shape == (8, 8, 8, 3)
        assert np.all(np.isfinite(heat_flux_field))


class TestFluidThermalCoupledSolver:
    """Test FluidThermalCoupledSolver class."""

    @pytest.fixture
    def fluid_solver(self):
        """Create mock fluid solver."""
        mock_solver = Mock(spec=SimpleFDFluidSolver)
        mock_solver.solve_momentum.return_value = np.zeros((5, 5, 5, 3))
        mock_solver.solve_continuity.return_value = np.zeros((5, 5, 5))
        mock_solver.get_state.return_value = FluidState(
            velocity_field=np.zeros((5, 5, 5, 3)),
            pressure_field=np.zeros((5, 5, 5)),
            temperature_field=np.full((5, 5, 5), 300.0),
            density_field=np.full((5, 5, 5), 1000.0),
            viscosity_field=np.full((5, 5, 5), 0.001),
        )
        return mock_solver

    @pytest.fixture
    def heat_solver(self):
        """Create mock heat solver."""
        mock_solver = Mock(spec=FiniteDifferenceHeatSolver)
        mock_solver.solve_energy.return_value = np.full((5, 5, 5), 310.0)
        mock_solver.get_temperature_field.return_value = np.full((5, 5, 5), 310.0)
        return mock_solver

    @pytest.fixture
    def coupled_solver(self, fluid_solver, heat_solver):
        """Create FluidThermalCoupledSolver instance."""
        return FluidThermalCoupledSolver(
            fluid_solver, heat_solver, CouplingScheme.SEGREGATED
        )

    @pytest.fixture
    def domain(self):
        """Create FluidDomain instance."""
        return FluidDomain(dimensions=(0.1, 0.1, 0.1), mesh_resolution=(5, 5, 5))

    @pytest.fixture
    def properties(self):
        """Create FluidProperties instance."""
        return FluidProperties()

    @pytest.fixture
    def process_conditions(self):
        """Create ProcessConditions instance."""
        return ProcessConditions(temperature=373.15, pressure=101325.0, flow_rate=0.1)

    def test_initialization(
        self, coupled_solver, domain, properties, process_conditions
    ):
        """Test coupled solver initialization."""
        coupled_solver.initialize(domain, properties, process_conditions)

        # Check that both solvers were initialized
        coupled_solver.fluid_solver.initialize.assert_called_once()
        coupled_solver.heat_solver.initialize.assert_called_once()

        # Check that boundary conditions were set
        coupled_solver.fluid_solver.set_boundary_conditions.assert_called_once()
        coupled_solver.heat_solver.set_thermal_boundary_conditions.assert_called_once()

    def test_solve_coupled_step_explicit(self, fluid_solver, heat_solver):
        """Test explicit coupling step."""
        coupled_solver = FluidThermalCoupledSolver(
            fluid_solver, heat_solver, CouplingScheme.EXPLICIT
        )

        dt = 0.001
        state = coupled_solver.solve_coupled_step(dt)

        assert isinstance(state, FluidState)
        fluid_solver.solve_momentum.assert_called_once_with(dt)
        fluid_solver.solve_continuity.assert_called_once()
        heat_solver.solve_energy.assert_called_once()

    def test_solve_coupled_step_segregated(self, fluid_solver, heat_solver):
        """Test segregated coupling step."""
        coupled_solver = FluidThermalCoupledSolver(
            fluid_solver, heat_solver, CouplingScheme.SEGREGATED
        )

        # Set convergence tolerance high for fast convergence in test
        coupled_solver.convergence_tolerance = 1.0

        dt = 0.001
        state = coupled_solver.solve_coupled_step(dt)

        assert isinstance(state, FluidState)
        # Should have called solvers at least once
        fluid_solver.solve_momentum.assert_called()
        fluid_solver.solve_continuity.assert_called()
        heat_solver.solve_energy.assert_called()

    def test_solve_coupled_step_unsupported_scheme(self, fluid_solver, heat_solver):
        """Test unsupported coupling scheme."""
        coupled_solver = FluidThermalCoupledSolver(
            fluid_solver, heat_solver, CouplingScheme.MONOLITHIC
        )

        with pytest.raises(ChemiAccelError):
            coupled_solver.solve_coupled_step(0.001)


class TestMaterialPropertyCoupling:
    """Test MaterialPropertyCoupling class."""

    @pytest.fixture
    def mock_multiscale_framework(self):
        """Create mock MultiScaleFramework."""
        mock_framework = Mock(spec=MultiScaleFramework)
        mock_framework.step.return_value = {
            "atomistic_model": {
                "density": 1100.0,
                "thermal_conductivity": 0.7,
                "elastic_modulus": 1.1e9,
            }
        }
        mock_framework.models = {"atomistic_model": {}}
        mock_framework.get_model_state.return_value = {"temperature": 300.0}
        return mock_framework

    @pytest.fixture
    def coupling(self, mock_multiscale_framework):
        """Create MaterialPropertyCoupling instance."""
        return MaterialPropertyCoupling(mock_multiscale_framework)

    @pytest.fixture
    def sample_fluid_state(self):
        """Create sample fluid state."""
        shape = (5, 5, 5)
        return FluidState(
            velocity_field=np.random.random(shape + (3,)),
            pressure_field=np.random.random(shape) * 1000,
            temperature_field=np.full(shape, 320.0),
            density_field=np.full(shape, 1000.0),
            viscosity_field=np.full(shape, 0.001),
            time=5.0,
        )

    def test_couple_with_multiscale(
        self, coupling, sample_fluid_state, mock_multiscale_framework
    ):
        """Test coupling with multiscale framework."""
        dt = 0.001

        properties = coupling.couple_with_multiscale(sample_fluid_state, dt)

        # Check that multiscale step was called
        mock_multiscale_framework.step.assert_called_once_with(dt)

        # Check that properties were extracted
        from src.chemi_accel.digital_twin.manufacturing_integration import (
            MaterialFlowProperties,
        )

        assert isinstance(properties, MaterialFlowProperties)

        # Check that coupling history was stored
        assert len(coupling.coupling_history) == 1
        assert coupling.coupling_history[0]["time"] == 5.0

    def test_extract_flow_conditions(self, coupling, sample_fluid_state):
        """Test flow conditions extraction."""
        conditions = coupling._extract_flow_conditions(sample_fluid_state)

        assert "average_shear_rate" in conditions
        assert "maximum_velocity" in conditions
        assert "average_temperature" in conditions
        assert "pressure_range" in conditions
        assert "flow_uniformity" in conditions

        # Check reasonable values
        assert conditions["average_temperature"] == 320.0
        assert conditions["maximum_velocity"] >= 0
        assert 0 <= conditions["flow_uniformity"] <= 1


class TestAdaptiveMeshRefinement:
    """Test AdaptiveMeshRefinement class."""

    @pytest.fixture
    def refinement(self):
        """Create AdaptiveMeshRefinement instance."""
        criteria = {"velocity_gradient": 50.0, "temperature_gradient": 5.0}
        return AdaptiveMeshRefinement(criteria)

    @pytest.fixture
    def sample_fluid_state(self):
        """Create sample fluid state with gradients."""
        shape = (10, 10, 10)

        # Create fields with some gradients
        velocity_field = np.zeros(shape + (3,))
        velocity_field[:, :, :, 0] = np.linspace(0, 2, shape[0])[:, None, None]

        temperature_field = np.zeros(shape)
        temperature_field[:, :, :] = 300 + np.linspace(0, 20, shape[0])[:, None, None]

        return FluidState(
            velocity_field=velocity_field,
            pressure_field=np.zeros(shape),
            temperature_field=temperature_field,
            density_field=np.full(shape, 1000.0),
            viscosity_field=np.full(shape, 0.001),
        )

    def test_analyze_mesh_quality(self, refinement, sample_fluid_state):
        """Test mesh quality analysis."""
        analysis = refinement.analyze_mesh_quality(sample_fluid_state)

        assert "velocity_gradients" in analysis
        assert "temperature_gradients" in analysis
        assert "refinement_regions" in analysis
        assert "mesh_quality_score" in analysis

        # Check types
        assert isinstance(analysis["velocity_gradients"], np.ndarray)
        assert isinstance(analysis["temperature_gradients"], np.ndarray)
        assert isinstance(analysis["refinement_regions"], np.ndarray)
        assert isinstance(analysis["mesh_quality_score"], float)

        # Check score range
        assert 0 <= analysis["mesh_quality_score"] <= 1


class TestFluidDynamicsCouplingManager:
    """Test FluidDynamicsCouplingManager class."""

    @pytest.fixture
    def manager(self):
        """Create FluidDynamicsCouplingManager instance."""
        return FluidDynamicsCouplingManager()

    @pytest.fixture
    def domain(self):
        """Create FluidDomain instance."""
        return FluidDomain(dimensions=(0.1, 0.1, 0.1), mesh_resolution=(5, 5, 5))

    @pytest.fixture
    def properties(self):
        """Create FluidProperties instance."""
        return FluidProperties()

    @pytest.fixture
    def process_conditions(self):
        """Create ProcessConditions instance."""
        return ProcessConditions(temperature=373.15, pressure=101325.0, flow_rate=0.1)

    @pytest.fixture
    def mock_multiscale_framework(self):
        """Create mock MultiScaleFramework."""
        mock_framework = Mock(spec=MultiScaleFramework)
        mock_framework.step.return_value = {"model1": {"property": 1.0}}
        mock_framework.models = {"model1": {}}
        mock_framework.get_model_state.return_value = {"temperature": 300.0}
        return mock_framework

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.coupled_solver is None
        assert manager.material_coupling is None
        assert manager.adaptive_refinement is None
        assert manager.simulation_history == []

    def test_setup_coupling(
        self, manager, domain, properties, process_conditions, mock_multiscale_framework
    ):
        """Test coupling setup."""
        manager.setup_coupling(
            domain, properties, process_conditions, mock_multiscale_framework
        )

        assert manager.coupled_solver is not None
        assert manager.material_coupling is not None
        assert manager.adaptive_refinement is not None

    def test_run_coupled_simulation_not_setup(self, manager):
        """Test running simulation without setup."""
        with pytest.raises(ChemiAccelError):
            manager.run_coupled_simulation(1.0, 0.1)

    @patch("src.chemi_accel.digital_twin.fluid_dynamics_coupling.SimpleFDFluidSolver")
    @patch(
        "src.chemi_accel.digital_twin.fluid_dynamics_coupling.FiniteDifferenceHeatSolver"
    )
    def test_run_coupled_simulation(
        self,
        mock_heat_solver,
        mock_fluid_solver,
        manager,
        domain,
        properties,
        process_conditions,
        mock_multiscale_framework,
    ):
        """Test running coupled simulation."""
        # Setup mocks
        mock_fluid_instance = Mock()
        mock_heat_instance = Mock()
        mock_fluid_solver.return_value = mock_fluid_instance
        mock_heat_solver.return_value = mock_heat_instance

        # Setup manager
        manager.setup_coupling(
            domain, properties, process_conditions, mock_multiscale_framework
        )

        # Mock coupled solver
        mock_state = FluidState(
            velocity_field=np.zeros((5, 5, 5, 3)),
            pressure_field=np.zeros((5, 5, 5)),
            temperature_field=np.full((5, 5, 5), 310.0),
            density_field=np.full((5, 5, 5), 1000.0),
            viscosity_field=np.full((5, 5, 5), 0.001),
        )
        manager.coupled_solver.solve_coupled_step = Mock(return_value=mock_state)

        # Run short simulation
        results = manager.run_coupled_simulation(
            total_time=0.01, dt=0.005, material_coupling_interval=1
        )

        assert len(results) == 2  # 0.01 / 0.005 = 2 steps
        assert len(manager.simulation_history) == 2

        # Check that coupled solver was called
        assert manager.coupled_solver.solve_coupled_step.call_count == 2

    def test_get_simulation_summary_empty(self, manager):
        """Test getting summary with no simulation data."""
        summary = manager.get_simulation_summary()

        assert "No simulation data available" in summary["message"]

    def test_get_simulation_summary_with_data(self, manager):
        """Test getting summary with simulation data."""
        # Add mock simulation data
        mock_result = {
            "time": 1.0,
            "flow_metrics": {"velocity_uniformity": 0.8},
            "mesh_analysis": None,
        }
        manager.simulation_history = [mock_result]

        # Mock coupled solver
        manager.coupled_solver = Mock()
        manager.coupled_solver.coupling_scheme.value = "segregated"

        # Mock material coupling
        manager.material_coupling = Mock()
        manager.material_coupling.coupling_history = [{"time": 1.0}]

        summary = manager.get_simulation_summary()

        assert summary["total_steps"] == 1
        assert summary["simulation_time"] == 1.0
        assert summary["latest_flow_metrics"]["velocity_uniformity"] == 0.8
        assert summary["coupling_scheme"] == "segregated"
        assert summary["material_coupling_points"] == 1


class TestIntegration:
    """Integration tests for fluid dynamics coupling."""

    def test_end_to_end_coupling_workflow(self):
        """Test complete coupling workflow."""
        # Create domain and properties
        domain = FluidDomain(dimensions=(0.1, 0.1, 0.1), mesh_resolution=(5, 5, 5))

        properties = FluidProperties(
            density=1000.0, viscosity=0.001, thermal_conductivity=0.6
        )

        conditions = ProcessConditions(
            temperature=373.15, pressure=101325.0, flow_rate=0.1
        )

        # Create mock multiscale framework
        mock_framework = Mock(spec=MultiScaleFramework)
        mock_framework.step.return_value = {"model1": {"density": 1100.0}}
        mock_framework.models = {"model1": {}}
        mock_framework.get_model_state.return_value = {"temperature": 300.0}

        # Create and setup manager
        manager = FluidDynamicsCouplingManager()
        manager.setup_coupling(domain, properties, conditions, mock_framework)

        # Verify setup
        assert manager.coupled_solver is not None
        assert manager.material_coupling is not None
        assert manager.adaptive_refinement is not None

        # Check that solvers were initialized
        assert manager.coupled_solver.fluid_solver is not None
        assert manager.coupled_solver.heat_solver is not None


if __name__ == "__main__":
    pytest.main([__file__])

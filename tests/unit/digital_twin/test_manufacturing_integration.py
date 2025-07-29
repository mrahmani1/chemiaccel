"""
Unit tests for manufacturing integration module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.chemi_accel.core.exceptions import ChemiAccelError
from src.chemi_accel.digital_twin.manufacturing_integration import (
    FluidDynamicsSimulator,
    HeatTransferSimulator,
    ManufacturingDigitalTwin,
    MaterialFlowProperties,
    ProcessConditions,
    ProcessOutput,
    ProcessType,
)
from src.chemi_accel.digital_twin.multiscale_framework import MultiScaleFramework


class TestProcessConditions:
    """Test ProcessConditions dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        conditions = ProcessConditions()

        assert conditions.temperature is None
        assert conditions.pressure is None
        assert conditions.flow_rate is None
        assert conditions.pressure_profile == []
        assert conditions.temperature_profile == []

    def test_full_initialization(self):
        """Test full initialization with all parameters."""
        conditions = ProcessConditions(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
            cooling_rate=5.0,
            feed_rate=10.0,
            screw_speed=100.0,
            melt_temperature=493.15,
            mold_temperature=343.15,
            cycle_time=120.0,
            pressure_profile=[1e5, 2e5, 1.5e5],
            temperature_profile=[473.15, 463.15, 453.15],
        )

        assert conditions.temperature == 473.15
        assert conditions.pressure == 2e5
        assert conditions.flow_rate == 0.1
        assert conditions.shear_rate == 100.0
        assert conditions.residence_time == 60.0
        assert conditions.cooling_rate == 5.0
        assert conditions.feed_rate == 10.0
        assert conditions.screw_speed == 100.0
        assert conditions.melt_temperature == 493.15
        assert conditions.mold_temperature == 343.15
        assert conditions.cycle_time == 120.0
        assert conditions.pressure_profile == [1e5, 2e5, 1.5e5]
        assert conditions.temperature_profile == [473.15, 463.15, 453.15]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        conditions = ProcessConditions(temperature=473.15, pressure=2e5, flow_rate=0.1)

        result = conditions.to_dict()

        assert isinstance(result, dict)
        assert result["temperature"] == 473.15
        assert result["pressure"] == 2e5
        assert result["flow_rate"] == 0.1
        assert result["shear_rate"] is None


class TestMaterialFlowProperties:
    """Test MaterialFlowProperties dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        properties = MaterialFlowProperties()

        assert properties.viscosity == 1000.0
        assert properties.density == 1000.0
        assert properties.thermal_conductivity == 0.2
        assert properties.specific_heat == 2000.0
        assert properties.thermal_expansion == 1e-4
        assert properties.elastic_modulus == 1e9
        assert properties.yield_strength == 50e6

    def test_update_with_temperature(self):
        """Test temperature-dependent property updates."""
        properties = MaterialFlowProperties(
            viscosity=1000.0, density=1000.0, thermal_expansion=1e-4
        )

        # Test at high temperature
        properties.update_with_temperature(373.15)  # 100°C

        # Viscosity should decrease with temperature
        assert properties.viscosity < 1000.0

        # Density should decrease with temperature
        assert properties.density < 1000.0


class TestProcessOutput:
    """Test ProcessOutput dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        output = ProcessOutput()

        assert output.part_quality == 0.0
        assert output.dimensional_accuracy == 0.0
        assert output.surface_finish == 0.0
        assert output.mechanical_properties == {}
        assert output.morphology == {}
        assert output.residual_stress == 0.0
        assert output.crystallinity == 0.0
        assert output.orientation == {}
        assert output.defects == []
        assert output.processing_time == 0.0
        assert output.energy_consumption == 0.0

    def test_calculate_overall_quality(self):
        """Test overall quality calculation."""
        # Test with positive quality factors
        output = ProcessOutput(
            part_quality=0.8, dimensional_accuracy=0.9, surface_finish=0.7
        )

        overall_quality = output.calculate_overall_quality()
        assert overall_quality == pytest.approx(0.8, abs=0.01)

        # Test with zero quality factors
        output_zero = ProcessOutput()
        overall_quality_zero = output_zero.calculate_overall_quality()
        assert overall_quality_zero == 0.0


class TestFluidDynamicsSimulator:
    """Test FluidDynamicsSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create FluidDynamicsSimulator instance."""
        return FluidDynamicsSimulator()

    @pytest.fixture
    def material_properties(self):
        """Create MaterialFlowProperties instance."""
        return MaterialFlowProperties(
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            specific_heat=2000.0,
        )

    @pytest.fixture
    def process_conditions(self):
        """Create ProcessConditions instance."""
        return ProcessConditions(
            temperature=473.15, pressure=2e5, flow_rate=0.1, cycle_time=60.0
        )

    def test_initialization(self, simulator):
        """Test proper initialization."""
        config = {"mesh_size": [20, 20, 20], "domain_size": [1.0, 1.0, 1.0]}

        simulator.initialize(config)

        assert simulator.mesh is not None
        assert simulator.velocity_field is not None
        assert simulator.pressure_field is not None
        assert simulator.temperature_field is not None
        assert simulator.velocity_field.shape == (20, 20, 20, 3)
        assert simulator.pressure_field.shape == (20, 20, 20)

    def test_set_material_properties(self, simulator, material_properties):
        """Test setting material properties."""
        config = {"mesh_size": [10, 10, 10]}
        simulator.initialize(config)

        simulator.set_material_properties(material_properties)

        assert simulator.material_properties == material_properties
        assert np.all(simulator.viscosity_field == material_properties.viscosity)

    def test_set_process_conditions(self, simulator, process_conditions):
        """Test setting process conditions."""
        config = {"mesh_size": [10, 10, 10]}
        simulator.initialize(config)

        simulator.set_process_conditions(process_conditions)

        assert simulator.process_conditions == process_conditions
        assert np.all(simulator.temperature_field == process_conditions.temperature)
        assert simulator.velocity_field[0, 0, 0, 0] > 0  # Inlet velocity set

    def test_run_simulation(self, simulator, material_properties, process_conditions):
        """Test running simulation."""
        config = {"mesh_size": [5, 5, 5]}
        simulator.initialize(config)
        simulator.set_material_properties(material_properties)
        simulator.set_process_conditions(process_conditions)

        output = simulator.run_simulation()

        assert isinstance(output, ProcessOutput)
        assert output.part_quality >= 0.0
        assert output.dimensional_accuracy >= 0.0
        assert output.processing_time > 0.0

    def test_run_simulation_without_setup(self, simulator):
        """Test running simulation without proper setup."""
        with pytest.raises(ChemiAccelError):
            simulator.run_simulation()

    def test_get_field_data(self, simulator):
        """Test getting field data."""
        config = {"mesh_size": [5, 5, 5]}
        simulator.initialize(config)

        velocity_data = simulator.get_field_data("velocity")
        pressure_data = simulator.get_field_data("pressure")
        temperature_data = simulator.get_field_data("temperature")

        assert isinstance(velocity_data, np.ndarray)
        assert isinstance(pressure_data, np.ndarray)
        assert isinstance(temperature_data, np.ndarray)
        assert velocity_data.shape == (5, 5, 5, 3)
        assert pressure_data.shape == (5, 5, 5)
        assert temperature_data.shape == (5, 5, 5)

    def test_get_field_data_invalid_field(self, simulator):
        """Test getting invalid field data."""
        config = {"mesh_size": [5, 5, 5]}
        simulator.initialize(config)

        with pytest.raises(ValueError):
            simulator.get_field_data("invalid_field")


class TestHeatTransferSimulator:
    """Test HeatTransferSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create HeatTransferSimulator instance."""
        return HeatTransferSimulator()

    @pytest.fixture
    def material_properties(self):
        """Create MaterialFlowProperties instance."""
        return MaterialFlowProperties(
            thermal_conductivity=0.2, density=1000.0, specific_heat=2000.0
        )

    @pytest.fixture
    def process_conditions(self):
        """Create ProcessConditions instance."""
        return ProcessConditions(
            temperature=473.15, mold_temperature=343.15, cycle_time=120.0
        )

    def test_initialization(self, simulator):
        """Test proper initialization."""
        config = {"mesh_size": [15, 15, 15], "domain_size": [0.1, 0.1, 0.1]}

        simulator.initialize(config)

        assert simulator.mesh is not None
        assert simulator.temperature_field is not None
        assert simulator.heat_flux_field is not None
        assert simulator.temperature_field.shape == (15, 15, 15)
        assert simulator.heat_flux_field.shape == (15, 15, 15, 3)

    def test_run_simulation(self, simulator, material_properties, process_conditions):
        """Test running heat transfer simulation."""
        config = {"mesh_size": [5, 5, 5]}
        simulator.initialize(config)
        simulator.set_material_properties(material_properties)
        simulator.set_process_conditions(process_conditions)

        output = simulator.run_simulation()

        assert isinstance(output, ProcessOutput)
        assert output.part_quality >= 0.0
        assert output.processing_time > 0.0

        # Check that temperature field has been modified
        temp_data = simulator.get_field_data("temperature")
        assert np.any(temp_data != 293.15)  # Should differ from initial room temp

    def test_get_field_data(self, simulator):
        """Test getting field data."""
        config = {"mesh_size": [5, 5, 5]}
        simulator.initialize(config)

        temperature_data = simulator.get_field_data("temperature")
        heat_flux_data = simulator.get_field_data("heat_flux")

        assert isinstance(temperature_data, np.ndarray)
        assert isinstance(heat_flux_data, np.ndarray)
        assert temperature_data.shape == (5, 5, 5)
        assert heat_flux_data.shape == (5, 5, 5, 3)


class TestManufacturingDigitalTwin:
    """Test ManufacturingDigitalTwin class."""

    @pytest.fixture
    def mock_multiscale_framework(self):
        """Create mock MultiScaleFramework."""
        mock_framework = Mock(spec=MultiScaleFramework)
        mock_framework.get_material_properties.return_value = {
            "atomistic_model": {
                "density": 1200.0,
                "thermal_conductivity": 0.25,
                "elastic_modulus": 1.2e9,
            }
        }
        return mock_framework

    @pytest.fixture
    def digital_twin(self, mock_multiscale_framework):
        """Create ManufacturingDigitalTwin instance."""
        return ManufacturingDigitalTwin(mock_multiscale_framework)

    @pytest.fixture
    def mock_simulator(self):
        """Create mock process simulator."""
        mock_sim = Mock()
        mock_sim.initialize = Mock()
        mock_sim.set_material_properties = Mock()
        mock_sim.set_process_conditions = Mock()
        mock_sim.run_simulation.return_value = ProcessOutput(
            part_quality=0.8,
            dimensional_accuracy=0.85,
            surface_finish=0.75,
            processing_time=60.0,
        )
        return mock_sim

    def test_initialization(self, digital_twin, mock_multiscale_framework):
        """Test proper initialization."""
        assert digital_twin.multiscale_framework == mock_multiscale_framework
        assert digital_twin.process_simulators == {}
        assert digital_twin.process_conditions is None
        assert isinstance(digital_twin.material_flow_properties, MaterialFlowProperties)
        assert digital_twin.process_history == []

    def test_add_process_simulator(self, digital_twin, mock_simulator):
        """Test adding process simulator."""
        simulator_id = digital_twin.add_process_simulator(
            ProcessType.EXTRUSION, mock_simulator
        )

        assert simulator_id == "extrusion_simulator"
        assert simulator_id in digital_twin.process_simulators
        assert (
            digital_twin.process_simulators[simulator_id]["simulator"] == mock_simulator
        )
        assert (
            digital_twin.process_simulators[simulator_id]["process_type"]
            == ProcessType.EXTRUSION
        )
        assert digital_twin.process_simulators[simulator_id]["active"] is True

    def test_add_process_simulator_custom_id(self, digital_twin, mock_simulator):
        """Test adding process simulator with custom ID."""
        custom_id = "my_custom_simulator"
        simulator_id = digital_twin.add_process_simulator(
            ProcessType.INJECTION_MOLDING, mock_simulator, custom_id
        )

        assert simulator_id == custom_id
        assert custom_id in digital_twin.process_simulators

    def test_set_process_conditions(self, digital_twin, mock_simulator):
        """Test setting process conditions."""
        digital_twin.add_process_simulator(ProcessType.EXTRUSION, mock_simulator)

        conditions = ProcessConditions(temperature=473.15, pressure=2e5, flow_rate=0.1)

        digital_twin.set_process_conditions(conditions)

        assert digital_twin.process_conditions == conditions
        mock_simulator.set_process_conditions.assert_called_once_with(conditions)
        mock_simulator.set_material_properties.assert_called_once()

    def test_run_manufacturing_simulation(self, digital_twin, mock_simulator):
        """Test running manufacturing simulation."""
        digital_twin.add_process_simulator(ProcessType.EXTRUSION, mock_simulator)
        digital_twin.set_process_conditions(ProcessConditions(temperature=473.15))

        output = digital_twin.run_manufacturing_simulation(ProcessType.EXTRUSION)

        assert isinstance(output, ProcessOutput)
        assert output.part_quality == 0.8
        assert len(digital_twin.process_history) == 1
        mock_simulator.run_simulation.assert_called_once()

    def test_run_manufacturing_simulation_no_conditions(
        self, digital_twin, mock_simulator
    ):
        """Test running simulation without conditions."""
        digital_twin.add_process_simulator(ProcessType.EXTRUSION, mock_simulator)

        with pytest.raises(ChemiAccelError):
            digital_twin.run_manufacturing_simulation(ProcessType.EXTRUSION)

    def test_run_manufacturing_simulation_no_simulator(self, digital_twin):
        """Test running simulation without appropriate simulator."""
        digital_twin.set_process_conditions(ProcessConditions(temperature=473.15))

        with pytest.raises(ChemiAccelError):
            digital_twin.run_manufacturing_simulation(ProcessType.EXTRUSION)

    def test_run_coupled_simulation(
        self, digital_twin, mock_simulator, mock_multiscale_framework
    ):
        """Test running coupled simulation."""
        digital_twin.add_process_simulator(ProcessType.EXTRUSION, mock_simulator)
        digital_twin.set_process_conditions(ProcessConditions(temperature=473.15))

        # Mock multiscale framework step
        mock_multiscale_framework.step.return_value = {
            "atomistic_model": {"energy": 100.0, "temperature": 473.15}
        }

        multiscale_results, process_output = digital_twin.run_coupled_simulation(
            multiscale_dt=0.001, multiscale_steps=10, process_type=ProcessType.EXTRUSION
        )

        assert len(multiscale_results) == 10
        assert isinstance(process_output, ProcessOutput)
        assert mock_multiscale_framework.step.call_count == 10
        mock_simulator.run_simulation.assert_called_once()

    def test_optimize_process_conditions(self, digital_twin, mock_simulator):
        """Test process condition optimization."""
        digital_twin.add_process_simulator(ProcessType.EXTRUSION, mock_simulator)

        target_properties = {"part_quality": 0.9, "dimensional_accuracy": 0.85}

        parameter_ranges = {
            "temperature": (450.0, 500.0),
            "pressure": (1e5, 3e5),
            "flow_rate": (0.05, 0.2),
        }

        best_conditions, best_output = digital_twin.optimize_process_conditions(
            ProcessType.EXTRUSION, target_properties, parameter_ranges, num_iterations=5
        )

        assert isinstance(best_conditions, ProcessConditions)
        assert isinstance(best_output, ProcessOutput)
        assert best_conditions.temperature is not None
        assert best_conditions.pressure is not None
        assert best_conditions.flow_rate is not None
        assert mock_simulator.run_simulation.call_count == 5

    def test_get_process_field_data(self, digital_twin, mock_simulator):
        """Test getting process field data."""
        simulator_id = digital_twin.add_process_simulator(
            ProcessType.EXTRUSION, mock_simulator
        )

        # Mock field data
        mock_field_data = np.random.random((10, 10, 10))
        mock_simulator.get_field_data.return_value = mock_field_data

        field_data = digital_twin.get_process_field_data(simulator_id, "velocity")

        assert np.array_equal(field_data, mock_field_data)
        mock_simulator.get_field_data.assert_called_once_with("velocity")

    def test_get_process_field_data_invalid_simulator(self, digital_twin):
        """Test getting field data from invalid simulator."""
        with pytest.raises(ValueError):
            digital_twin.get_process_field_data("invalid_simulator", "velocity")

    def test_get_manufacturing_summary(self, digital_twin, mock_simulator):
        """Test getting manufacturing summary."""
        # Test with no simulations
        summary = digital_twin.get_manufacturing_summary()
        assert "No manufacturing simulations run yet" in summary["message"]

        # Test with simulation data
        digital_twin.add_process_simulator(ProcessType.EXTRUSION, mock_simulator)
        digital_twin.set_process_conditions(ProcessConditions(temperature=473.15))
        digital_twin.run_manufacturing_simulation(ProcessType.EXTRUSION)

        summary = digital_twin.get_manufacturing_summary()

        assert summary["total_simulations"] == 1
        assert summary["latest_process"] == "extrusion"
        assert summary["latest_quality"] == 0.8
        assert summary["latest_processing_time"] == 60.0
        assert summary["average_quality"] == 0.8
        assert "extrusion_simulator" in summary["available_simulators"]
        assert summary["current_conditions"] is not None


class TestIntegration:
    """Integration tests for manufacturing integration."""

    def test_end_to_end_manufacturing_workflow(self):
        """Test complete manufacturing workflow."""
        # Create components
        mock_framework = Mock(spec=MultiScaleFramework)
        mock_framework.get_material_properties.return_value = {
            "continuum_model": {"density": 1100.0, "elastic_modulus": 1.1e9}
        }
        mock_framework.step.return_value = {"continuum_model": {"stress": 1e6}}

        digital_twin = ManufacturingDigitalTwin(mock_framework)

        # Add simulators
        fluid_sim = FluidDynamicsSimulator()
        heat_sim = HeatTransferSimulator()

        digital_twin.add_process_simulator(
            ProcessType.EXTRUSION, fluid_sim, "fluid_sim"
        )
        digital_twin.add_process_simulator(ProcessType.EXTRUSION, heat_sim, "heat_sim")

        # Set conditions
        conditions = ProcessConditions(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            mold_temperature=343.15,
            cycle_time=60.0,
        )
        digital_twin.set_process_conditions(conditions)

        # Initialize simulators
        config = {"mesh_size": [5, 5, 5], "domain_size": [0.1, 0.1, 0.1]}
        fluid_sim.initialize(config)
        heat_sim.initialize(config)

        # Run simulations
        fluid_output = digital_twin.run_manufacturing_simulation(ProcessType.EXTRUSION)

        # Verify results
        assert isinstance(fluid_output, ProcessOutput)
        assert fluid_output.part_quality >= 0.0
        assert len(digital_twin.process_history) >= 1

        # Get summary
        summary = digital_twin.get_manufacturing_summary()
        assert summary["total_simulations"] >= 1
        assert len(summary["available_simulators"]) == 2


if __name__ == "__main__":
    pytest.main([__file__])

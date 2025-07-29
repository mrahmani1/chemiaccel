"""
Unit tests for real-time process simulation module.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch

from src.chemi_accel.core.exceptions import ChemiAccelError
from src.chemi_accel.digital_twin.manufacturing_integration import (
    ProcessConditions,
    ProcessType,
)
from src.chemi_accel.digital_twin.multiscale_framework import MultiScaleFramework
from src.chemi_accel.digital_twin.realtime_process_simulation import (
    MaterialProperty,
    MechanicalPropertyModel,
    MicrostructureState,
    PropertyType,
    RealTimeSimulationCore,
    SimulationConfig,
    SimulationMode,
    SimulationMonitor,
    SimulationState,
    ThermalPropertyModel,
    UpdateStrategy,
)


class TestSimulationConfig:
    """Test SimulationConfig dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        config = SimulationConfig()

        assert config.mode == SimulationMode.REAL_TIME
        assert config.time_step == 0.01
        assert config.max_simulation_time == 3600.0
        assert config.enable_gpu_acceleration == True
        assert config.adaptive_time_stepping == True

    def test_custom_initialization(self):
        """Test custom initialization."""
        config = SimulationConfig(
            mode=SimulationMode.ACCELERATED,
            time_step=0.1,
            max_simulation_time=1800.0,
            enable_gpu_acceleration=False,
        )

        assert config.mode == SimulationMode.ACCELERATED
        assert config.time_step == 0.1
        assert config.max_simulation_time == 1800.0
        assert config.enable_gpu_acceleration == False


class TestMaterialProperty:
    """Test MaterialProperty dataclass."""

    def test_initialization(self):
        """Test property initialization."""
        prop = MaterialProperty(
            "tensile_strength", PropertyType.MECHANICAL, 100.0, "MPa", uncertainty=5.0
        )

        assert prop.name == "tensile_strength"
        assert prop.property_type == PropertyType.MECHANICAL
        assert prop.value == 100.0
        assert prop.units == "MPa"
        assert prop.uncertainty == 5.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        prop = MaterialProperty(
            "thermal_conductivity", PropertyType.THERMAL, 0.2, "W/m·K"
        )

        prop_dict = prop.to_dict()

        assert prop_dict["name"] == "thermal_conductivity"
        assert prop_dict["property_type"] == "thermal"
        assert prop_dict["value"] == 0.2
        assert prop_dict["units"] == "W/m·K"

    def test_array_value_serialization(self):
        """Test serialization with array values."""
        prop = MaterialProperty(
            "stress_field", PropertyType.MECHANICAL, np.array([1.0, 2.0, 3.0]), "MPa"
        )

        prop_dict = prop.to_dict()
        assert prop_dict["value"] == [1.0, 2.0, 3.0]


class TestMicrostructureState:
    """Test MicrostructureState dataclass."""

    @pytest.fixture
    def sample_microstructure(self):
        """Create sample microstructure state."""
        return MicrostructureState(
            grain_size=10.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 0.7, "beta": 0.3},
            defect_density=0.01,
            crystallinity=0.8,
        )

    def test_initialization(self, sample_microstructure):
        """Test microstructure initialization."""
        assert sample_microstructure.grain_size == 10.0
        assert np.array_equal(sample_microstructure.grain_orientation, [1.0, 0.0, 0.0])
        assert sample_microstructure.phase_fractions["alpha"] == 0.7
        assert sample_microstructure.defect_density == 0.01
        assert sample_microstructure.crystallinity == 0.8

    def test_evolve(self, sample_microstructure):
        """Test microstructure evolution."""
        process_conditions = ProcessConditions(
            temperature=500.0, pressure=2e5, material_composition={"polymer": 1.0}
        )

        evolved = sample_microstructure.evolve(process_conditions, 0.1)

        assert isinstance(evolved, MicrostructureState)
        assert evolved.grain_size >= sample_microstructure.grain_size  # Should grow
        assert evolved.crystallinity >= 0.0  # Should remain positive
        assert evolved.defect_density >= 0.0  # Should remain positive

    def test_phase_fraction_normalization(self, sample_microstructure):
        """Test phase fraction normalization during evolution."""
        process_conditions = ProcessConditions(
            temperature=600.0, pressure=3e5, material_composition={"polymer": 1.0}
        )

        evolved = sample_microstructure.evolve(process_conditions, 1.0)

        # Phase fractions should sum to approximately 1
        total_fraction = sum(evolved.phase_fractions.values())
        assert abs(total_fraction - 1.0) < 0.01


class TestMechanicalPropertyModel:
    """Test MechanicalPropertyModel class."""

    @pytest.fixture
    def property_model(self):
        """Create MechanicalPropertyModel instance."""
        return MechanicalPropertyModel(device="cpu")

    @pytest.fixture
    def sample_properties(self):
        """Create sample material properties."""
        return {
            "tensile_strength": MaterialProperty(
                "tensile_strength", PropertyType.MECHANICAL, 100.0, "MPa"
            ),
            "elastic_modulus": MaterialProperty(
                "elastic_modulus", PropertyType.MECHANICAL, 2000.0, "MPa"
            ),
        }

    @pytest.fixture
    def sample_conditions(self):
        """Create sample process conditions."""
        return ProcessConditions(
            temperature=473.15, pressure=2e5, material_composition={"polymer": 1.0}
        )

    @pytest.fixture
    def sample_microstructure(self):
        """Create sample microstructure."""
        return MicrostructureState(
            grain_size=10.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 0.7, "beta": 0.3},
            defect_density=0.01,
            crystallinity=0.8,
        )

    def test_initialization(self, property_model):
        """Test model initialization."""
        assert hasattr(property_model, "strength_model")
        assert hasattr(property_model, "modulus_model")
        assert property_model.device == "cpu"

    def test_predict_evolution(
        self,
        property_model,
        sample_properties,
        sample_conditions,
        sample_microstructure,
    ):
        """Test property evolution prediction."""
        evolved_properties = property_model.predict_evolution(
            sample_properties, sample_conditions, sample_microstructure, 0.1
        )

        assert "tensile_strength" in evolved_properties
        assert "elastic_modulus" in evolved_properties

        # Check that values are reasonable
        strength_prop = evolved_properties["tensile_strength"]
        modulus_prop = evolved_properties["elastic_modulus"]

        assert strength_prop.value >= 0
        assert modulus_prop.value >= 0
        assert strength_prop.uncertainty is not None
        assert modulus_prop.uncertainty is not None

    def test_get_property_types(self, property_model):
        """Test property types."""
        types = property_model.get_property_types()
        assert PropertyType.MECHANICAL in types


class TestThermalPropertyModel:
    """Test ThermalPropertyModel class."""

    @pytest.fixture
    def property_model(self):
        """Create ThermalPropertyModel instance."""
        return ThermalPropertyModel()

    def test_predict_evolution(self, property_model):
        """Test thermal property evolution prediction."""
        properties = {
            "thermal_conductivity": MaterialProperty(
                "thermal_conductivity", PropertyType.THERMAL, 0.2, "W/m·K"
            ),
            "heat_capacity": MaterialProperty(
                "heat_capacity", PropertyType.THERMAL, 1000.0, "J/kg·K"
            ),
        }

        conditions = ProcessConditions(
            temperature=400.0, pressure=1e5, material_composition={"polymer": 1.0}
        )

        microstructure = MicrostructureState(
            grain_size=5.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 1.0},
            defect_density=0.001,
            crystallinity=0.9,
        )

        evolved_properties = property_model.predict_evolution(
            properties, conditions, microstructure, 0.05
        )

        assert "thermal_conductivity" in evolved_properties
        assert "heat_capacity" in evolved_properties

        # Check that values are reasonable
        conductivity = evolved_properties["thermal_conductivity"]
        capacity = evolved_properties["heat_capacity"]

        assert conductivity.value > 0
        assert capacity.value > 0

    def test_get_property_types(self, property_model):
        """Test property types."""
        types = property_model.get_property_types()
        assert PropertyType.THERMAL in types


class TestRealTimeSimulationCore:
    """Test RealTimeSimulationCore class."""

    @pytest.fixture
    def simulation_config(self):
        """Create simulation configuration."""
        return SimulationConfig(
            mode=SimulationMode.REAL_TIME,
            time_step=0.01,
            max_simulation_time=1.0,  # Short for testing
            enable_gpu_acceleration=False,
        )

    @pytest.fixture
    def mock_multiscale_framework(self):
        """Create mock multiscale framework."""
        return Mock(spec=MultiScaleFramework)

    @pytest.fixture
    def simulation_core(self, simulation_config, mock_multiscale_framework):
        """Create RealTimeSimulationCore instance."""
        return RealTimeSimulationCore(simulation_config, mock_multiscale_framework)

    @pytest.fixture
    def initial_conditions(self):
        """Create initial process conditions."""
        return ProcessConditions(
            temperature=298.15, pressure=1e5, material_composition={"polymer": 1.0}
        )

    @pytest.fixture
    def initial_microstructure(self):
        """Create initial microstructure."""
        return MicrostructureState(
            grain_size=5.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 1.0},
            defect_density=0.001,
            crystallinity=0.8,
        )

    def test_initialization(self, simulation_core, simulation_config):
        """Test core initialization."""
        assert simulation_core.config == simulation_config
        assert len(simulation_core.property_models) > 0
        assert simulation_core.current_state is None
        assert not simulation_core.is_running

    def test_initialize_simulation(
        self, simulation_core, initial_conditions, initial_microstructure
    ):
        """Test simulation initialization."""
        state = simulation_core.initialize_simulation(
            initial_conditions, initial_microstructure
        )

        assert isinstance(state, SimulationState)
        assert state.time == 0.0
        assert state.process_conditions == initial_conditions
        assert state.microstructure == initial_microstructure
        assert len(state.material_properties) > 0

        assert simulation_core.current_state == state
        assert len(simulation_core.simulation_history) == 1

    @pytest.mark.asyncio
    async def test_run_simulation(
        self, simulation_core, initial_conditions, initial_microstructure
    ):
        """Test simulation execution."""
        # Initialize simulation
        simulation_core.initialize_simulation(
            initial_conditions, initial_microstructure
        )

        # Run simulation for short duration
        history = await simulation_core.run_simulation(duration=0.1)

        assert len(history) > 1
        assert all(isinstance(state, SimulationState) for state in history)

        # Check time progression
        times = [state.time for state in history]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    def test_add_update_callback(self, simulation_core):
        """Test adding update callbacks."""
        callback_called = []

        def test_callback(state):
            callback_called.append(state.time)

        simulation_core.add_update_callback(test_callback)
        assert test_callback in simulation_core.update_callbacks

    def test_remove_update_callback(self, simulation_core):
        """Test removing update callbacks."""

        def test_callback(state):
            pass

        simulation_core.add_update_callback(test_callback)
        simulation_core.remove_update_callback(test_callback)
        assert test_callback not in simulation_core.update_callbacks

    def test_pause_resume_simulation(self, simulation_core):
        """Test pausing and resuming simulation."""
        simulation_core.pause_simulation()
        assert not simulation_core.is_running

        simulation_core.resume_simulation()
        assert simulation_core.is_running

    def test_get_property_evolution(
        self, simulation_core, initial_conditions, initial_microstructure
    ):
        """Test property evolution tracking."""
        # Initialize and add some history
        simulation_core.initialize_simulation(
            initial_conditions, initial_microstructure
        )

        # Add some fake history
        for i in range(5):
            state = SimulationState(
                time=i * 0.1,
                process_conditions=initial_conditions,
                material_properties={
                    "tensile_strength": MaterialProperty(
                        "tensile_strength", PropertyType.MECHANICAL, 100.0 + i, "MPa"
                    )
                },
                microstructure=initial_microstructure,
            )
            simulation_core.simulation_history.append(state)

        times, values = simulation_core.get_property_evolution("tensile_strength")

        assert len(times) == len(values)
        assert len(times) > 0
        assert all(isinstance(t, float) for t in times)
        assert all(isinstance(v, (int, float)) for v in values)

    def test_get_microstructure_evolution(
        self, simulation_core, initial_conditions, initial_microstructure
    ):
        """Test microstructure evolution tracking."""
        # Initialize and add some history
        simulation_core.initialize_simulation(
            initial_conditions, initial_microstructure
        )

        times, values = simulation_core.get_microstructure_evolution("grain_size")

        assert len(times) == len(values)
        assert len(times) > 0

    def test_export_trajectory(
        self, simulation_core, initial_conditions, initial_microstructure, tmp_path
    ):
        """Test trajectory export."""
        # Initialize simulation
        simulation_core.initialize_simulation(
            initial_conditions, initial_microstructure
        )

        # Export trajectory
        filepath = tmp_path / "trajectory.json"
        simulation_core.export_trajectory(str(filepath))

        assert filepath.exists()

        # Check file content
        import json

        with open(filepath) as f:
            data = json.load(f)

        assert "config" in data
        assert "performance_metrics" in data
        assert "history" in data

    def test_adaptive_time_stepping(
        self, simulation_core, initial_conditions, initial_microstructure
    ):
        """Test adaptive time stepping."""
        # Enable adaptive time stepping
        simulation_core.config.adaptive_time_stepping = True

        # Initialize simulation
        simulation_core.initialize_simulation(
            initial_conditions, initial_microstructure
        )

        # Add some history to enable adaptive stepping
        for i in range(3):
            state = SimulationState(
                time=i * 0.01,
                process_conditions=initial_conditions,
                material_properties={
                    "tensile_strength": MaterialProperty(
                        "tensile_strength",
                        PropertyType.MECHANICAL,
                        100.0 + i * 10,
                        "MPa",
                    )
                },
                microstructure=initial_microstructure,
            )
            simulation_core.simulation_history.append(state)

        # Calculate time step
        time_step = simulation_core._calculate_time_step()

        assert isinstance(time_step, float)
        assert time_step > 0
        assert (
            simulation_core.config.min_time_step
            <= time_step
            <= simulation_core.config.max_time_step
        )


class TestSimulationMonitor:
    """Test SimulationMonitor class."""

    @pytest.fixture
    def simulation_core(self):
        """Create mock simulation core."""
        mock_core = Mock(spec=RealTimeSimulationCore)
        mock_core.get_performance_metrics.return_value = {
            "time_ratio": 0.5,
            "updates_per_second": 10.0,
            "memory_usage": 500000,
        }
        return mock_core

    @pytest.fixture
    def monitor(self, simulation_core):
        """Create SimulationMonitor instance."""
        return SimulationMonitor(simulation_core)

    def test_initialization(self, monitor, simulation_core):
        """Test monitor initialization."""
        assert monitor.simulation_core == simulation_core
        assert len(monitor.performance_history) == 0
        assert "time_ratio" in monitor.alert_thresholds

    def test_record_performance(self, monitor):
        """Test performance recording."""
        # Create mock state
        mock_state = Mock(spec=SimulationState)

        # Trigger recording
        monitor._record_performance(mock_state)

        assert len(monitor.performance_history) == 1
        assert "timestamp" in monitor.performance_history[0]

    def test_performance_summary(self, monitor):
        """Test performance summary generation."""
        # Add some mock performance data
        for i in range(5):
            monitor.performance_history.append(
                {
                    "time_ratio": 0.5 + i * 0.1,
                    "updates_per_second": 10.0 + i,
                    "memory_usage": 500000 + i * 10000,
                    "timestamp": time.time(),
                }
            )

        summary = monitor.get_performance_summary()

        assert "time_ratio" in summary
        assert "updates_per_second" in summary
        assert "memory_usage" in summary

        # Check that each metric has statistical data
        for metric in ["time_ratio", "updates_per_second", "memory_usage"]:
            assert "current" in summary[metric]
            assert "average" in summary[metric]
            assert "min" in summary[metric]
            assert "max" in summary[metric]


if __name__ == "__main__":
    pytest.main([__file__])

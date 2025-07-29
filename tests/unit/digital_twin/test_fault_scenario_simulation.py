"""
Unit tests for fault scenario simulation module.
"""

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from src.chemi_accel.digital_twin.fault_scenario_simulation import (
    AlternativeRouteOptimizer,
    CorrectiveAction,
    CorrectiveActionType,
    DecisionSupportSystem,
    FaultAnalysisResult,
    FaultCause,
    FaultDetector,
    FaultInjector,
    FaultScenario,
    FaultScenarioSimulator,
    FaultSeverity,
    FaultType,
    create_contamination_fault,
    create_pressure_fault,
    create_temperature_fault,
)
from src.chemi_accel.digital_twin.manufacturing_integration import (
    ProcessConditions,
    ProcessOutput,
)
from src.chemi_accel.digital_twin.predictive_control import ModelPredictiveController
from src.chemi_accel.digital_twin.realtime_process_simulation import (
    MaterialProperty,
    MicrostructureState,
    PropertyType,
    RealTimeSimulationCore,
    SimulationState,
)
from src.chemi_accel.digital_twin.sensor_data_integration import (
    DataQuality,
    SensorReading,
)


class TestFaultScenario:
    """Test FaultScenario dataclass."""

    def test_initialization(self):
        """Test fault scenario initialization."""
        scenario = FaultScenario(
            fault_id="test_fault",
            fault_type=FaultType.TEMPERATURE_DEVIATION,
            severity=FaultSeverity.MAJOR,
            description="Test temperature fault",
            affected_parameters=["temperature"],
            fault_magnitude=0.7,
            onset_time=60.0,
        )

        assert scenario.fault_id == "test_fault"
        assert scenario.fault_type == FaultType.TEMPERATURE_DEVIATION
        assert scenario.severity == FaultSeverity.MAJOR
        assert scenario.fault_magnitude == 0.7
        assert scenario.onset_time == 60.0
        assert scenario.root_cause == FaultCause.UNKNOWN

    def test_to_dict(self):
        """Test dictionary conversion."""
        scenario = FaultScenario(
            fault_id="test_fault",
            fault_type=FaultType.CONTAMINATION,
            severity=FaultSeverity.CRITICAL,
            description="Test contamination",
            affected_parameters=["composition"],
            fault_magnitude=0.9,
            onset_time=30.0,
            duration=300.0,
            root_cause=FaultCause.MATERIAL_VARIATION,
        )

        scenario_dict = scenario.to_dict()

        assert scenario_dict["fault_id"] == "test_fault"
        assert scenario_dict["fault_type"] == "contamination"
        assert scenario_dict["severity"] == "critical"
        assert scenario_dict["fault_magnitude"] == 0.9
        assert scenario_dict["duration"] == 300.0
        assert scenario_dict["root_cause"] == "material_variation"


class TestCorrectiveAction:
    """Test CorrectiveAction dataclass."""

    def test_initialization(self):
        """Test corrective action initialization."""
        action = CorrectiveAction(
            action_id="test_action",
            action_type=CorrectiveActionType.PARAMETER_ADJUSTMENT,
            target_parameter="temperature",
            target_value=350.0,
            current_value=300.0,
            priority=1,
            estimated_effectiveness=0.8,
            implementation_time=60.0,
            cost_estimate=100.0,
            risk_level=0.2,
            description="Increase temperature to restore normal operation",
        )

        assert action.action_id == "test_action"
        assert action.action_type == CorrectiveActionType.PARAMETER_ADJUSTMENT
        assert action.target_parameter == "temperature"
        assert action.target_value == 350.0
        assert action.priority == 1
        assert action.estimated_effectiveness == 0.8

    def test_to_dict(self):
        """Test dictionary conversion."""
        action = CorrectiveAction(
            action_id="test_action",
            action_type=CorrectiveActionType.EMERGENCY_SHUTDOWN,
            target_parameter="system_power",
            target_value=0.0,
            current_value=1.0,
            priority=1,
            estimated_effectiveness=1.0,
            implementation_time=10.0,
            cost_estimate=1000.0,
            risk_level=0.8,
            description="Emergency shutdown",
            success_criteria={"power_level": 0.0},
        )

        action_dict = action.to_dict()

        assert action_dict["action_id"] == "test_action"
        assert action_dict["action_type"] == "emergency_shutdown"
        assert action_dict["success_criteria"]["power_level"] == 0.0


class TestFaultInjector:
    """Test FaultInjector class."""

    @pytest.fixture
    def fault_injector(self):
        """Create FaultInjector instance."""
        return FaultInjector()

    @pytest.fixture
    def mock_simulation_state(self):
        """Create mock simulation state."""
        process_conditions = ProcessConditions(
            temperature=300.0,
            pressure=5e5,
            material_composition={"polymer": 0.8, "additive": 0.2},
        )

        microstructure = MicrostructureState(
            grain_size=5.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 1.0},
            defect_density=0.001,
            crystallinity=0.8,
        )

        return SimulationState(
            time=0.0,
            process_conditions=process_conditions,
            material_properties={},
            microstructure=microstructure,
        )

    @pytest.fixture
    def temperature_fault_scenario(self):
        """Create temperature fault scenario."""
        return FaultScenario(
            fault_id="temp_drop_test",
            fault_type=FaultType.TEMPERATURE_DEVIATION,
            severity=FaultSeverity.MAJOR,
            description="Temperature drop test",
            affected_parameters=["temperature"],
            fault_magnitude=0.5,
            onset_time=0.0,
        )

    def test_initialization(self, fault_injector):
        """Test fault injector initialization."""
        assert len(fault_injector.active_faults) == 0
        assert len(fault_injector.fault_history) == 0
        assert len(fault_injector.injection_callbacks) == 0

    def test_inject_temperature_fault(
        self, fault_injector, mock_simulation_state, temperature_fault_scenario
    ):
        """Test temperature fault injection."""
        original_temp = mock_simulation_state.process_conditions.temperature

        modified_state = fault_injector.inject_fault(
            temperature_fault_scenario, mock_simulation_state
        )

        # Fault should be recorded
        assert temperature_fault_scenario.fault_id in fault_injector.active_faults

        # Temperature should be modified
        assert modified_state.process_conditions.temperature != original_temp
        assert (
            modified_state.process_conditions.temperature < original_temp
        )  # Temperature drop

    def test_inject_pressure_fault(self, fault_injector, mock_simulation_state):
        """Test pressure fault injection."""
        pressure_fault = FaultScenario(
            fault_id="pressure_spike_test",
            fault_type=FaultType.PRESSURE_ANOMALY,
            severity=FaultSeverity.MAJOR,
            description="Pressure spike test",
            affected_parameters=["pressure"],
            fault_magnitude=0.6,
            onset_time=0.0,
        )

        original_pressure = mock_simulation_state.process_conditions.pressure

        modified_state = fault_injector.inject_fault(
            pressure_fault, mock_simulation_state
        )

        # Pressure should increase for spike
        assert modified_state.process_conditions.pressure > original_pressure

    def test_inject_contamination_fault(self, fault_injector, mock_simulation_state):
        """Test contamination fault injection."""
        contamination_fault = FaultScenario(
            fault_id="contamination_test",
            fault_type=FaultType.CONTAMINATION,
            severity=FaultSeverity.MAJOR,
            description="Contamination test",
            affected_parameters=["composition"],
            fault_magnitude=0.4,
            onset_time=0.0,
        )

        modified_state = fault_injector.inject_fault(
            contamination_fault, mock_simulation_state
        )

        # Contaminant should be present
        assert "contaminant" in modified_state.process_conditions.material_composition
        assert modified_state.process_conditions.material_composition["contaminant"] > 0

    def test_fault_progression(
        self, fault_injector, mock_simulation_state, temperature_fault_scenario
    ):
        """Test fault progression over time."""
        # Inject fault
        fault_injector.inject_fault(temperature_fault_scenario, mock_simulation_state)

        # Check initial magnitude
        magnitude_0 = fault_injector.update_fault_progression(
            temperature_fault_scenario.fault_id, 0.0
        )
        assert magnitude_0 == 0.0  # At onset time

        # Check magnitude after some time
        magnitude_10 = fault_injector.update_fault_progression(
            temperature_fault_scenario.fault_id, 10.0
        )
        assert magnitude_10 > 0.0  # Should have progressed

    def test_clear_fault(
        self, fault_injector, mock_simulation_state, temperature_fault_scenario
    ):
        """Test fault clearing."""
        # Inject fault
        fault_injector.inject_fault(temperature_fault_scenario, mock_simulation_state)
        assert temperature_fault_scenario.fault_id in fault_injector.active_faults

        # Clear fault
        fault_injector.clear_fault(temperature_fault_scenario.fault_id)
        assert temperature_fault_scenario.fault_id not in fault_injector.active_faults
        assert len(fault_injector.fault_history) == 1

    def test_injection_callback(
        self, fault_injector, mock_simulation_state, temperature_fault_scenario
    ):
        """Test fault injection callbacks."""
        callback_calls = []

        def test_callback(scenario, state):
            callback_calls.append((scenario, state))

        fault_injector.add_injection_callback(test_callback)
        fault_injector.inject_fault(temperature_fault_scenario, mock_simulation_state)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == temperature_fault_scenario


class TestFaultDetector:
    """Test FaultDetector class."""

    @pytest.fixture
    def fault_detector(self):
        """Create FaultDetector instance."""
        return FaultDetector()

    @pytest.fixture
    def baseline_state(self):
        """Create baseline simulation state."""
        process_conditions = ProcessConditions(
            temperature=300.0,
            pressure=5e5,
            material_composition={"polymer": 0.8, "additive": 0.2},
        )

        microstructure = MicrostructureState(
            grain_size=5.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 1.0},
            defect_density=0.001,
            crystallinity=0.8,
        )

        return SimulationState(
            time=0.0,
            process_conditions=process_conditions,
            material_properties={},
            microstructure=microstructure,
        )

    @pytest.fixture
    def faulted_state(self, baseline_state):
        """Create faulted simulation state."""
        process_conditions = ProcessConditions(
            temperature=270.0,  # 30K drop
            pressure=baseline_state.process_conditions.pressure,
            material_composition=baseline_state.process_conditions.material_composition.copy(),
        )

        return SimulationState(
            time=baseline_state.time,
            process_conditions=process_conditions,
            material_properties=baseline_state.material_properties,
            microstructure=baseline_state.microstructure,
        )

    def test_initialization(self, fault_detector):
        """Test fault detector initialization."""
        assert fault_detector.baseline_state is None
        assert len(fault_detector.detection_history) == 0
        assert fault_detector.detection_threshold == 0.1

    def test_set_baseline(self, fault_detector, baseline_state):
        """Test setting baseline state."""
        fault_detector.set_baseline(baseline_state)
        assert fault_detector.baseline_state == baseline_state

    def test_detect_temperature_fault(
        self, fault_detector, baseline_state, faulted_state
    ):
        """Test temperature fault detection."""
        fault_detector.set_baseline(baseline_state)

        result = fault_detector.detect_faults(faulted_state)

        assert result.fault_detected == True
        assert len(result.fault_scenarios) > 0

        # Should detect temperature deviation
        temp_faults = [
            s
            for s in result.fault_scenarios
            if s.fault_type == FaultType.TEMPERATURE_DEVIATION
        ]
        assert len(temp_faults) > 0
        assert temp_faults[0].severity in [FaultSeverity.MAJOR, FaultSeverity.MODERATE]

    def test_detect_pressure_fault(self, fault_detector, baseline_state):
        """Test pressure fault detection."""
        fault_detector.set_baseline(baseline_state)

        # Create state with pressure fault
        process_conditions = ProcessConditions(
            temperature=baseline_state.process_conditions.temperature,
            pressure=baseline_state.process_conditions.pressure * 0.5,  # 50% drop
            material_composition=baseline_state.process_conditions.material_composition.copy(),
        )

        pressure_faulted_state = SimulationState(
            time=baseline_state.time,
            process_conditions=process_conditions,
            material_properties=baseline_state.material_properties,
            microstructure=baseline_state.microstructure,
        )

        result = fault_detector.detect_faults(pressure_faulted_state)

        assert result.fault_detected == True
        pressure_faults = [
            s
            for s in result.fault_scenarios
            if s.fault_type == FaultType.PRESSURE_ANOMALY
        ]
        assert len(pressure_faults) > 0

    def test_detect_contamination_fault(self, fault_detector, baseline_state):
        """Test contamination fault detection."""
        fault_detector.set_baseline(baseline_state)

        # Create state with contamination
        contaminated_composition = (
            baseline_state.process_conditions.material_composition.copy()
        )
        contaminated_composition["contaminant"] = 0.03  # 3% contamination

        process_conditions = ProcessConditions(
            temperature=baseline_state.process_conditions.temperature,
            pressure=baseline_state.process_conditions.pressure,
            material_composition=contaminated_composition,
        )

        contaminated_state = SimulationState(
            time=baseline_state.time,
            process_conditions=process_conditions,
            material_properties=baseline_state.material_properties,
            microstructure=baseline_state.microstructure,
        )

        result = fault_detector.detect_faults(contaminated_state)

        assert result.fault_detected == True
        contamination_faults = [
            s for s in result.fault_scenarios if s.fault_type == FaultType.CONTAMINATION
        ]
        assert len(contamination_faults) > 0

    def test_detect_sensor_faults(self, fault_detector, baseline_state):
        """Test sensor fault detection."""
        fault_detector.set_baseline(baseline_state)

        # Create sensor data with poor quality
        sensor_data = {
            "temp_sensor": SensorReading(
                "temp_sensor", time.time(), 300.0, "K", DataQuality.POOR
            ),
            "pressure_sensor": SensorReading(
                "pressure_sensor", time.time(), 5e5, "Pa", DataQuality.INVALID
            ),
        }

        result = fault_detector.detect_faults(baseline_state, sensor_data)

        assert result.fault_detected == True
        sensor_faults = [
            s
            for s in result.fault_scenarios
            if s.fault_type == FaultType.SENSOR_FAILURE
        ]
        assert len(sensor_faults) > 0

    def test_no_fault_detection(self, fault_detector, baseline_state):
        """Test no fault detection for normal state."""
        fault_detector.set_baseline(baseline_state)

        # Test with same state (no deviation)
        result = fault_detector.detect_faults(baseline_state)

        assert result.fault_detected == False
        assert len(result.fault_scenarios) == 0


class TestDecisionSupportSystem:
    """Test DecisionSupportSystem class."""

    @pytest.fixture
    def decision_support(self):
        """Create DecisionSupportSystem instance."""
        return DecisionSupportSystem()

    @pytest.fixture
    def mock_simulation_state(self):
        """Create mock simulation state."""
        process_conditions = ProcessConditions(
            temperature=270.0,  # Faulted temperature
            pressure=5e5,
            material_composition={"polymer": 0.8, "additive": 0.2},
        )

        microstructure = MicrostructureState(
            grain_size=5.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 1.0},
            defect_density=0.001,
            crystallinity=0.8,
        )

        return SimulationState(
            time=0.0,
            process_conditions=process_conditions,
            material_properties={},
            microstructure=microstructure,
        )

    @pytest.fixture
    def temperature_fault_analysis(self):
        """Create temperature fault analysis."""
        temp_scenario = FaultScenario(
            fault_id="temp_fault",
            fault_type=FaultType.TEMPERATURE_DEVIATION,
            severity=FaultSeverity.MAJOR,
            description="Temperature drop detected",
            affected_parameters=["temperature"],
            fault_magnitude=0.7,
            onset_time=0.0,
        )

        return FaultAnalysisResult(
            fault_detected=True,
            fault_scenarios=[temp_scenario],
            severity_assessment=FaultSeverity.MAJOR,
            confidence=0.8,
            affected_systems=["thermal_control"],
            estimated_impact={"quality": 0.6, "efficiency": 0.4},
            time_to_criticality=300.0,
            recommended_actions=[],
            alternative_routes=[],
        )

    def test_initialization(self, decision_support):
        """Test decision support system initialization."""
        assert len(decision_support.action_templates) > 0
        assert decision_support.effectiveness_model is not None
        assert len(decision_support.action_history) == 0

    def test_generate_temperature_actions(
        self, decision_support, temperature_fault_analysis, mock_simulation_state
    ):
        """Test generation of temperature corrective actions."""
        actions = decision_support.generate_corrective_actions(
            temperature_fault_analysis, mock_simulation_state
        )

        assert len(actions) > 0

        # Should have heating-related actions
        heating_actions = [
            a
            for a in actions
            if "heat" in a.description.lower() or "temperature" in a.target_parameter
        ]
        assert len(heating_actions) > 0

        # Actions should be prioritized
        assert all(action.priority >= 1 for action in actions)
        assert all(0 <= action.estimated_effectiveness <= 1 for action in actions)

    def test_generate_pressure_actions(self, decision_support, mock_simulation_state):
        """Test generation of pressure corrective actions."""
        pressure_scenario = FaultScenario(
            fault_id="pressure_fault",
            fault_type=FaultType.PRESSURE_ANOMALY,
            severity=FaultSeverity.MAJOR,
            description="Pressure spike detected",
            affected_parameters=["pressure"],
            fault_magnitude=0.8,
            onset_time=0.0,
        )

        fault_analysis = FaultAnalysisResult(
            fault_detected=True,
            fault_scenarios=[pressure_scenario],
            severity_assessment=FaultSeverity.MAJOR,
            confidence=0.9,
            affected_systems=["pressure_control"],
            estimated_impact={"safety": 0.8},
            time_to_criticality=60.0,
            recommended_actions=[],
            alternative_routes=[],
        )

        actions = decision_support.generate_corrective_actions(
            fault_analysis, mock_simulation_state
        )

        assert len(actions) > 0

        # Should have pressure-related actions
        pressure_actions = [
            a
            for a in actions
            if "pressure" in a.description.lower() or "valve" in a.description.lower()
        ]
        assert len(pressure_actions) > 0

    def test_generate_contamination_actions(
        self, decision_support, mock_simulation_state
    ):
        """Test generation of contamination corrective actions."""
        contamination_scenario = FaultScenario(
            fault_id="contamination_fault",
            fault_type=FaultType.CONTAMINATION,
            severity=FaultSeverity.MAJOR,
            description="Contamination detected",
            affected_parameters=["composition"],
            fault_magnitude=0.6,
            onset_time=0.0,
        )

        fault_analysis = FaultAnalysisResult(
            fault_detected=True,
            fault_scenarios=[contamination_scenario],
            severity_assessment=FaultSeverity.MAJOR,
            confidence=0.85,
            affected_systems=["material_handling"],
            estimated_impact={"quality": 0.9},
            time_to_criticality=180.0,
            recommended_actions=[],
            alternative_routes=[],
        )

        actions = decision_support.generate_corrective_actions(
            fault_analysis, mock_simulation_state
        )

        assert len(actions) > 0

        # Should have quality hold and purge actions
        hold_actions = [
            a for a in actions if a.action_type == CorrectiveActionType.QUALITY_HOLD
        ]
        purge_actions = [a for a in actions if "purge" in a.description.lower()]

        assert len(hold_actions) > 0 or len(purge_actions) > 0

    def test_action_ranking(
        self, decision_support, temperature_fault_analysis, mock_simulation_state
    ):
        """Test action ranking by effectiveness."""
        actions = decision_support.generate_corrective_actions(
            temperature_fault_analysis, mock_simulation_state
        )

        if len(actions) > 1:
            # Actions should be sorted by effectiveness (descending)
            for i in range(len(actions) - 1):
                # Higher effectiveness or higher priority should come first
                assert (
                    actions[i].estimated_effectiveness
                    >= actions[i + 1].estimated_effectiveness
                    or actions[i].priority <= actions[i + 1].priority
                )

    def test_evaluate_action_effectiveness(
        self, decision_support, mock_simulation_state
    ):
        """Test evaluation of action effectiveness."""
        action = CorrectiveAction(
            action_id="test_action",
            action_type=CorrectiveActionType.PARAMETER_ADJUSTMENT,
            target_parameter="temperature",
            target_value=300.0,
            current_value=270.0,
            priority=1,
            estimated_effectiveness=0.8,
            implementation_time=60.0,
            cost_estimate=100.0,
            risk_level=0.2,
            description="Test action",
            success_criteria={"temperature": 300.0},
        )

        # Create post-action state
        post_conditions = ProcessConditions(
            temperature=298.0,  # Close to target
            pressure=mock_simulation_state.process_conditions.pressure,
            material_composition=mock_simulation_state.process_conditions.material_composition,
        )

        post_state = SimulationState(
            time=mock_simulation_state.time,
            process_conditions=post_conditions,
            material_properties=mock_simulation_state.material_properties,
            microstructure=mock_simulation_state.microstructure,
        )

        effectiveness = decision_support.evaluate_action_effectiveness(
            action, mock_simulation_state, post_state
        )

        assert 0.0 <= effectiveness <= 1.0
        assert len(decision_support.action_history) == 1


class TestAlternativeRouteOptimizer:
    """Test AlternativeRouteOptimizer class."""

    @pytest.fixture
    def route_optimizer(self):
        """Create AlternativeRouteOptimizer instance."""
        return AlternativeRouteOptimizer()

    @pytest.fixture
    def mock_simulation_state(self):
        """Create mock simulation state."""
        process_conditions = ProcessConditions(
            temperature=300.0,
            pressure=5e5,
            material_composition={"polymer": 0.8, "additive": 0.2},
        )

        microstructure = MicrostructureState(
            grain_size=5.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 1.0},
            defect_density=0.001,
            crystallinity=0.8,
        )

        return SimulationState(
            time=0.0,
            process_conditions=process_conditions,
            material_properties={},
            microstructure=microstructure,
        )

    @pytest.fixture
    def target_output(self):
        """Create target process output."""
        return ProcessOutput(
            quality_metrics={"purity": 0.95, "yield": 0.9},
            quantity=100.0,
            processing_time=3600.0,
        )

    def test_initialization(self, route_optimizer):
        """Test route optimizer initialization."""
        assert len(route_optimizer.route_database) > 0
        assert len(route_optimizer.route_history) == 0

    def test_find_thermal_alternatives(
        self, route_optimizer, mock_simulation_state, target_output
    ):
        """Test finding thermal alternative routes."""
        fault_analysis = FaultAnalysisResult(
            fault_detected=True,
            fault_scenarios=[],
            severity_assessment=FaultSeverity.MAJOR,
            confidence=0.8,
            affected_systems=["thermal_control"],
            estimated_impact={"quality": 0.6},
            time_to_criticality=300.0,
            recommended_actions=[],
            alternative_routes=[],
        )

        routes = route_optimizer.find_alternative_routes(
            fault_analysis, mock_simulation_state, target_output
        )

        assert len(routes) > 0

        # Should have thermal alternatives
        thermal_routes = [
            r for r in routes if "temperature" in str(r.get("modifications", {}))
        ]
        assert len(thermal_routes) > 0

        # Routes should have required fields
        for route in routes:
            assert "route_id" in route
            assert "name" in route
            assert "estimated_yield" in route
            assert "estimated_quality" in route
            assert "feasibility" in route

    def test_find_pressure_alternatives(
        self, route_optimizer, mock_simulation_state, target_output
    ):
        """Test finding pressure alternative routes."""
        fault_analysis = FaultAnalysisResult(
            fault_detected=True,
            fault_scenarios=[],
            severity_assessment=FaultSeverity.CRITICAL,
            confidence=0.9,
            affected_systems=["pressure_control"],
            estimated_impact={"safety": 0.8},
            time_to_criticality=60.0,
            recommended_actions=[],
            alternative_routes=[],
        )

        routes = route_optimizer.find_alternative_routes(
            fault_analysis, mock_simulation_state, target_output
        )

        assert len(routes) > 0

        # Should include batch processing for critical faults
        batch_routes = [r for r in routes if "batch" in r.get("name", "").lower()]
        assert len(batch_routes) > 0

    def test_find_material_alternatives(
        self, route_optimizer, mock_simulation_state, target_output
    ):
        """Test finding material handling alternatives."""
        fault_analysis = FaultAnalysisResult(
            fault_detected=True,
            fault_scenarios=[],
            severity_assessment=FaultSeverity.MODERATE,
            confidence=0.7,
            affected_systems=["material_handling"],
            estimated_impact={"quality": 0.5},
            time_to_criticality=600.0,
            recommended_actions=[],
            alternative_routes=[],
        )

        routes = route_optimizer.find_alternative_routes(
            fault_analysis, mock_simulation_state, target_output
        )

        assert len(routes) > 0

        # Should have material alternatives
        material_routes = [
            r
            for r in routes
            if "feedstock" in r.get("name", "").lower()
            or "material" in r.get("name", "").lower()
        ]
        assert len(material_routes) > 0

    def test_route_ranking(self, route_optimizer, mock_simulation_state, target_output):
        """Test route ranking by score."""
        fault_analysis = FaultAnalysisResult(
            fault_detected=True,
            fault_scenarios=[],
            severity_assessment=FaultSeverity.MAJOR,
            confidence=0.8,
            affected_systems=["thermal_control", "pressure_control"],
            estimated_impact={"quality": 0.6, "efficiency": 0.4},
            time_to_criticality=300.0,
            recommended_actions=[],
            alternative_routes=[],
        )

        routes = route_optimizer.find_alternative_routes(
            fault_analysis, mock_simulation_state, target_output
        )

        if len(routes) > 1:
            # Routes should be ranked (higher score first)
            for i in range(len(routes) - 1):
                route1 = routes[i]
                route2 = routes[i + 1]

                # Calculate simple scores for comparison
                score1 = (
                    route1["estimated_yield"]
                    * route1["estimated_quality"]
                    * route1["feasibility"]
                )
                score2 = (
                    route2["estimated_yield"]
                    * route2["estimated_quality"]
                    * route2["feasibility"]
                )

                # Allow for some tolerance in ranking due to other factors
                assert score1 >= score2 * 0.8


class TestFaultScenarioSimulator:
    """Test FaultScenarioSimulator class."""

    @pytest.fixture
    def mock_simulation_core(self):
        """Create mock simulation core."""
        mock_core = Mock(spec=RealTimeSimulationCore)

        process_conditions = ProcessConditions(
            temperature=300.0,
            pressure=5e5,
            material_composition={"polymer": 0.8, "additive": 0.2},
        )

        microstructure = MicrostructureState(
            grain_size=5.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 1.0},
            defect_density=0.001,
            crystallinity=0.8,
        )

        mock_core.current_state = SimulationState(
            time=0.0,
            process_conditions=process_conditions,
            material_properties={},
            microstructure=microstructure,
        )

        return mock_core

    @pytest.fixture
    def fault_simulator(self, mock_simulation_core):
        """Create FaultScenarioSimulator instance."""
        return FaultScenarioSimulator(mock_simulation_core)

    @pytest.fixture
    def test_scenario(self):
        """Create test fault scenario."""
        return FaultScenario(
            fault_id="test_scenario",
            fault_type=FaultType.TEMPERATURE_DEVIATION,
            severity=FaultSeverity.MODERATE,
            description="Test temperature fault",
            affected_parameters=["temperature"],
            fault_magnitude=0.4,
            onset_time=0.1,  # Short onset for testing
        )

    def test_initialization(self, fault_simulator, mock_simulation_core):
        """Test fault simulator initialization."""
        assert fault_simulator.simulation_core == mock_simulation_core
        assert fault_simulator.fault_injector is not None
        assert fault_simulator.fault_detector is not None
        assert fault_simulator.decision_support is not None
        assert fault_simulator.route_optimizer is not None
        assert len(fault_simulator.active_scenarios) == 0

    @pytest.mark.asyncio
    async def test_simulate_fault_scenario(self, fault_simulator, test_scenario):
        """Test complete fault scenario simulation."""
        result = await fault_simulator.simulate_fault_scenario(test_scenario)

        assert "scenario" in result
        assert "fault_analysis" in result
        assert "corrective_actions" in result
        assert "alternative_routes" in result
        assert "performance" in result

        # Check scenario data
        assert result["scenario"]["fault_id"] == test_scenario.fault_id

        # Check performance metrics
        assert "detection_time" in result["performance"]
        assert "recovery_time" in result["performance"]
        assert "total_simulation_time" in result["performance"]

        # Should have updated metrics
        assert fault_simulator.metrics["scenarios_simulated"] > 0

    @pytest.mark.asyncio
    async def test_run_fault_scenario_suite(self, fault_simulator):
        """Test running multiple fault scenarios."""
        scenarios = [
            FaultScenario(
                fault_id="scenario_1",
                fault_type=FaultType.TEMPERATURE_DEVIATION,
                severity=FaultSeverity.MINOR,
                description="Test scenario 1",
                affected_parameters=["temperature"],
                fault_magnitude=0.2,
                onset_time=0.1,
            ),
            FaultScenario(
                fault_id="scenario_2",
                fault_type=FaultType.PRESSURE_ANOMALY,
                severity=FaultSeverity.MODERATE,
                description="Test scenario 2",
                affected_parameters=["pressure"],
                fault_magnitude=0.3,
                onset_time=0.1,
            ),
        ]

        results = await fault_simulator.run_fault_scenario_suite(scenarios)

        assert len(results) == 2
        assert fault_simulator.metrics["scenarios_simulated"] == 2

        # Each result should have required fields
        for result in results:
            assert "scenario" in result
            assert "timestamp" in result

    def test_create_standard_fault_scenarios(self, fault_simulator):
        """Test creation of standard fault scenarios."""
        scenarios = fault_simulator.create_standard_fault_scenarios()

        assert len(scenarios) > 0

        # Should have different fault types
        fault_types = {scenario.fault_type for scenario in scenarios}
        assert FaultType.TEMPERATURE_DEVIATION in fault_types
        assert FaultType.PRESSURE_ANOMALY in fault_types
        assert FaultType.CONTAMINATION in fault_types

        # Should have different severities
        severities = {scenario.severity for scenario in scenarios}
        assert len(severities) > 1

    def test_get_simulation_metrics(self, fault_simulator):
        """Test simulation metrics retrieval."""
        metrics = fault_simulator.get_simulation_metrics()

        assert "scenarios_simulated" in metrics
        assert "faults_detected" in metrics
        assert "successful_recoveries" in metrics
        assert "fault_detection_rate" in metrics
        assert "recovery_success_rate" in metrics
        assert "active_scenarios" in metrics

    def test_export_simulation_results(self, fault_simulator, tmp_path):
        """Test exporting simulation results."""
        # Add some mock data
        fault_simulator.metrics["scenarios_simulated"] = 5
        fault_simulator.simulation_history.append({"test": "data"})

        output_file = tmp_path / "test_results.json"
        fault_simulator.export_simulation_results(str(output_file))

        assert output_file.exists()

        # Check file contents
        import json

        with open(output_file) as f:
            data = json.load(f)

        assert "metrics" in data
        assert "simulation_history" in data
        assert "export_timestamp" in data


class TestConvenienceFunctions:
    """Test convenience functions for creating fault scenarios."""

    def test_create_temperature_fault(self):
        """Test temperature fault creation function."""
        # Test temperature drop
        drop_fault = create_temperature_fault(
            magnitude=0.6, severity=FaultSeverity.MAJOR, is_drop=True
        )

        assert drop_fault.fault_type == FaultType.TEMPERATURE_DEVIATION
        assert drop_fault.severity == FaultSeverity.MAJOR
        assert drop_fault.fault_magnitude == 0.6
        assert "drop" in drop_fault.description
        assert drop_fault.root_cause == FaultCause.EQUIPMENT_WEAR

        # Test temperature spike
        spike_fault = create_temperature_fault(
            magnitude=0.8, severity=FaultSeverity.CRITICAL, is_drop=False
        )

        assert "spike" in spike_fault.description
        assert spike_fault.root_cause == FaultCause.EQUIPMENT_MALFUNCTION

    def test_create_contamination_fault(self):
        """Test contamination fault creation function."""
        contamination_fault = create_contamination_fault(
            contaminant_level=0.03, severity=FaultSeverity.MAJOR
        )

        assert contamination_fault.fault_type == FaultType.CONTAMINATION
        assert contamination_fault.severity == FaultSeverity.MAJOR
        assert contamination_fault.fault_magnitude == 0.6  # 0.03 * 20
        assert "contamination" in contamination_fault.description.lower()
        assert contamination_fault.root_cause == FaultCause.MATERIAL_VARIATION

    def test_create_pressure_fault(self):
        """Test pressure fault creation function."""
        # Test pressure drop
        drop_fault = create_pressure_fault(
            magnitude=0.4, severity=FaultSeverity.MODERATE, is_drop=True
        )

        assert drop_fault.fault_type == FaultType.PRESSURE_ANOMALY
        assert drop_fault.severity == FaultSeverity.MODERATE
        assert drop_fault.fault_magnitude == 0.4
        assert "drop" in drop_fault.description
        assert drop_fault.root_cause == FaultCause.EQUIPMENT_WEAR

        # Test pressure spike
        spike_fault = create_pressure_fault(
            magnitude=0.7, severity=FaultSeverity.MAJOR, is_drop=False
        )

        assert "spike" in spike_fault.description
        assert spike_fault.root_cause == FaultCause.BLOCKAGE


if __name__ == "__main__":
    pytest.main([__file__])

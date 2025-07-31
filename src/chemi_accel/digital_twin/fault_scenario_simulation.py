"""
Fault Scenario Simulation and Corrective Actions Module

This module provides comprehensive fault simulation capabilities for digital twins,
enabling the analysis of various failure scenarios and implementation of automated
corrective actions to maintain process stability and product quality.

Key Features:
- Comprehensive fault scenario modeling and injection
- Real-time fault detection and severity assessment
- Automated corrective action generation and execution
- Alternative process route optimization
- Decision support system for fault mitigation
- Process recovery and stability restoration
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ChemiAccel imports
from ..core.exceptions import ChemiAccelError
from .manufacturing_integration import ProcessConditions, ProcessOutput, ProcessType
from .predictive_control import ControlAction, ModelPredictiveController
from .realtime_process_simulation import (
    MaterialProperty,
    MicrostructureState,
    PropertyType,
    RealTimeSimulationCore,
    SimulationState,
)
from .sensor_data_integration import DataQuality, SensorReading


class FaultType(Enum):
    """Types of process faults that can occur."""

    TEMPERATURE_DEVIATION = "temperature_deviation"
    PRESSURE_ANOMALY = "pressure_anomaly"
    COMPOSITION_DRIFT = "composition_drift"
    FLOW_RATE_DISRUPTION = "flow_rate_disruption"
    CONTAMINATION = "contamination"
    EQUIPMENT_MALFUNCTION = "equipment_malfunction"
    SENSOR_FAILURE = "sensor_failure"
    POWER_FLUCTUATION = "power_fluctuation"
    COOLING_SYSTEM_FAILURE = "cooling_system_failure"
    MIXING_INEFFICIENCY = "mixing_inefficiency"
    CATALYST_DEACTIVATION = "catalyst_deactivation"
    BLOCKAGE = "blockage"


class FaultSeverity(Enum):
    """Severity levels for process faults."""

    CRITICAL = "critical"  # Immediate shutdown required
    MAJOR = "major"  # Significant impact, urgent action needed
    MODERATE = "moderate"  # Process deviation, corrective action recommended
    MINOR = "minor"  # Small deviation, monitoring required
    WARNING = "warning"  # Early indication, preventive action possible


class FaultCause(Enum):
    """Root causes of process faults."""

    EQUIPMENT_WEAR = "equipment_wear"
    HUMAN_ERROR = "human_error"
    EXTERNAL_CONDITIONS = "external_conditions"
    MATERIAL_VARIATION = "material_variation"
    CONTROL_SYSTEM_ERROR = "control_system_error"
    MAINTENANCE_REQUIRED = "maintenance_required"
    DESIGN_LIMITATION = "design_limitation"
    UNKNOWN = "unknown"


class CorrectiveActionType(Enum):
    """Types of corrective actions."""

    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    PROCESS_ROUTE_CHANGE = "process_route_change"
    EQUIPMENT_ISOLATION = "equipment_isolation"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    MATERIAL_REPLACEMENT = "material_replacement"
    CLEANING_CYCLE = "cleaning_cycle"
    MAINTENANCE_INTERVENTION = "maintenance_intervention"
    OPERATOR_NOTIFICATION = "operator_notification"
    QUALITY_HOLD = "quality_hold"
    RECOVERY_SEQUENCE = "recovery_sequence"


@dataclass
class FaultScenario:
    """Definition of a fault scenario for simulation."""

    fault_id: str
    fault_type: FaultType
    severity: FaultSeverity
    description: str
    affected_parameters: List[str]
    fault_magnitude: float  # Relative magnitude (0-1)
    onset_time: float  # Time when fault begins (seconds)
    duration: Optional[float] = None  # Duration of fault (None = permanent)
    propagation_rate: float = 1.0  # How quickly fault develops
    recovery_time: Optional[float] = None  # Time to natural recovery
    root_cause: FaultCause = FaultCause.UNKNOWN
    preconditions: Dict[str, Any] = field(default_factory=dict)
    environmental_factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_parameters": self.affected_parameters,
            "fault_magnitude": self.fault_magnitude,
            "onset_time": self.onset_time,
            "duration": self.duration,
            "propagation_rate": self.propagation_rate,
            "recovery_time": self.recovery_time,
            "root_cause": self.root_cause.value,
            "preconditions": self.preconditions,
            "environmental_factors": self.environmental_factors,
        }


@dataclass
class CorrectiveAction:
    """Corrective action to address a fault."""

    action_id: str
    action_type: CorrectiveActionType
    target_parameter: str
    target_value: float
    current_value: float
    priority: int  # 1 = highest priority
    estimated_effectiveness: float  # 0-1
    implementation_time: float  # seconds to implement
    cost_estimate: float
    risk_level: float  # 0-1, higher = riskier
    description: str
    prerequisites: List[str] = field(default_factory=list)
    side_effects: Dict[str, float] = field(default_factory=dict)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    rollback_procedure: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class FaultAnalysisResult:
    """Result of fault analysis."""

    fault_detected: bool
    fault_scenarios: List[FaultScenario]
    severity_assessment: FaultSeverity
    confidence: float
    affected_systems: List[str]
    estimated_impact: Dict[str, float]  # Impact on different KPIs
    time_to_criticality: Optional[float]  # Time until critical if uncorrected
    recommended_actions: List[CorrectiveAction]
    alternative_routes: List[Dict[str, Any]]
    analysis_timestamp: float = field(default_factory=time.time)


class FaultInjector:
    """Injects faults into simulation for testing."""

    def __init__(self):
        self.active_faults = {}
        self.fault_history = deque(maxlen=1000)
        self.injection_callbacks = []

    def inject_fault(
        self, scenario: FaultScenario, simulation_state: SimulationState
    ) -> SimulationState:
        """Inject a fault into the simulation state."""
        try:
            # Record fault injection
            self.active_faults[scenario.fault_id] = {
                "scenario": scenario,
                "injection_time": time.time(),
                "current_magnitude": 0.0,
            }

            # Apply fault effects based on type
            modified_state = self._apply_fault_effects(scenario, simulation_state)

            # Log fault injection
            logging.warning(
                f"Injected fault: {scenario.fault_type.value} - {scenario.description}"
            )

            # Notify callbacks
            for callback in self.injection_callbacks:
                try:
                    callback(scenario, modified_state)
                except Exception as e:
                    logging.error(f"Error in fault injection callback: {e}")

            return modified_state

        except Exception as e:
            logging.error(f"Failed to inject fault {scenario.fault_id}: {e}")
            return simulation_state

    def _apply_fault_effects(
        self, scenario: FaultScenario, state: SimulationState
    ) -> SimulationState:
        """Apply fault effects to simulation state."""
        # Create a copy of the state to modify
        new_conditions = ProcessConditions(
            temperature=state.process_conditions.temperature,
            pressure=state.process_conditions.pressure,
            material_composition=state.process_conditions.material_composition.copy(),
        )

        # Apply fault effects based on type
        if scenario.fault_type == FaultType.TEMPERATURE_DEVIATION:
            if "temperature_drop" in scenario.description.lower():
                delta = -scenario.fault_magnitude * 50.0  # Up to 50K drop
            else:
                delta = scenario.fault_magnitude * 100.0  # Up to 100K increase
            new_conditions.temperature = max(273.15, new_conditions.temperature + delta)

        elif scenario.fault_type == FaultType.PRESSURE_ANOMALY:
            if "pressure_drop" in scenario.description.lower():
                delta = -scenario.fault_magnitude * 0.5 * new_conditions.pressure
            else:
                delta = scenario.fault_magnitude * 0.3 * new_conditions.pressure
            new_conditions.pressure = max(1e4, new_conditions.pressure + delta)

        elif scenario.fault_type == FaultType.COMPOSITION_DRIFT:
            # Modify composition based on affected parameters
            for param in scenario.affected_parameters:
                if param in new_conditions.material_composition:
                    current_value = new_conditions.material_composition[param]
                    delta = (
                        scenario.fault_magnitude * 0.2 * current_value
                    )  # Up to 20% change
                    new_conditions.material_composition[param] = max(
                        0.0, min(1.0, current_value + delta)
                    )

        elif scenario.fault_type == FaultType.CONTAMINATION:
            # Add contaminant to composition
            contaminant_level = (
                scenario.fault_magnitude * 0.05
            )  # Up to 5% contamination
            new_conditions.material_composition["contaminant"] = contaminant_level

            # Normalize other components
            total_others = sum(
                v
                for k, v in new_conditions.material_composition.items()
                if k != "contaminant"
            )
            if total_others > 0:
                scale_factor = (1.0 - contaminant_level) / total_others
                for k, v in new_conditions.material_composition.items():
                    if k != "contaminant":
                        new_conditions.material_composition[k] = v * scale_factor

        # Create modified simulation state
        modified_state = SimulationState(
            time=state.time,
            process_conditions=new_conditions,
            material_properties=state.material_properties.copy(),
            microstructure=state.microstructure,
        )

        return modified_state

    def update_fault_progression(self, fault_id: str, current_time: float) -> float:
        """Update fault progression over time."""
        if fault_id not in self.active_faults:
            return 0.0

        fault_info = self.active_faults[fault_id]
        scenario = fault_info["scenario"]
        injection_time = fault_info["injection_time"]

        elapsed = current_time - injection_time

        # Calculate current magnitude based on propagation rate
        if elapsed < scenario.onset_time:
            magnitude = 0.0
        else:
            progress_time = elapsed - scenario.onset_time
            magnitude = (
                min(1.0, progress_time * scenario.propagation_rate)
                * scenario.fault_magnitude
            )

        # Check for natural recovery
        if scenario.recovery_time and elapsed > scenario.recovery_time:
            recovery_progress = (elapsed - scenario.recovery_time) / max(
                60.0, scenario.recovery_time * 0.1
            )
            magnitude *= max(0.0, 1.0 - recovery_progress)

        fault_info["current_magnitude"] = magnitude
        return magnitude

    def clear_fault(self, fault_id: str):
        """Clear an active fault."""
        if fault_id in self.active_faults:
            self.fault_history.append(self.active_faults[fault_id])
            del self.active_faults[fault_id]
            logging.info(f"Cleared fault: {fault_id}")

    def get_active_faults(self) -> Dict[str, Dict]:
        """Get all currently active faults."""
        return self.active_faults.copy()

    def add_injection_callback(
        self, callback: Callable[[FaultScenario, SimulationState], None]
    ):
        """Add callback for fault injection events."""
        self.injection_callbacks.append(callback)


class FaultDetector:
    """Detects faults from simulation state and sensor data."""

    def __init__(self, detection_threshold: float = 0.1):
        self.detection_threshold = detection_threshold
        self.baseline_state = None
        self.detection_history = deque(maxlen=100)
        self.anomaly_thresholds = {
            "temperature": {"min_deviation": 5.0, "max_rate": 2.0},  # K/s
            "pressure": {"min_deviation": 0.05, "max_rate": 0.1},  # relative
            "composition": {"min_deviation": 0.02, "max_rate": 0.01},
        }

    def set_baseline(self, state: SimulationState):
        """Set baseline state for anomaly detection."""
        self.baseline_state = state
        logging.info("Set new baseline state for fault detection")

    def detect_faults(
        self,
        current_state: SimulationState,
        sensor_data: Optional[Dict[str, SensorReading]] = None,
    ) -> FaultAnalysisResult:
        """Detect faults in current state."""
        detected_scenarios = []
        severity = FaultSeverity.WARNING
        confidence = 0.0

        try:
            if self.baseline_state is None:
                self.set_baseline(current_state)
                return FaultAnalysisResult(
                    fault_detected=False,
                    fault_scenarios=[],
                    severity_assessment=FaultSeverity.WARNING,
                    confidence=0.0,
                    affected_systems=[],
                    estimated_impact={},
                    time_to_criticality=None,
                    recommended_actions=[],
                    alternative_routes=[],
                )

            # Check temperature deviations
            temp_scenario = self._check_temperature_faults(current_state)
            if temp_scenario:
                detected_scenarios.append(temp_scenario)

            # Check pressure anomalies
            pressure_scenario = self._check_pressure_faults(current_state)
            if pressure_scenario:
                detected_scenarios.append(pressure_scenario)

            # Check composition drifts
            composition_scenarios = self._check_composition_faults(current_state)
            detected_scenarios.extend(composition_scenarios)

            # Check sensor faults if sensor data available
            if sensor_data:
                sensor_scenarios = self._check_sensor_faults(sensor_data)
                detected_scenarios.extend(sensor_scenarios)

            # Assess overall severity
            if detected_scenarios:
                severity = max(scenario.severity for scenario in detected_scenarios)
                confidence = min(1.0, len(detected_scenarios) * 0.3)

            # Generate analysis result
            result = FaultAnalysisResult(
                fault_detected=len(detected_scenarios) > 0,
                fault_scenarios=detected_scenarios,
                severity_assessment=severity,
                confidence=confidence,
                affected_systems=self._identify_affected_systems(detected_scenarios),
                estimated_impact=self._estimate_impact(
                    detected_scenarios, current_state
                ),
                time_to_criticality=self._estimate_time_to_criticality(
                    detected_scenarios
                ),
                recommended_actions=[],  # Will be filled by decision support system
                alternative_routes=[],  # Will be filled by route optimizer
            )

            self.detection_history.append(result)
            return result

        except Exception as e:
            logging.error(f"Error in fault detection: {e}")
            return FaultAnalysisResult(
                fault_detected=False,
                fault_scenarios=[],
                severity_assessment=FaultSeverity.WARNING,
                confidence=0.0,
                affected_systems=[],
                estimated_impact={},
                time_to_criticality=None,
                recommended_actions=[],
                alternative_routes=[],
            )

    def _check_temperature_faults(
        self, state: SimulationState
    ) -> Optional[FaultScenario]:
        """Check for temperature-related faults."""
        current_temp = state.process_conditions.temperature
        baseline_temp = self.baseline_state.process_conditions.temperature

        deviation = abs(current_temp - baseline_temp)
        relative_deviation = deviation / baseline_temp

        if deviation > self.anomaly_thresholds["temperature"]["min_deviation"]:
            if current_temp < baseline_temp - 20.0:
                severity = (
                    FaultSeverity.MAJOR if deviation > 50.0 else FaultSeverity.MODERATE
                )
                return FaultScenario(
                    fault_id=f"temp_drop_{int(time.time())}",
                    fault_type=FaultType.TEMPERATURE_DEVIATION,
                    severity=severity,
                    description=f"Temperature drop detected: {current_temp:.1f}K (baseline: {baseline_temp:.1f}K)",
                    affected_parameters=["temperature"],
                    fault_magnitude=min(1.0, relative_deviation * 2.0),
                    onset_time=0.0,
                    root_cause=(
                        FaultCause.EQUIPMENT_WEAR
                        if deviation > 30.0
                        else FaultCause.EXTERNAL_CONDITIONS
                    ),
                )
            elif current_temp > baseline_temp + 30.0:
                severity = (
                    FaultSeverity.CRITICAL if deviation > 100.0 else FaultSeverity.MAJOR
                )
                return FaultScenario(
                    fault_id=f"temp_rise_{int(time.time())}",
                    fault_type=FaultType.TEMPERATURE_DEVIATION,
                    severity=severity,
                    description=f"Temperature spike detected: {current_temp:.1f}K (baseline: {baseline_temp:.1f}K)",
                    affected_parameters=["temperature"],
                    fault_magnitude=min(1.0, relative_deviation * 1.5),
                    onset_time=0.0,
                    root_cause=FaultCause.EQUIPMENT_MALFUNCTION,
                )

        return None

    def _check_pressure_faults(self, state: SimulationState) -> Optional[FaultScenario]:
        """Check for pressure-related faults."""
        current_pressure = state.process_conditions.pressure
        baseline_pressure = self.baseline_state.process_conditions.pressure

        relative_deviation = (
            abs(current_pressure - baseline_pressure) / baseline_pressure
        )

        if relative_deviation > self.anomaly_thresholds["pressure"]["min_deviation"]:
            if current_pressure < baseline_pressure * 0.8:
                severity = (
                    FaultSeverity.MAJOR
                    if relative_deviation > 0.3
                    else FaultSeverity.MODERATE
                )
                return FaultScenario(
                    fault_id=f"pressure_drop_{int(time.time())}",
                    fault_type=FaultType.PRESSURE_ANOMALY,
                    severity=severity,
                    description=f"Pressure drop detected: {current_pressure/1e5:.2f} bar (baseline: {baseline_pressure/1e5:.2f} bar)",
                    affected_parameters=["pressure"],
                    fault_magnitude=relative_deviation,
                    onset_time=0.0,
                    root_cause=FaultCause.EQUIPMENT_WEAR,
                )
            elif current_pressure > baseline_pressure * 1.2:
                severity = (
                    FaultSeverity.CRITICAL
                    if relative_deviation > 0.5
                    else FaultSeverity.MAJOR
                )
                return FaultScenario(
                    fault_id=f"pressure_spike_{int(time.time())}",
                    fault_type=FaultType.PRESSURE_ANOMALY,
                    severity=severity,
                    description=f"Pressure spike detected: {current_pressure/1e5:.2f} bar (baseline: {baseline_pressure/1e5:.2f} bar)",
                    affected_parameters=["pressure"],
                    fault_magnitude=relative_deviation,
                    onset_time=0.0,
                    root_cause=FaultCause.BLOCKAGE,
                )

        return None

    def _check_composition_faults(self, state: SimulationState) -> List[FaultScenario]:
        """Check for composition-related faults."""
        scenarios = []
        current_comp = state.process_conditions.material_composition
        baseline_comp = self.baseline_state.process_conditions.material_composition

        for component, current_value in current_comp.items():
            baseline_value = baseline_comp.get(component, 0.0)
            deviation = abs(current_value - baseline_value)

            if deviation > self.anomaly_thresholds["composition"]["min_deviation"]:
                if component == "contaminant" and current_value > 0.01:
                    severity = (
                        FaultSeverity.MAJOR
                        if current_value > 0.05
                        else FaultSeverity.MODERATE
                    )
                    scenarios.append(
                        FaultScenario(
                            fault_id=f"contamination_{component}_{int(time.time())}",
                            fault_type=FaultType.CONTAMINATION,
                            severity=severity,
                            description=f"Contamination detected: {component} at {current_value:.3f}",
                            affected_parameters=[component],
                            fault_magnitude=current_value * 20.0,  # Scale to 0-1
                            onset_time=0.0,
                            root_cause=FaultCause.MATERIAL_VARIATION,
                        )
                    )
                elif deviation > 0.05:
                    severity = (
                        FaultSeverity.MODERATE
                        if deviation > 0.1
                        else FaultSeverity.MINOR
                    )
                    scenarios.append(
                        FaultScenario(
                            fault_id=f"composition_drift_{component}_{int(time.time())}",
                            fault_type=FaultType.COMPOSITION_DRIFT,
                            severity=severity,
                            description=f"Composition drift in {component}: {current_value:.3f} (baseline: {baseline_value:.3f})",
                            affected_parameters=[component],
                            fault_magnitude=min(1.0, deviation * 10.0),
                            onset_time=0.0,
                            root_cause=FaultCause.CONTROL_SYSTEM_ERROR,
                        )
                    )

        return scenarios

    def _check_sensor_faults(
        self, sensor_data: Dict[str, SensorReading]
    ) -> List[FaultScenario]:
        """Check for sensor-related faults."""
        scenarios = []

        for sensor_id, reading in sensor_data.items():
            if reading.quality in [DataQuality.POOR, DataQuality.INVALID]:
                severity = (
                    FaultSeverity.MAJOR
                    if reading.quality == DataQuality.INVALID
                    else FaultSeverity.MODERATE
                )
                scenarios.append(
                    FaultScenario(
                        fault_id=f"sensor_fault_{sensor_id}_{int(time.time())}",
                        fault_type=FaultType.SENSOR_FAILURE,
                        severity=severity,
                        description=f"Sensor quality degradation: {sensor_id} quality = {reading.quality.value}",
                        affected_parameters=[sensor_id],
                        fault_magnitude=(
                            0.8 if reading.quality == DataQuality.INVALID else 0.4
                        ),
                        onset_time=0.0,
                        root_cause=FaultCause.EQUIPMENT_WEAR,
                    )
                )

        return scenarios

    def _identify_affected_systems(self, scenarios: List[FaultScenario]) -> List[str]:
        """Identify affected systems from fault scenarios."""
        systems = set()

        for scenario in scenarios:
            if scenario.fault_type in [
                FaultType.TEMPERATURE_DEVIATION,
                FaultType.COOLING_SYSTEM_FAILURE,
            ]:
                systems.add("thermal_control")
            elif scenario.fault_type in [
                FaultType.PRESSURE_ANOMALY,
                FaultType.BLOCKAGE,
            ]:
                systems.add("pressure_control")
            elif scenario.fault_type in [
                FaultType.COMPOSITION_DRIFT,
                FaultType.CONTAMINATION,
            ]:
                systems.add("material_handling")
            elif scenario.fault_type == FaultType.SENSOR_FAILURE:
                systems.add("instrumentation")
            elif scenario.fault_type == FaultType.EQUIPMENT_MALFUNCTION:
                systems.add("mechanical_systems")

        return list(systems)

    def _estimate_impact(
        self, scenarios: List[FaultScenario], state: SimulationState
    ) -> Dict[str, float]:
        """Estimate impact on key performance indicators."""
        impact = {"quality": 0.0, "efficiency": 0.0, "safety": 0.0, "cost": 0.0}

        for scenario in scenarios:
            # Quality impact
            if scenario.fault_type in [
                FaultType.CONTAMINATION,
                FaultType.COMPOSITION_DRIFT,
            ]:
                impact["quality"] += scenario.fault_magnitude * 0.7
            elif scenario.fault_type == FaultType.TEMPERATURE_DEVIATION:
                impact["quality"] += scenario.fault_magnitude * 0.5

            # Efficiency impact
            if scenario.fault_type in [
                FaultType.EQUIPMENT_MALFUNCTION,
                FaultType.BLOCKAGE,
            ]:
                impact["efficiency"] += scenario.fault_magnitude * 0.8
            elif scenario.fault_type == FaultType.PRESSURE_ANOMALY:
                impact["efficiency"] += scenario.fault_magnitude * 0.4

            # Safety impact
            if scenario.severity in [FaultSeverity.CRITICAL, FaultSeverity.MAJOR]:
                impact["safety"] += scenario.fault_magnitude * 0.9

            # Cost impact
            impact["cost"] += scenario.fault_magnitude * 0.3

        # Normalize impacts
        for key in impact:
            impact[key] = min(1.0, impact[key])

        return impact

    def _estimate_time_to_criticality(
        self, scenarios: List[FaultScenario]
    ) -> Optional[float]:
        """Estimate time until fault becomes critical."""
        if not scenarios:
            return None

        # Find scenario with highest severity
        max_severity = max(scenario.severity for scenario in scenarios)

        if max_severity == FaultSeverity.CRITICAL:
            return 0.0  # Already critical
        elif max_severity == FaultSeverity.MAJOR:
            return 300.0  # 5 minutes
        elif max_severity == FaultSeverity.MODERATE:
            return 1800.0  # 30 minutes
        else:
            return 3600.0  # 1 hour


class DecisionSupportSystem:
    """Generates and evaluates corrective actions for faults."""

    def __init__(self):
        self.action_templates = self._initialize_action_templates()
        self.effectiveness_model = self._create_effectiveness_model()
        self.action_history = deque(maxlen=500)

    def generate_corrective_actions(
        self, fault_analysis: FaultAnalysisResult, current_state: SimulationState
    ) -> List[CorrectiveAction]:
        """Generate corrective actions for detected faults."""
        actions = []

        try:
            for scenario in fault_analysis.fault_scenarios:
                scenario_actions = self._generate_actions_for_scenario(
                    scenario, current_state
                )
                actions.extend(scenario_actions)

            # Remove duplicates and rank by effectiveness
            unique_actions = self._deduplicate_actions(actions)
            ranked_actions = self._rank_actions(unique_actions, fault_analysis)

            return ranked_actions[:10]  # Return top 10 actions

        except Exception as e:
            logging.error(f"Error generating corrective actions: {e}")
            return []

    def _generate_actions_for_scenario(
        self, scenario: FaultScenario, state: SimulationState
    ) -> List[CorrectiveAction]:
        """Generate specific actions for a fault scenario."""
        actions = []

        if scenario.fault_type == FaultType.TEMPERATURE_DEVIATION:
            actions.extend(self._generate_temperature_actions(scenario, state))
        elif scenario.fault_type == FaultType.PRESSURE_ANOMALY:
            actions.extend(self._generate_pressure_actions(scenario, state))
        elif scenario.fault_type == FaultType.COMPOSITION_DRIFT:
            actions.extend(self._generate_composition_actions(scenario, state))
        elif scenario.fault_type == FaultType.CONTAMINATION:
            actions.extend(self._generate_contamination_actions(scenario, state))
        elif scenario.fault_type == FaultType.SENSOR_FAILURE:
            actions.extend(self._generate_sensor_actions(scenario, state))
        elif scenario.fault_type == FaultType.EQUIPMENT_MALFUNCTION:
            actions.extend(self._generate_equipment_actions(scenario, state))

        return actions

    def _generate_temperature_actions(
        self, scenario: FaultScenario, state: SimulationState
    ) -> List[CorrectiveAction]:
        """Generate temperature-related corrective actions."""
        actions = []
        current_temp = state.process_conditions.temperature

        if "drop" in scenario.description.lower():
            # Temperature too low - increase heating
            target_temp = current_temp + min(50.0, scenario.fault_magnitude * 100.0)
            actions.append(
                CorrectiveAction(
                    action_id=f"increase_heating_{int(time.time())}",
                    action_type=CorrectiveActionType.PARAMETER_ADJUSTMENT,
                    target_parameter="heater_power",
                    target_value=min(1.0, 0.7 + scenario.fault_magnitude * 0.3),
                    current_value=0.5,
                    priority=1 if scenario.severity == FaultSeverity.CRITICAL else 2,
                    estimated_effectiveness=0.85,
                    implementation_time=30.0,
                    cost_estimate=50.0,
                    risk_level=0.2,
                    description=f"Increase heater power to restore temperature to {target_temp:.1f}K",
                    success_criteria={"temperature": target_temp},
                )
            )

            # Alternative: Reduce cooling
            actions.append(
                CorrectiveAction(
                    action_id=f"reduce_cooling_{int(time.time())}",
                    action_type=CorrectiveActionType.PARAMETER_ADJUSTMENT,
                    target_parameter="cooling_rate",
                    target_value=max(0.1, 0.5 - scenario.fault_magnitude * 0.4),
                    current_value=0.5,
                    priority=2,
                    estimated_effectiveness=0.7,
                    implementation_time=15.0,
                    cost_estimate=20.0,
                    risk_level=0.3,
                    description="Reduce cooling rate to prevent further heat loss",
                    success_criteria={"temperature_rate": 0.0},
                )
            )

        else:
            # Temperature too high - increase cooling
            target_temp = current_temp - min(50.0, scenario.fault_magnitude * 80.0)
            actions.append(
                CorrectiveAction(
                    action_id=f"increase_cooling_{int(time.time())}",
                    action_type=CorrectiveActionType.PARAMETER_ADJUSTMENT,
                    target_parameter="cooling_power",
                    target_value=min(1.0, 0.6 + scenario.fault_magnitude * 0.4),
                    current_value=0.4,
                    priority=1,
                    estimated_effectiveness=0.9,
                    implementation_time=20.0,
                    cost_estimate=30.0,
                    risk_level=0.1,
                    description=f"Increase cooling power to reduce temperature to {target_temp:.1f}K",
                    success_criteria={"temperature": target_temp},
                )
            )

            # Emergency action for critical overheating
            if scenario.severity == FaultSeverity.CRITICAL:
                actions.append(
                    CorrectiveAction(
                        action_id=f"emergency_shutdown_{int(time.time())}",
                        action_type=CorrectiveActionType.EMERGENCY_SHUTDOWN,
                        target_parameter="system_power",
                        target_value=0.0,
                        current_value=1.0,
                        priority=1,
                        estimated_effectiveness=1.0,
                        implementation_time=10.0,
                        cost_estimate=1000.0,
                        risk_level=0.8,
                        description="Emergency shutdown to prevent equipment damage",
                        success_criteria={"temperature_rate": -5.0},
                    )
                )

        return actions

    def _generate_pressure_actions(
        self, scenario: FaultScenario, state: SimulationState
    ) -> List[CorrectiveAction]:
        """Generate pressure-related corrective actions."""
        actions = []
        current_pressure = state.process_conditions.pressure

        if "drop" in scenario.description.lower():
            # Pressure too low
            actions.append(
                CorrectiveAction(
                    action_id=f"increase_pump_speed_{int(time.time())}",
                    action_type=CorrectiveActionType.PARAMETER_ADJUSTMENT,
                    target_parameter="pump_speed",
                    target_value=min(1.0, 0.8 + scenario.fault_magnitude * 0.2),
                    current_value=0.6,
                    priority=1,
                    estimated_effectiveness=0.8,
                    implementation_time=60.0,
                    cost_estimate=40.0,
                    risk_level=0.3,
                    description="Increase pump speed to restore pressure",
                    success_criteria={"pressure": current_pressure * 1.2},
                )
            )

        else:
            # Pressure too high
            actions.append(
                CorrectiveAction(
                    action_id=f"open_relief_valve_{int(time.time())}",
                    action_type=CorrectiveActionType.PARAMETER_ADJUSTMENT,
                    target_parameter="relief_valve_opening",
                    target_value=min(1.0, 0.2 + scenario.fault_magnitude * 0.3),
                    current_value=0.0,
                    priority=1,
                    estimated_effectiveness=0.9,
                    implementation_time=5.0,
                    cost_estimate=10.0,
                    risk_level=0.2,
                    description="Open relief valve to reduce pressure",
                    success_criteria={"pressure": current_pressure * 0.9},
                )
            )

            # Check for blockage
            if scenario.root_cause == FaultCause.EQUIPMENT_WEAR:
                actions.append(
                    CorrectiveAction(
                        action_id=f"cleaning_cycle_{int(time.time())}",
                        action_type=CorrectiveActionType.CLEANING_CYCLE,
                        target_parameter="cleaning_mode",
                        target_value=1.0,
                        current_value=0.0,
                        priority=2,
                        estimated_effectiveness=0.7,
                        implementation_time=300.0,
                        cost_estimate=200.0,
                        risk_level=0.4,
                        description="Initiate cleaning cycle to remove potential blockages",
                        success_criteria={"pressure": current_pressure * 0.85},
                    )
                )

        return actions

    def _generate_composition_actions(
        self, scenario: FaultScenario, state: SimulationState
    ) -> List[CorrectiveAction]:
        """Generate composition-related corrective actions."""
        actions = []

        for param in scenario.affected_parameters:
            if param in state.process_conditions.material_composition:
                current_value = state.process_conditions.material_composition[param]

                # Adjust feed rate
                actions.append(
                    CorrectiveAction(
                        action_id=f"adjust_feed_{param}_{int(time.time())}",
                        action_type=CorrectiveActionType.PARAMETER_ADJUSTMENT,
                        target_parameter=f"feed_rate_{param}",
                        target_value=max(
                            0.1, current_value * (1.0 + scenario.fault_magnitude * 0.5)
                        ),
                        current_value=current_value,
                        priority=2,
                        estimated_effectiveness=0.75,
                        implementation_time=120.0,
                        cost_estimate=60.0,
                        risk_level=0.2,
                        description=f"Adjust feed rate for {param} to correct composition",
                        success_criteria={f"composition_{param}": current_value * 1.05},
                    )
                )

        return actions

    def _generate_contamination_actions(
        self, scenario: FaultScenario, state: SimulationState
    ) -> List[CorrectiveAction]:
        """Generate contamination-related corrective actions."""
        actions = []

        # Quality hold
        actions.append(
            CorrectiveAction(
                action_id=f"quality_hold_{int(time.time())}",
                action_type=CorrectiveActionType.QUALITY_HOLD,
                target_parameter="production_hold",
                target_value=1.0,
                current_value=0.0,
                priority=1,
                estimated_effectiveness=1.0,
                implementation_time=10.0,
                cost_estimate=500.0,
                risk_level=0.1,
                description="Hold production to prevent contaminated product release",
                success_criteria={"production_rate": 0.0},
            )
        )

        # Purge contaminated material
        actions.append(
            CorrectiveAction(
                action_id=f"purge_system_{int(time.time())}",
                action_type=CorrectiveActionType.MATERIAL_REPLACEMENT,
                target_parameter="purge_cycles",
                target_value=3.0,
                current_value=0.0,
                priority=2,
                estimated_effectiveness=0.85,
                implementation_time=600.0,
                cost_estimate=800.0,
                risk_level=0.3,
                description="Purge contaminated material from system",
                success_criteria={"contaminant_level": 0.001},
            )
        )

        return actions

    def _generate_sensor_actions(
        self, scenario: FaultScenario, state: SimulationState
    ) -> List[CorrectiveAction]:
        """Generate sensor-related corrective actions."""
        actions = []

        # Switch to backup sensor
        actions.append(
            CorrectiveAction(
                action_id=f"switch_backup_sensor_{int(time.time())}",
                action_type=CorrectiveActionType.PARAMETER_ADJUSTMENT,
                target_parameter="sensor_selection",
                target_value=1.0,  # Use backup
                current_value=0.0,  # Use primary
                priority=1,
                estimated_effectiveness=0.9,
                implementation_time=30.0,
                cost_estimate=20.0,
                risk_level=0.1,
                description="Switch to backup sensor for critical measurement",
                success_criteria={"sensor_quality": 0.8},
            )
        )

        # Recalibrate sensor
        actions.append(
            CorrectiveAction(
                action_id=f"recalibrate_sensor_{int(time.time())}",
                action_type=CorrectiveActionType.MAINTENANCE_INTERVENTION,
                target_parameter="sensor_calibration",
                target_value=1.0,
                current_value=0.0,
                priority=2,
                estimated_effectiveness=0.8,
                implementation_time=900.0,
                cost_estimate=150.0,
                risk_level=0.2,
                description="Recalibrate faulty sensor",
                success_criteria={"sensor_accuracy": 0.95},
            )
        )

        return actions

    def _generate_equipment_actions(
        self, scenario: FaultScenario, state: SimulationState
    ) -> List[CorrectiveAction]:
        """Generate equipment-related corrective actions."""
        actions = []

        # Reduce load
        actions.append(
            CorrectiveAction(
                action_id=f"reduce_load_{int(time.time())}",
                action_type=CorrectiveActionType.PARAMETER_ADJUSTMENT,
                target_parameter="equipment_load",
                target_value=max(0.3, 0.8 - scenario.fault_magnitude * 0.5),
                current_value=0.8,
                priority=1,
                estimated_effectiveness=0.7,
                implementation_time=60.0,
                cost_estimate=100.0,
                risk_level=0.2,
                description="Reduce equipment load to prevent further damage",
                success_criteria={"equipment_stress": 0.6},
            )
        )

        # Schedule maintenance
        actions.append(
            CorrectiveAction(
                action_id=f"schedule_maintenance_{int(time.time())}",
                action_type=CorrectiveActionType.MAINTENANCE_INTERVENTION,
                target_parameter="maintenance_schedule",
                target_value=1.0,
                current_value=0.0,
                priority=3,
                estimated_effectiveness=0.95,
                implementation_time=7200.0,  # 2 hours
                cost_estimate=2000.0,
                risk_level=0.5,
                description="Schedule preventive maintenance for equipment",
                success_criteria={"equipment_health": 0.9},
            )
        )

        return actions

    def _deduplicate_actions(
        self, actions: List[CorrectiveAction]
    ) -> List[CorrectiveAction]:
        """Remove duplicate actions."""
        unique_actions = []
        seen_combinations = set()

        for action in actions:
            key = (action.action_type, action.target_parameter)
            if key not in seen_combinations:
                seen_combinations.add(key)
                unique_actions.append(action)

        return unique_actions

    def _rank_actions(
        self, actions: List[CorrectiveAction], fault_analysis: FaultAnalysisResult
    ) -> List[CorrectiveAction]:
        """Rank actions by effectiveness and feasibility."""

        def score_action(action):
            # Base score from effectiveness
            score = action.estimated_effectiveness * 0.4

            # Priority bonus (lower number = higher priority)
            priority_bonus = (4 - action.priority) * 0.2
            score += priority_bonus

            # Time penalty (faster implementation is better)
            time_penalty = min(0.2, action.implementation_time / 3600.0)
            score -= time_penalty

            # Risk penalty
            risk_penalty = action.risk_level * 0.15
            score -= risk_penalty

            # Cost penalty (relative to fault severity)
            cost_penalty = min(0.1, action.cost_estimate / 1000.0)
            if fault_analysis.severity_assessment in [
                FaultSeverity.CRITICAL,
                FaultSeverity.MAJOR,
            ]:
                cost_penalty *= 0.5  # Cost less important for severe faults
            score -= cost_penalty

            return score

        # Sort by score (descending)
        scored_actions = [(action, score_action(action)) for action in actions]
        scored_actions.sort(key=lambda x: x[1], reverse=True)

        return [action for action, score in scored_actions]

    def evaluate_action_effectiveness(
        self,
        action: CorrectiveAction,
        pre_state: SimulationState,
        post_state: SimulationState,
    ) -> float:
        """Evaluate effectiveness of a corrective action."""
        try:
            # Check if success criteria were met
            effectiveness = 0.0
            criteria_met = 0

            for criterion, target_value in action.success_criteria.items():
                if criterion == "temperature":
                    actual_value = post_state.process_conditions.temperature
                    if abs(actual_value - target_value) < 10.0:  # Within 10K
                        criteria_met += 1
                elif criterion == "pressure":
                    actual_value = post_state.process_conditions.pressure
                    if (
                        abs(actual_value - target_value) / target_value < 0.1
                    ):  # Within 10%
                        criteria_met += 1

            if action.success_criteria:
                effectiveness = criteria_met / len(action.success_criteria)
            else:
                effectiveness = 0.5  # Default if no criteria

            # Store for learning
            self.action_history.append(
                {
                    "action": action,
                    "effectiveness": effectiveness,
                    "timestamp": time.time(),
                }
            )

            return effectiveness

        except Exception as e:
            logging.error(f"Error evaluating action effectiveness: {e}")
            return 0.0

    def _initialize_action_templates(self) -> Dict[str, Dict]:
        """Initialize templates for common corrective actions."""
        return {
            "temperature_control": {
                "heater_adjustment": {"effectiveness": 0.85, "risk": 0.2},
                "cooling_adjustment": {"effectiveness": 0.9, "risk": 0.1},
                "emergency_shutdown": {"effectiveness": 1.0, "risk": 0.8},
            },
            "pressure_control": {
                "pump_adjustment": {"effectiveness": 0.8, "risk": 0.3},
                "valve_adjustment": {"effectiveness": 0.9, "risk": 0.2},
                "relief_activation": {"effectiveness": 0.95, "risk": 0.1},
            },
            "composition_control": {
                "feed_adjustment": {"effectiveness": 0.75, "risk": 0.2},
                "mixing_optimization": {"effectiveness": 0.7, "risk": 0.3},
            },
        }

    def _create_effectiveness_model(self):
        """Create a simple neural network for action effectiveness prediction."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )


class AlternativeRouteOptimizer:
    """Optimizes alternative process routes when faults occur."""

    def __init__(self):
        self.route_database = self._initialize_route_database()
        self.route_history = deque(maxlen=100)

    def find_alternative_routes(
        self,
        fault_analysis: FaultAnalysisResult,
        current_state: SimulationState,
        target_output: ProcessOutput,
    ) -> List[Dict[str, Any]]:
        """Find alternative process routes to maintain production."""
        alternative_routes = []

        try:
            affected_systems = fault_analysis.affected_systems
            severity = fault_analysis.severity_assessment

            # Generate routes based on affected systems
            if "thermal_control" in affected_systems:
                routes = self._generate_thermal_alternatives(
                    current_state, target_output, severity
                )
                alternative_routes.extend(routes)

            if "pressure_control" in affected_systems:
                routes = self._generate_pressure_alternatives(
                    current_state, target_output, severity
                )
                alternative_routes.extend(routes)

            if "material_handling" in affected_systems:
                routes = self._generate_material_alternatives(
                    current_state, target_output, severity
                )
                alternative_routes.extend(routes)

            # Rank routes by feasibility and effectiveness
            ranked_routes = self._rank_routes(alternative_routes, fault_analysis)

            return ranked_routes[:5]  # Return top 5 alternatives

        except Exception as e:
            logging.error(f"Error finding alternative routes: {e}")
            return []

    def _generate_thermal_alternatives(
        self, state: SimulationState, target: ProcessOutput, severity: FaultSeverity
    ) -> List[Dict[str, Any]]:
        """Generate thermal process alternatives."""
        alternatives = []

        # Lower temperature route
        alternatives.append(
            {
                "route_id": f"low_temp_route_{int(time.time())}",
                "name": "Low Temperature Process Route",
                "description": "Reduce process temperature and extend residence time",
                "modifications": {
                    "temperature": state.process_conditions.temperature * 0.9,
                    "residence_time": 1.5,
                    "pressure": state.process_conditions.pressure * 1.1,
                },
                "estimated_yield": 0.85,
                "estimated_quality": 0.9,
                "implementation_time": 300.0,
                "feasibility": 0.8,
                "resource_requirements": {"energy": 0.7, "time": 1.5},
                "risks": ["Extended cycle time", "Potential incomplete reaction"],
            }
        )

        # Multi-stage heating
        if severity != FaultSeverity.CRITICAL:
            alternatives.append(
                {
                    "route_id": f"multistage_heating_{int(time.time())}",
                    "name": "Multi-Stage Heating Route",
                    "description": "Use multiple heating zones with gradual temperature increase",
                    "modifications": {
                        "heating_stages": 3,
                        "stage_temperatures": [
                            state.process_conditions.temperature * 0.7,
                            state.process_conditions.temperature * 0.85,
                            state.process_conditions.temperature * 0.95,
                        ],
                    },
                    "estimated_yield": 0.92,
                    "estimated_quality": 0.95,
                    "implementation_time": 600.0,
                    "feasibility": 0.7,
                    "resource_requirements": {"energy": 0.9, "equipment": 1.2},
                    "risks": ["Complex control", "Higher equipment utilization"],
                }
            )

        return alternatives

    def _generate_pressure_alternatives(
        self, state: SimulationState, target: ProcessOutput, severity: FaultSeverity
    ) -> List[Dict[str, Any]]:
        """Generate pressure process alternatives."""
        alternatives = []

        # Low pressure route
        alternatives.append(
            {
                "route_id": f"low_pressure_route_{int(time.time())}",
                "name": "Low Pressure Process Route",
                "description": "Operate at reduced pressure with modified conditions",
                "modifications": {
                    "pressure": state.process_conditions.pressure * 0.7,
                    "temperature": state.process_conditions.temperature * 1.05,
                    "catalyst_loading": 1.2,
                },
                "estimated_yield": 0.8,
                "estimated_quality": 0.85,
                "implementation_time": 180.0,
                "feasibility": 0.9,
                "resource_requirements": {"catalyst": 1.2, "energy": 1.1},
                "risks": ["Lower conversion efficiency", "Higher catalyst costs"],
            }
        )

        # Batch processing
        if severity in [FaultSeverity.MAJOR, FaultSeverity.CRITICAL]:
            alternatives.append(
                {
                    "route_id": f"batch_processing_{int(time.time())}",
                    "name": "Emergency Batch Processing",
                    "description": "Switch from continuous to batch operation",
                    "modifications": {
                        "operation_mode": "batch",
                        "batch_size": 0.5,
                        "cycle_time": 120.0,
                    },
                    "estimated_yield": 0.75,
                    "estimated_quality": 0.9,
                    "implementation_time": 900.0,
                    "feasibility": 0.6,
                    "resource_requirements": {"time": 2.0, "labor": 1.5},
                    "risks": ["Reduced throughput", "Higher labor requirements"],
                }
            )

        return alternatives

    def _generate_material_alternatives(
        self, state: SimulationState, target: ProcessOutput, severity: FaultSeverity
    ) -> List[Dict[str, Any]]:
        """Generate material handling alternatives."""
        alternatives = []

        # Alternative feedstock
        alternatives.append(
            {
                "route_id": f"alt_feedstock_{int(time.time())}",
                "name": "Alternative Feedstock Route",
                "description": "Use backup feedstock with adjusted process conditions",
                "modifications": {
                    "feedstock_type": "backup_grade",
                    "composition_adjustment": 0.95,
                    "reaction_conditions": "modified",
                },
                "estimated_yield": 0.88,
                "estimated_quality": 0.85,
                "implementation_time": 240.0,
                "feasibility": 0.8,
                "resource_requirements": {"materials": 1.1, "testing": 1.3},
                "risks": ["Quality variation", "Additional testing required"],
            }
        )

        # Dilution and purification
        if "contamination" in str(state.process_conditions.material_composition):
            alternatives.append(
                {
                    "route_id": f"dilution_purification_{int(time.time())}",
                    "name": "Dilution and Purification Route",
                    "description": "Dilute contaminated stream and add purification step",
                    "modifications": {
                        "dilution_ratio": 2.0,
                        "purification_steps": ["filtration", "distillation"],
                        "recovery_yield": 0.85,
                    },
                    "estimated_yield": 0.7,
                    "estimated_quality": 0.95,
                    "implementation_time": 1200.0,
                    "feasibility": 0.7,
                    "resource_requirements": {"solvents": 2.0, "energy": 1.4},
                    "risks": ["Higher solvent usage", "Extended processing time"],
                }
            )

        return alternatives

    def _rank_routes(
        self, routes: List[Dict[str, Any]], fault_analysis: FaultAnalysisResult
    ) -> List[Dict[str, Any]]:
        """Rank alternative routes by overall score."""

        def score_route(route):
            # Base score from yield and quality
            score = route["estimated_yield"] * 0.3 + route["estimated_quality"] * 0.3

            # Feasibility bonus
            score += route["feasibility"] * 0.2

            # Implementation time penalty
            time_penalty = min(0.15, route["implementation_time"] / 3600.0)
            score -= time_penalty

            # Adjust for fault severity
            if fault_analysis.severity_assessment == FaultSeverity.CRITICAL:
                # Prioritize fast implementation for critical faults
                score += max(0.0, 0.1 - time_penalty * 2)

            return score

        # Sort by score (descending)
        scored_routes = [(route, score_route(route)) for route in routes]
        scored_routes.sort(key=lambda x: x[1], reverse=True)

        return [route for route, score in scored_routes]

    def _initialize_route_database(self) -> Dict[str, Dict]:
        """Initialize database of known alternative routes."""
        return {
            "thermal_routes": {
                "low_temperature": {"effectiveness": 0.8, "complexity": 0.3},
                "multi_stage": {"effectiveness": 0.9, "complexity": 0.7},
            },
            "pressure_routes": {
                "low_pressure": {"effectiveness": 0.75, "complexity": 0.4},
                "batch_mode": {"effectiveness": 0.7, "complexity": 0.8},
            },
            "material_routes": {
                "alternative_feedstock": {"effectiveness": 0.85, "complexity": 0.5},
                "purification": {"effectiveness": 0.9, "complexity": 0.9},
            },
        }


class FaultScenarioSimulator:
    """Main orchestrator for fault scenario simulation and response."""

    def __init__(
        self,
        simulation_core: RealTimeSimulationCore,
        mpc_controller: Optional[ModelPredictiveController] = None,
    ):
        self.simulation_core = simulation_core
        self.mpc_controller = mpc_controller

        # Initialize components
        self.fault_injector = FaultInjector()
        self.fault_detector = FaultDetector()
        self.decision_support = DecisionSupportSystem()
        self.route_optimizer = AlternativeRouteOptimizer()

        # Simulation state
        self.is_running = False
        self.active_scenarios = {}
        self.simulation_history = deque(maxlen=1000)

        # Performance metrics
        self.metrics = {
            "scenarios_simulated": 0,
            "faults_detected": 0,
            "actions_generated": 0,
            "successful_recoveries": 0,
            "average_detection_time": 0.0,
            "average_recovery_time": 0.0,
        }

        # Setup callbacks
        self.fault_injector.add_injection_callback(self._on_fault_injected)

    async def simulate_fault_scenario(self, scenario: FaultScenario) -> Dict[str, Any]:
        """Simulate a complete fault scenario with response."""
        simulation_start = time.time()

        try:
            logging.info(f"Starting fault scenario simulation: {scenario.fault_id}")

            # Get initial state
            initial_state = self.simulation_core.current_state
            if initial_state is None:
                raise ChemiAccelError("No current simulation state available")

            # Set baseline for fault detection
            self.fault_detector.set_baseline(initial_state)

            # Inject fault
            faulted_state = self.fault_injector.inject_fault(scenario, initial_state)

            # Wait for fault onset
            await asyncio.sleep(scenario.onset_time)

            # Detect fault
            detection_start = time.time()
            fault_analysis = self.fault_detector.detect_faults(faulted_state)
            detection_time = time.time() - detection_start

            # Generate corrective actions
            corrective_actions = self.decision_support.generate_corrective_actions(
                fault_analysis, faulted_state
            )

            # Find alternative routes if needed
            target_output = ProcessOutput(
                quality_metrics={"purity": 0.95, "yield": 0.9},
                quantity=100.0,
                processing_time=3600.0,
            )
            alternative_routes = self.route_optimizer.find_alternative_routes(
                fault_analysis, faulted_state, target_output
            )

            # Execute top corrective action (simulation)
            recovery_start = time.time()
            recovery_success = False
            if corrective_actions:
                top_action = corrective_actions[0]
                recovery_success = await self._simulate_action_execution(
                    top_action, faulted_state
                )
            recovery_time = time.time() - recovery_start

            # Update metrics
            self.metrics["scenarios_simulated"] += 1
            if fault_analysis.fault_detected:
                self.metrics["faults_detected"] += 1
            self.metrics["actions_generated"] += len(corrective_actions)
            if recovery_success:
                self.metrics["successful_recoveries"] += 1

            # Update timing metrics
            self.metrics["average_detection_time"] = (
                0.9 * self.metrics["average_detection_time"] + 0.1 * detection_time
            )
            self.metrics["average_recovery_time"] = (
                0.9 * self.metrics["average_recovery_time"] + 0.1 * recovery_time
            )

            # Compile results
            simulation_result = {
                "scenario": scenario.to_dict(),
                "fault_analysis": {
                    "fault_detected": fault_analysis.fault_detected,
                    "severity": fault_analysis.severity_assessment.value,
                    "confidence": fault_analysis.confidence,
                    "affected_systems": fault_analysis.affected_systems,
                    "estimated_impact": fault_analysis.estimated_impact,
                    "time_to_criticality": fault_analysis.time_to_criticality,
                },
                "corrective_actions": [
                    action.to_dict() for action in corrective_actions
                ],
                "alternative_routes": alternative_routes,
                "performance": {
                    "detection_time": detection_time,
                    "recovery_time": recovery_time,
                    "recovery_success": recovery_success,
                    "total_simulation_time": time.time() - simulation_start,
                },
                "timestamp": time.time(),
            }

            # Store in history
            self.simulation_history.append(simulation_result)

            # Clean up
            self.fault_injector.clear_fault(scenario.fault_id)

            logging.info(f"Completed fault scenario simulation: {scenario.fault_id}")
            return simulation_result

        except Exception as e:
            logging.error(f"Error in fault scenario simulation: {e}")
            return {
                "scenario": scenario.to_dict(),
                "error": str(e),
                "timestamp": time.time(),
            }

    async def run_fault_scenario_suite(
        self, scenarios: List[FaultScenario]
    ) -> List[Dict[str, Any]]:
        """Run a suite of fault scenarios."""
        results = []

        for scenario in scenarios:
            try:
                result = await self.simulate_fault_scenario(scenario)
                results.append(result)

                # Wait between scenarios to allow system to stabilize
                await asyncio.sleep(2.0)

            except Exception as e:
                logging.error(f"Error in scenario {scenario.fault_id}: {e}")
                results.append(
                    {
                        "scenario": scenario.to_dict(),
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                )

        return results

    async def _simulate_action_execution(
        self, action: CorrectiveAction, current_state: SimulationState
    ) -> bool:
        """Simulate execution of a corrective action."""
        try:
            # Simulate implementation delay
            await asyncio.sleep(min(1.0, action.implementation_time / 100.0))

            # Simulate action effectiveness
            success_probability = action.estimated_effectiveness * (
                1.0 - action.risk_level * 0.5
            )
            success = np.random.random() < success_probability

            if success:
                logging.info(f"Corrective action succeeded: {action.description}")
            else:
                logging.warning(f"Corrective action failed: {action.description}")

            return success

        except Exception as e:
            logging.error(f"Error simulating action execution: {e}")
            return False

    def _on_fault_injected(self, scenario: FaultScenario, state: SimulationState):
        """Handle fault injection events."""
        logging.info(
            f"Fault injected: {scenario.fault_type.value} - severity: {scenario.severity.value}"
        )

    def create_standard_fault_scenarios(self) -> List[FaultScenario]:
        """Create a set of standard fault scenarios for testing."""
        scenarios = []

        # Temperature deviations
        scenarios.append(
            FaultScenario(
                fault_id="temp_drop_sudden",
                fault_type=FaultType.TEMPERATURE_DEVIATION,
                severity=FaultSeverity.MAJOR,
                description="Sudden temperature drop of 30K due to cooling system malfunction",
                affected_parameters=["temperature"],
                fault_magnitude=0.7,
                onset_time=0.0,
                duration=300.0,
                propagation_rate=2.0,
                root_cause=FaultCause.EQUIPMENT_WEAR,
            )
        )

        scenarios.append(
            FaultScenario(
                fault_id="temp_spike_heater",
                fault_type=FaultType.TEMPERATURE_DEVIATION,
                severity=FaultSeverity.CRITICAL,
                description="Heater malfunction causing temperature spike",
                affected_parameters=["temperature"],
                fault_magnitude=0.9,
                onset_time=0.0,
                duration=None,
                propagation_rate=3.0,
                root_cause=FaultCause.EQUIPMENT_MALFUNCTION,
            )
        )

        # Pressure anomalies
        scenarios.append(
            FaultScenario(
                fault_id="pressure_drop_leak",
                fault_type=FaultType.PRESSURE_ANOMALY,
                severity=FaultSeverity.MODERATE,
                description="Gradual pressure drop due to system leak",
                affected_parameters=["pressure"],
                fault_magnitude=0.4,
                onset_time=60.0,
                duration=None,
                propagation_rate=0.5,
                root_cause=FaultCause.EQUIPMENT_WEAR,
            )
        )

        scenarios.append(
            FaultScenario(
                fault_id="blockage_pressure_spike",
                fault_type=FaultType.PRESSURE_ANOMALY,
                severity=FaultSeverity.MAJOR,
                description="Pressure spike due to line blockage",
                affected_parameters=["pressure"],
                fault_magnitude=0.8,
                onset_time=30.0,
                duration=None,
                propagation_rate=1.5,
                root_cause=FaultCause.BLOCKAGE,
            )
        )

        # Contamination
        scenarios.append(
            FaultScenario(
                fault_id="contaminant_introduction",
                fault_type=FaultType.CONTAMINATION,
                severity=FaultSeverity.MAJOR,
                description="Contaminant introduction through raw material feed",
                affected_parameters=["composition"],
                fault_magnitude=0.6,
                onset_time=120.0,
                duration=600.0,
                propagation_rate=0.3,
                root_cause=FaultCause.MATERIAL_VARIATION,
            )
        )

        # Sensor failures
        scenarios.append(
            FaultScenario(
                fault_id="temp_sensor_drift",
                fault_type=FaultType.SENSOR_FAILURE,
                severity=FaultSeverity.MODERATE,
                description="Temperature sensor calibration drift",
                affected_parameters=["temperature_sensor"],
                fault_magnitude=0.5,
                onset_time=300.0,
                duration=None,
                propagation_rate=0.1,
                root_cause=FaultCause.EQUIPMENT_WEAR,
            )
        )

        # Equipment malfunction
        scenarios.append(
            FaultScenario(
                fault_id="pump_efficiency_loss",
                fault_type=FaultType.EQUIPMENT_MALFUNCTION,
                severity=FaultSeverity.MODERATE,
                description="Pump efficiency loss due to impeller wear",
                affected_parameters=["flow_rate", "pressure"],
                fault_magnitude=0.4,
                onset_time=180.0,
                duration=None,
                propagation_rate=0.2,
                root_cause=FaultCause.EQUIPMENT_WEAR,
            )
        )

        # Power fluctuation
        scenarios.append(
            FaultScenario(
                fault_id="power_fluctuation",
                fault_type=FaultType.POWER_FLUCTUATION,
                severity=FaultSeverity.MINOR,
                description="Brief power fluctuation affecting process stability",
                affected_parameters=["all_systems"],
                fault_magnitude=0.3,
                onset_time=0.0,
                duration=10.0,
                propagation_rate=5.0,
                recovery_time=60.0,
                root_cause=FaultCause.EXTERNAL_CONDITIONS,
            )
        )

        return scenarios

    def export_simulation_results(self, output_path: str):
        """Export simulation results to file."""
        try:
            results_data = {
                "metrics": self.metrics,
                "simulation_history": list(self.simulation_history),
                "export_timestamp": time.time(),
            }

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            logging.info(f"Exported simulation results to {output_path}")

        except Exception as e:
            logging.error(f"Error exporting simulation results: {e}")

    def get_simulation_metrics(self) -> Dict[str, Any]:
        """Get current simulation metrics."""
        return {
            **self.metrics,
            "active_scenarios": len(self.active_scenarios),
            "simulation_history_count": len(self.simulation_history),
            "fault_detection_rate": (
                self.metrics["faults_detected"]
                / max(1, self.metrics["scenarios_simulated"])
            ),
            "recovery_success_rate": (
                self.metrics["successful_recoveries"]
                / max(1, self.metrics["scenarios_simulated"])
            ),
        }


# Convenience functions for creating fault scenarios
def create_temperature_fault(
    magnitude: float = 0.5,
    severity: FaultSeverity = FaultSeverity.MODERATE,
    is_drop: bool = True,
) -> FaultScenario:
    """Create a temperature deviation fault scenario."""
    fault_type = "drop" if is_drop else "spike"
    return FaultScenario(
        fault_id=f"temp_{fault_type}_{int(time.time())}",
        fault_type=FaultType.TEMPERATURE_DEVIATION,
        severity=severity,
        description=f"Temperature {fault_type} with magnitude {magnitude}",
        affected_parameters=["temperature"],
        fault_magnitude=magnitude,
        onset_time=0.0,
        root_cause=(
            FaultCause.EQUIPMENT_WEAR if is_drop else FaultCause.EQUIPMENT_MALFUNCTION
        ),
    )


def create_contamination_fault(
    contaminant_level: float = 0.02, severity: FaultSeverity = FaultSeverity.MAJOR
) -> FaultScenario:
    """Create a contamination fault scenario."""
    return FaultScenario(
        fault_id=f"contamination_{int(time.time())}",
        fault_type=FaultType.CONTAMINATION,
        severity=severity,
        description=f"Material contamination at {contaminant_level:.1%} level",
        affected_parameters=["composition"],
        fault_magnitude=contaminant_level * 20.0,  # Scale to 0-1
        onset_time=60.0,
        duration=300.0,
        propagation_rate=0.5,
        root_cause=FaultCause.MATERIAL_VARIATION,
    )


def create_pressure_fault(
    magnitude: float = 0.3,
    severity: FaultSeverity = FaultSeverity.MODERATE,
    is_drop: bool = True,
) -> FaultScenario:
    """Create a pressure anomaly fault scenario."""
    fault_type = "drop" if is_drop else "spike"
    return FaultScenario(
        fault_id=f"pressure_{fault_type}_{int(time.time())}",
        fault_type=FaultType.PRESSURE_ANOMALY,
        severity=severity,
        description=f"Pressure {fault_type} with relative magnitude {magnitude}",
        affected_parameters=["pressure"],
        fault_magnitude=magnitude,
        onset_time=30.0,
        propagation_rate=1.0,
        root_cause=FaultCause.EQUIPMENT_WEAR if is_drop else FaultCause.BLOCKAGE,
    )

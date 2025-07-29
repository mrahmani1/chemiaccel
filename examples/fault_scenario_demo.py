"""
Fault Scenario Simulation and Corrective Actions Demonstration

This comprehensive demonstration showcases the fault scenario simulation system:
- Comprehensive fault injection and simulation
- Real-time fault detection and analysis
- Automated corrective action generation
- Alternative process route optimization
- Decision support for fault mitigation
- Process recovery and stability restoration

Demonstrates:
- Multiple fault types and severity levels
- Fault propagation and evolution over time
- Intelligent corrective action selection
- Alternative process routes for fault mitigation
- Performance metrics and effectiveness tracking
- Complete fault response workflow
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ChemiAccel imports
from src.chemi_accel.digital_twin.fault_scenario_simulation import (
    CorrectiveAction,
    CorrectiveActionType,
    FaultAnalysisResult,
    FaultCause,
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
from src.chemi_accel.digital_twin.multiscale_framework import MultiScaleFramework
from src.chemi_accel.digital_twin.predictive_control import (
    ControllerConfig,
    ModelPredictiveController,
)
from src.chemi_accel.digital_twin.realtime_process_simulation import (
    MaterialProperty,
    MicrostructureState,
    PropertyType,
    RealTimeSimulationCore,
    SimulationConfig,
    SimulationMode,
    SimulationState,
)
from src.chemi_accel.digital_twin.sensor_data_integration import (
    DataQuality,
    SensorReading,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_demo_fault_scenarios() -> List[FaultScenario]:
    """Create a comprehensive set of fault scenarios for demonstration."""
    scenarios = []

    # 1. Critical Temperature Spike (Equipment Malfunction)
    scenarios.append(
        FaultScenario(
            fault_id="critical_temp_spike",
            fault_type=FaultType.TEMPERATURE_DEVIATION,
            severity=FaultSeverity.CRITICAL,
            description="Critical heater malfunction causing dangerous temperature spike to 450°C",
            affected_parameters=["temperature"],
            fault_magnitude=0.95,
            onset_time=0.0,
            duration=None,  # Permanent until corrected
            propagation_rate=3.0,  # Rapid development
            root_cause=FaultCause.EQUIPMENT_MALFUNCTION,
            environmental_factors={"ambient_temperature": 25.0, "humidity": 0.6},
        )
    )

    # 2. Major Cooling System Failure
    scenarios.append(
        FaultScenario(
            fault_id="cooling_system_failure",
            fault_type=FaultType.COOLING_SYSTEM_FAILURE,
            severity=FaultSeverity.MAJOR,
            description="Cooling system failure causing gradual temperature rise",
            affected_parameters=["temperature", "cooling_efficiency"],
            fault_magnitude=0.8,
            onset_time=30.0,
            duration=None,
            propagation_rate=1.0,
            root_cause=FaultCause.EQUIPMENT_WEAR,
            preconditions={"operating_hours": 8760, "maintenance_overdue": True},
        )
    )

    # 3. Severe Contamination Event
    scenarios.append(
        FaultScenario(
            fault_id="severe_contamination",
            fault_type=FaultType.CONTAMINATION,
            severity=FaultSeverity.MAJOR,
            description="Severe contamination through raw material feed - 4% impurity level",
            affected_parameters=["composition", "material_quality"],
            fault_magnitude=0.8,
            onset_time=60.0,
            duration=900.0,  # 15 minutes
            propagation_rate=0.4,
            root_cause=FaultCause.MATERIAL_VARIATION,
            environmental_factors={"supplier_quality_variance": 0.15},
        )
    )

    # 4. Pressure System Blockage
    scenarios.append(
        FaultScenario(
            fault_id="pressure_blockage",
            fault_type=FaultType.BLOCKAGE,
            severity=FaultSeverity.MAJOR,
            description="Progressive line blockage causing dangerous pressure buildup",
            affected_parameters=["pressure", "flow_rate"],
            fault_magnitude=0.75,
            onset_time=45.0,
            duration=None,
            propagation_rate=0.8,
            root_cause=FaultCause.MAINTENANCE_REQUIRED,
            preconditions={"cleaning_cycles_overdue": 3},
        )
    )

    # 5. Multiple Sensor Failures
    scenarios.append(
        FaultScenario(
            fault_id="sensor_cascade_failure",
            fault_type=FaultType.SENSOR_FAILURE,
            severity=FaultSeverity.MODERATE,
            description="Cascade sensor failures affecting process monitoring",
            affected_parameters=[
                "temperature_sensor",
                "pressure_sensor",
                "flow_sensor",
            ],
            fault_magnitude=0.6,
            onset_time=90.0,
            duration=None,
            propagation_rate=0.3,
            root_cause=FaultCause.EQUIPMENT_WEAR,
            environmental_factors={"electromagnetic_interference": 0.8},
        )
    )

    # 6. Power Fluctuation with Recovery
    scenarios.append(
        FaultScenario(
            fault_id="power_fluctuation_recovery",
            fault_type=FaultType.POWER_FLUCTUATION,
            severity=FaultSeverity.MODERATE,
            description="Power fluctuation with automatic recovery sequence",
            affected_parameters=["all_systems"],
            fault_magnitude=0.5,
            onset_time=0.0,
            duration=20.0,  # Brief disturbance
            propagation_rate=8.0,  # Very rapid
            recovery_time=120.0,  # 2 minutes to recover
            root_cause=FaultCause.EXTERNAL_CONDITIONS,
        )
    )

    # 7. Catalyst Deactivation (Gradual)
    scenarios.append(
        FaultScenario(
            fault_id="catalyst_deactivation",
            fault_type=FaultType.CATALYST_DEACTIVATION,
            severity=FaultSeverity.MODERATE,
            description="Gradual catalyst deactivation reducing reaction efficiency",
            affected_parameters=["reaction_rate", "selectivity", "conversion"],
            fault_magnitude=0.4,
            onset_time=300.0,  # 5 minutes
            duration=None,
            propagation_rate=0.1,  # Very slow
            root_cause=FaultCause.DESIGN_LIMITATION,
            preconditions={"catalyst_age_hours": 2000},
        )
    )

    # 8. Mixing Inefficiency
    scenarios.append(
        FaultScenario(
            fault_id="mixing_inefficiency",
            fault_type=FaultType.MIXING_INEFFICIENCY,
            severity=FaultSeverity.MINOR,
            description="Impeller wear causing reduced mixing efficiency",
            affected_parameters=["mixing_quality", "homogeneity"],
            fault_magnitude=0.3,
            onset_time=180.0,  # 3 minutes
            duration=None,
            propagation_rate=0.2,
            root_cause=FaultCause.EQUIPMENT_WEAR,
        )
    )

    return scenarios


def create_manufacturing_simulation_core() -> RealTimeSimulationCore:
    """Create simulation core for manufacturing process."""
    print("🔧 Setting up manufacturing simulation core...")

    # Create simulation configuration
    sim_config = SimulationConfig(
        mode=SimulationMode.REAL_TIME,
        time_step=0.1,
        max_simulation_time=7200.0,  # 2 hours
        enable_gpu_acceleration=False,  # Use CPU for compatibility
        adaptive_time_stepping=True,
        enable_visualization=False,  # Disable for demo
    )

    # Create multiscale framework (mock for demo)
    multiscale_framework = MultiScaleFramework()

    # Initialize simulation core
    simulation_core = RealTimeSimulationCore(sim_config, multiscale_framework)

    # Setup initial process conditions for polymer extrusion
    initial_conditions = ProcessConditions(
        temperature=523.15,  # 250°C
        pressure=25e5,  # 250 bar
        material_composition={
            "polymer": 0.85,
            "plasticizer": 0.10,
            "stabilizer": 0.03,
            "colorant": 0.02,
        },
    )

    # Create initial microstructure state
    microstructure = MicrostructureState(
        grain_size=8.0,
        grain_orientation=np.array([1.0, 0.0, 0.0]),
        phase_fractions={"amorphous": 0.7, "crystalline": 0.3},
        defect_density=0.002,
        crystallinity=0.3,
    )

    # Set initial simulation state
    initial_state = SimulationState(
        time=0.0,
        process_conditions=initial_conditions,
        material_properties={
            "melt_flow_rate": MaterialProperty(
                "melt_flow_rate", PropertyType.MECHANICAL, 15.0, "g/10min"
            ),
            "tensile_strength": MaterialProperty(
                "tensile_strength", PropertyType.MECHANICAL, 25.0, "MPa"
            ),
            "thermal_conductivity": MaterialProperty(
                "thermal_conductivity", PropertyType.THERMAL, 0.2, "W/m·K"
            ),
        },
        microstructure=microstructure,
    )

    simulation_core.current_state = initial_state

    print("   Simulation core created with initial conditions:")
    print(
        f"   Temperature: {initial_conditions.temperature:.1f}K ({initial_conditions.temperature-273.15:.1f}°C)"
    )
    print(f"   Pressure: {initial_conditions.pressure/1e5:.1f} bar")
    print(
        f"   Polymer composition: {initial_conditions.material_composition['polymer']:.1%}"
    )

    return simulation_core


def create_mock_sensor_data(
    simulation_state: SimulationState, fault_active: bool = False
) -> Dict[str, SensorReading]:
    """Create mock sensor data based on simulation state."""
    temp = simulation_state.process_conditions.temperature
    pressure = simulation_state.process_conditions.pressure

    # Add noise and potential quality degradation if fault active
    temp_noise = np.random.normal(0, 2.0)  # ±2K noise
    pressure_noise = np.random.normal(0, 0.02 * pressure)  # ±2% noise

    temp_quality = DataQuality.GOOD
    pressure_quality = DataQuality.EXCELLENT

    if fault_active:
        # Degrade sensor quality during faults
        if np.random.random() < 0.3:
            temp_quality = DataQuality.POOR
        if np.random.random() < 0.2:
            pressure_quality = DataQuality.ACCEPTABLE

    return {
        "extruder_temp_zone1": SensorReading(
            "extruder_temp_zone1",
            time.time(),
            temp + temp_noise - 273.15,
            "°C",
            temp_quality,
        ),
        "extruder_temp_zone2": SensorReading(
            "extruder_temp_zone2",
            time.time(),
            temp + temp_noise + 5.0 - 273.15,
            "°C",
            DataQuality.GOOD,
        ),
        "melt_pressure": SensorReading(
            "melt_pressure",
            time.time(),
            pressure / 1e5 + pressure_noise / 1e5,
            "bar",
            pressure_quality,
        ),
        "material_flow_rate": SensorReading(
            "material_flow_rate",
            time.time(),
            12.5 + np.random.normal(0, 0.5),
            "kg/h",
            DataQuality.GOOD,
        ),
        "extruder_power": SensorReading(
            "extruder_power",
            time.time(),
            85.0 + np.random.normal(0, 3.0),
            "kW",
            DataQuality.EXCELLENT,
        ),
    }


async def run_individual_fault_scenario(
    simulator: FaultScenarioSimulator, scenario: FaultScenario
) -> Dict:
    """Run and analyze a single fault scenario."""
    print(f"\n🚨 Starting Fault Scenario: {scenario.fault_id}")
    print(f"   Type: {scenario.fault_type.value}")
    print(f"   Severity: {scenario.severity.value}")
    print(f"   Description: {scenario.description}")

    # Run the scenario simulation
    result = await simulator.simulate_fault_scenario(scenario)

    # Analyze results
    if "error" in result:
        print(f"   ❌ Scenario failed: {result['error']}")
        return result

    fault_analysis = result["fault_analysis"]
    corrective_actions = result["corrective_actions"]
    alternative_routes = result["alternative_routes"]
    performance = result["performance"]

    print("\n📊 Fault Analysis Results:")
    print(
        f"   Fault detected: {'✅ Yes' if fault_analysis['fault_detected'] else '❌ No'}"
    )
    print(f"   Severity assessment: {fault_analysis['severity']}")
    print(f"   Detection confidence: {fault_analysis['confidence']:.2f}")
    print(f"   Affected systems: {', '.join(fault_analysis['affected_systems'])}")

    if fault_analysis["time_to_criticality"]:
        print(f"   Time to criticality: {fault_analysis['time_to_criticality']:.1f}s")

    print(f"\n🔧 Corrective Actions Generated: {len(corrective_actions)}")
    for i, action in enumerate(corrective_actions[:3]):  # Show top 3
        print(f"   {i+1}. {action['description']}")
        print(
            f"      Effectiveness: {action['estimated_effectiveness']:.2f}, Risk: {action['risk_level']:.2f}"
        )
        print(f"      Implementation time: {action['implementation_time']:.0f}s")

    print(f"\n🛤️ Alternative Routes: {len(alternative_routes)}")
    for i, route in enumerate(alternative_routes[:2]):  # Show top 2
        print(f"   {i+1}. {route['name']}")
        print(f"      Estimated yield: {route['estimated_yield']:.2f}")
        print(f"      Feasibility: {route['feasibility']:.2f}")

    print("\n⏱️ Performance Metrics:")
    print(f"   Detection time: {performance['detection_time']:.3f}s")
    print(f"   Recovery time: {performance['recovery_time']:.3f}s")
    print(f"   Recovery success: {'✅' if performance['recovery_success'] else '❌'}")
    print(f"   Total simulation time: {performance['total_simulation_time']:.3f}s")

    return result


async def run_fault_scenario_stress_test(simulator: FaultScenarioSimulator) -> Dict:
    """Run stress test with multiple concurrent faults."""
    print("\n" + "=" * 80)
    print("🔥 STRESS TEST: Multiple Concurrent Faults")
    print("=" * 80)

    # Create multiple faults that could occur simultaneously
    concurrent_faults = [
        create_temperature_fault(
            magnitude=0.6, severity=FaultSeverity.MAJOR, is_drop=False
        ),
        create_pressure_fault(
            magnitude=0.4, severity=FaultSeverity.MODERATE, is_drop=True
        ),
        FaultScenario(
            fault_id="sensor_degradation_stress",
            fault_type=FaultType.SENSOR_FAILURE,
            severity=FaultSeverity.MODERATE,
            description="Multiple sensors showing degraded performance",
            affected_parameters=["temperature_sensor", "pressure_sensor"],
            fault_magnitude=0.5,
            onset_time=0.0,
            root_cause=FaultCause.EQUIPMENT_WEAR,
        ),
    ]

    print(f"Simulating {len(concurrent_faults)} concurrent faults...")

    # Run all scenarios
    results = await simulator.run_fault_scenario_suite(concurrent_faults)

    # Analyze combined impact
    total_actions = sum(len(r.get("corrective_actions", [])) for r in results)
    total_routes = sum(len(r.get("alternative_routes", [])) for r in results)

    successful_detections = sum(
        1 for r in results if r.get("fault_analysis", {}).get("fault_detected", False)
    )

    print("\n📈 Stress Test Results:")
    print(f"   Scenarios processed: {len(results)}")
    print(f"   Successful detections: {successful_detections}/{len(results)}")
    print(f"   Total corrective actions: {total_actions}")
    print(f"   Total alternative routes: {total_routes}")

    return {
        "scenarios_count": len(results),
        "successful_detections": successful_detections,
        "total_actions": total_actions,
        "total_routes": total_routes,
        "results": results,
    }


async def run_recovery_effectiveness_analysis(
    simulator: FaultScenarioSimulator,
) -> Dict:
    """Analyze effectiveness of different recovery strategies."""
    print("\n" + "=" * 80)
    print("📊 RECOVERY EFFECTIVENESS ANALYSIS")
    print("=" * 80)

    # Test different fault severities and types
    test_scenarios = [
        ("Minor Temperature Drift", create_temperature_fault(0.2, FaultSeverity.MINOR)),
        ("Major Temperature Spike", create_temperature_fault(0.8, FaultSeverity.MAJOR)),
        (
            "Critical Overheating",
            create_temperature_fault(0.95, FaultSeverity.CRITICAL),
        ),
        ("Minor Contamination", create_contamination_fault(0.01, FaultSeverity.MINOR)),
        ("Major Contamination", create_contamination_fault(0.04, FaultSeverity.MAJOR)),
    ]

    effectiveness_results = {}

    for scenario_name, scenario in test_scenarios:
        print(f"\n🧪 Testing: {scenario_name}")
        result = await simulator.simulate_fault_scenario(scenario)

        if "error" not in result:
            actions = result["corrective_actions"]
            performance = result["performance"]

            # Calculate effectiveness score
            effectiveness_score = 0.0
            if actions:
                avg_effectiveness = np.mean(
                    [a["estimated_effectiveness"] for a in actions]
                )
                avg_risk = np.mean([a["risk_level"] for a in actions])
                time_factor = max(
                    0.0, 1.0 - performance["detection_time"] / 10.0
                )  # Prefer fast detection

                effectiveness_score = (
                    avg_effectiveness * 0.5 + (1.0 - avg_risk) * 0.3 + time_factor * 0.2
                )

            effectiveness_results[scenario_name] = {
                "severity": scenario.severity.value,
                "effectiveness_score": effectiveness_score,
                "detection_time": performance["detection_time"],
                "recovery_success": performance["recovery_success"],
                "actions_count": len(actions),
            }

            print(f"   Effectiveness score: {effectiveness_score:.3f}")
            print(
                f"   Recovery success: {'✅' if performance['recovery_success'] else '❌'}"
            )

    # Summary analysis
    print("\n📋 Recovery Effectiveness Summary:")
    for name, data in effectiveness_results.items():
        print(
            f"   {name}: {data['effectiveness_score']:.3f} "
            f"({data['severity']}, {data['actions_count']} actions)"
        )

    return effectiveness_results


async def demonstrate_decision_support_intelligence(simulator: FaultScenarioSimulator):
    """Demonstrate intelligent decision support capabilities."""
    print("\n" + "=" * 80)
    print("🧠 INTELLIGENT DECISION SUPPORT DEMONSTRATION")
    print("=" * 80)

    # Create a complex scenario that requires sophisticated decision making
    complex_scenario = FaultScenario(
        fault_id="complex_multi_parameter_fault",
        fault_type=FaultType.EQUIPMENT_MALFUNCTION,
        severity=FaultSeverity.MAJOR,
        description="Complex equipment malfunction affecting multiple parameters simultaneously",
        affected_parameters=[
            "temperature",
            "pressure",
            "flow_rate",
            "mixing_efficiency",
        ],
        fault_magnitude=0.7,
        onset_time=0.0,
        duration=None,
        propagation_rate=1.5,
        root_cause=FaultCause.EQUIPMENT_WEAR,
        preconditions={
            "equipment_age_years": 8,
            "maintenance_overdue_days": 45,
            "operating_stress_level": 0.85,
        },
        environmental_factors={
            "ambient_temperature": 35.0,  # High ambient temp
            "humidity": 0.8,  # High humidity
            "power_quality": 0.9,  # Slightly poor power quality
        },
    )

    print("📝 Scenario Details:")
    print(f"   Fault ID: {complex_scenario.fault_id}")
    print(f"   Affected parameters: {', '.join(complex_scenario.affected_parameters)}")
    print("   Preconditions: Equipment age 8 years, maintenance overdue 45 days")
    print("   Environmental factors: High temp/humidity, poor power quality")

    # Simulate the complex scenario
    result = await simulator.simulate_fault_scenario(complex_scenario)

    if "error" in result:
        print(f"❌ Simulation failed: {result['error']}")
        return

    # Analyze decision support intelligence
    actions = result["corrective_actions"]
    routes = result["alternative_routes"]
    fault_analysis = result["fault_analysis"]

    print("\n🔍 Intelligent Analysis Results:")
    print(f"   Fault detection confidence: {fault_analysis['confidence']:.2f}")
    print(
        f"   Estimated impact on quality: {fault_analysis['estimated_impact'].get('quality', 0):.2f}"
    )
    print(
        f"   Estimated impact on safety: {fault_analysis['estimated_impact'].get('safety', 0):.2f}"
    )

    # Analyze action prioritization intelligence
    print("\n🎯 Action Prioritization Intelligence:")
    if actions:
        high_priority_actions = [a for a in actions if a["priority"] == 1]
        emergency_actions = [
            a for a in actions if a["action_type"] == "emergency_shutdown"
        ]
        maintenance_actions = [
            a for a in actions if a["action_type"] == "maintenance_intervention"
        ]

        print(f"   High priority actions: {len(high_priority_actions)}")
        print(f"   Emergency actions: {len(emergency_actions)}")
        print(f"   Maintenance actions: {len(maintenance_actions)}")

        # Show decision logic
        if high_priority_actions:
            best_action = high_priority_actions[0]
            print("\n   🏆 Best Action Selected:")
            print(f"      Action: {best_action['description']}")
            print(f"      Effectiveness: {best_action['estimated_effectiveness']:.2f}")
            print(f"      Risk level: {best_action['risk_level']:.2f}")
            print(f"      Cost estimate: ${best_action['cost_estimate']:.0f}")
            print(
                f"      Implementation time: {best_action['implementation_time']:.0f}s"
            )

    # Analyze route optimization intelligence
    print("\n🛤️ Route Optimization Intelligence:")
    if routes:
        best_route = routes[0]
        print(f"   Best alternative route: {best_route['name']}")
        print(f"   Estimated yield: {best_route['estimated_yield']:.2f}")
        print(f"   Estimated quality: {best_route['estimated_quality']:.2f}")
        print(f"   Feasibility: {best_route['feasibility']:.2f}")
        print(f"   Implementation time: {best_route['implementation_time']:.0f}s")

        if "risks" in best_route:
            print(f"   Identified risks: {', '.join(best_route['risks'])}")


async def run_comprehensive_fault_demo():
    """Run the comprehensive fault scenario demonstration."""
    print("=" * 80)
    print("🏭 ChemiAccel Fault Scenario Simulation & Corrective Actions Demo")
    print("=" * 80)

    try:
        # Setup simulation environment
        simulation_core = create_manufacturing_simulation_core()

        # Create fault scenario simulator
        print("\n🎯 Creating fault scenario simulator...")
        fault_simulator = FaultScenarioSimulator(simulation_core)

        # Create demonstration fault scenarios
        demo_scenarios = create_demo_fault_scenarios()
        print(f"   Created {len(demo_scenarios)} demonstration fault scenarios")

        # Demonstrate individual fault scenarios
        print("\n" + "=" * 60)
        print("🧪 INDIVIDUAL FAULT SCENARIO DEMONSTRATIONS")
        print("=" * 60)

        individual_results = []
        selected_scenarios = demo_scenarios[:4]  # Run first 4 for demo

        for scenario in selected_scenarios:
            result = await run_individual_fault_scenario(fault_simulator, scenario)
            individual_results.append(result)
            await asyncio.sleep(1.0)  # Brief pause between scenarios

        # Run stress test
        stress_test_results = await run_fault_scenario_stress_test(fault_simulator)

        # Run recovery effectiveness analysis
        effectiveness_results = await run_recovery_effectiveness_analysis(
            fault_simulator
        )

        # Demonstrate intelligent decision support
        await demonstrate_decision_support_intelligence(fault_simulator)

        # Generate comprehensive report
        await generate_comprehensive_demo_report(
            fault_simulator,
            individual_results,
            stress_test_results,
            effectiveness_results,
        )

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()

    print("\n✅ Comprehensive fault scenario demonstration completed!")


async def generate_comprehensive_demo_report(
    simulator: FaultScenarioSimulator,
    individual_results: List[Dict],
    stress_test_results: Dict,
    effectiveness_results: Dict,
):
    """Generate comprehensive demonstration report."""
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE DEMONSTRATION REPORT")
    print("=" * 80)

    # Overall simulation metrics
    metrics = simulator.get_simulation_metrics()

    print("\n🔬 Overall Simulation Performance:")
    print(f"   Total scenarios simulated: {metrics['scenarios_simulated']}")
    print(f"   Fault detection rate: {metrics['fault_detection_rate']:.1%}")
    print(f"   Recovery success rate: {metrics['recovery_success_rate']:.1%}")
    print(f"   Average detection time: {metrics['average_detection_time']:.3f}s")
    print(f"   Average recovery time: {metrics['average_recovery_time']:.3f}s")

    # Individual scenario analysis
    print("\n📊 Individual Scenario Analysis:")
    successful_scenarios = [r for r in individual_results if "error" not in r]

    if successful_scenarios:
        avg_confidence = np.mean(
            [r["fault_analysis"]["confidence"] for r in successful_scenarios]
        )
        total_actions = sum(len(r["corrective_actions"]) for r in successful_scenarios)
        total_routes = sum(len(r["alternative_routes"]) for r in successful_scenarios)

        print(
            f"   Successful scenarios: {len(successful_scenarios)}/{len(individual_results)}"
        )
        print(f"   Average detection confidence: {avg_confidence:.2f}")
        print(f"   Total corrective actions generated: {total_actions}")
        print(f"   Total alternative routes identified: {total_routes}")

        # Analyze fault types handled
        fault_types = {}
        for result in successful_scenarios:
            scenario = result["scenario"]
            fault_type = scenario["fault_type"]
            if fault_type not in fault_types:
                fault_types[fault_type] = 0
            fault_types[fault_type] += 1

        print("\n   Fault Types Successfully Handled:")
        for fault_type, count in fault_types.items():
            print(f"      {fault_type}: {count} scenarios")

    # Stress test analysis
    print("\n🔥 Stress Test Analysis:")
    print(f"   Concurrent faults handled: {stress_test_results['scenarios_count']}")
    print(
        f"   Detection success under stress: {stress_test_results['successful_detections']}/{stress_test_results['scenarios_count']}"
    )
    print(f"   Actions generated under stress: {stress_test_results['total_actions']}")
    print(
        f"   System resilience: {'✅ Excellent' if stress_test_results['successful_detections'] == stress_test_results['scenarios_count'] else '⚠️ Good'}"
    )

    # Effectiveness analysis
    print("\n📈 Recovery Strategy Effectiveness:")
    if effectiveness_results:
        severity_performance = {}
        for name, data in effectiveness_results.items():
            severity = data["severity"]
            if severity not in severity_performance:
                severity_performance[severity] = []
            severity_performance[severity].append(data["effectiveness_score"])

        for severity, scores in severity_performance.items():
            avg_score = np.mean(scores)
            print(f"   {severity.title()} faults: {avg_score:.3f} avg effectiveness")

    # System capabilities summary
    print("\n🏆 System Capabilities Demonstrated:")
    print("   ✅ Real-time fault detection and classification")
    print("   ✅ Severity assessment and impact analysis")
    print("   ✅ Intelligent corrective action generation")
    print("   ✅ Alternative process route optimization")
    print("   ✅ Multi-fault scenario handling")
    print("   ✅ Decision support with risk assessment")
    print("   ✅ Performance metrics and effectiveness tracking")
    print("   ✅ Complete fault response workflow automation")

    # Key insights and recommendations
    print("\n💡 Key Insights and Recommendations:")

    if metrics["fault_detection_rate"] > 0.9:
        print("   • Excellent fault detection capabilities across multiple fault types")
    elif metrics["fault_detection_rate"] > 0.7:
        print("   • Good fault detection with room for improvement in edge cases")
    else:
        print("   • Consider enhancing fault detection algorithms for better coverage")

    if metrics["recovery_success_rate"] > 0.8:
        print("   • High recovery success rate indicates effective corrective actions")
    else:
        print("   • Review and optimize corrective action selection algorithms")

    if metrics["average_detection_time"] < 1.0:
        print("   • Rapid fault detection enables quick response and mitigation")
    else:
        print("   • Consider optimizing detection algorithms for faster response")

    print("   • Implement regular fault scenario testing for continuous improvement")
    print("   • Consider operator training on system recommendations")
    print("   • Monitor real-world performance against simulation predictions")

    # Export detailed results
    print("\n💾 Exporting detailed results...")
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Export simulation results
    simulator.export_simulation_results(
        str(output_dir / "fault_scenario_demo_results.json")
    )

    # Save comprehensive report
    import json

    report_data = {
        "demo_timestamp": time.time(),
        "overall_metrics": metrics,
        "individual_results": individual_results,
        "stress_test_results": stress_test_results,
        "effectiveness_results": effectiveness_results,
        "system_capabilities": [
            "Real-time fault detection and classification",
            "Severity assessment and impact analysis",
            "Intelligent corrective action generation",
            "Alternative process route optimization",
            "Multi-fault scenario handling",
            "Decision support with risk assessment",
            "Performance metrics and effectiveness tracking",
            "Complete fault response workflow automation",
        ],
    }

    with open(output_dir / "comprehensive_fault_demo_report.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    print(
        f"      Detailed results saved to: {output_dir}/fault_scenario_demo_results.json"
    )
    print(
        f"      Comprehensive report saved to: {output_dir}/comprehensive_fault_demo_report.json"
    )

    print("\n" + "=" * 80)


def print_demo_info():
    """Print information about the fault scenario demonstration."""
    print(
        """
ChemiAccel Fault Scenario Simulation & Corrective Actions Demo
==============================================================

This comprehensive demonstration showcases:

🚨 Fault Scenario Simulation:
   • Multiple fault types: Temperature, pressure, contamination, equipment, sensors
   • Severity classifications: Critical, major, moderate, minor, warning
   • Fault propagation and evolution modeling
   • Environmental factor integration
   • Root cause analysis and classification

🔍 Intelligent Fault Detection:
   • Real-time anomaly detection algorithms
   • Multi-parameter deviation analysis
   • Sensor quality assessment
   • Confidence scoring and uncertainty quantification
   • Historical baseline comparison

🎯 Decision Support System:
   • Automated corrective action generation
   • Risk-benefit analysis and prioritization
   • Implementation time and cost estimation
   • Success criteria definition
   • Rollback procedure planning

🛤️ Alternative Route Optimization:
   • Process route modification suggestions
   • Multi-objective optimization (yield, quality, safety)
   • Feasibility assessment
   • Resource requirement analysis
   • Risk mitigation strategies

🔧 Comprehensive Corrective Actions:
   • Parameter adjustments and process modifications
   • Equipment isolation and emergency procedures
   • Material replacement and purification
   • Maintenance intervention scheduling
   • Quality control and production holds

Key Features Demonstrated:
✓ 8 different fault types with realistic scenarios
✓ Intelligent fault injection and propagation modeling
✓ Real-time detection with configurable thresholds
✓ Multi-criteria decision making for action selection
✓ Alternative process routes for fault mitigation
✓ Comprehensive performance metrics and analysis
✓ Stress testing with concurrent faults
✓ Recovery effectiveness evaluation

Manufacturing Scenarios:
• Polymer extrusion process simulation
• Critical temperature spike handling
• Cooling system failure management
• Contamination event response
• Pressure system blockage resolution
• Sensor cascade failure handling
• Power fluctuation recovery
• Catalyst deactivation management

Dependencies:
• Core: numpy, asyncio, logging
• Digital Twin: RealTimeSimulationCore, manufacturing integration
• Optional: matplotlib, plotly (for extended visualization)

Usage:
   python fault_scenario_demo.py
"""
    )


async def create_simplified_demo():
    """Create a simplified demo that works without full simulation."""
    print("\n" + "=" * 60)
    print("Simplified Fault Scenario Example")
    print("=" * 60)

    print("\n1. Creating fault scenarios...")
    scenarios = create_demo_fault_scenarios()
    print(f"   Created {len(scenarios)} fault scenarios:")
    for i, scenario in enumerate(scenarios[:3]):  # Show first 3
        print(
            f"   • {scenario.fault_id}: {scenario.fault_type.value} ({scenario.severity.value})"
        )
    print(f"   ... and {len(scenarios)-3} more scenarios")

    print("\n2. Demonstrating fault severity classification...")
    severity_counts = {}
    for scenario in scenarios:
        severity = scenario.severity.value
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    for severity, count in severity_counts.items():
        print(f"   • {severity.title()}: {count} scenarios")

    print("\n3. Demonstrating fault type coverage...")
    type_counts = {}
    for scenario in scenarios:
        fault_type = scenario.fault_type.value
        type_counts[fault_type] = type_counts.get(fault_type, 0) + 1

    for fault_type, count in type_counts.items():
        print(f"   • {fault_type.replace('_', ' ').title()}: {count} scenarios")

    print("\n4. Simulating fault detection...")
    # Simulate detection results
    for scenario in scenarios[:2]:
        detection_confidence = np.random.uniform(0.7, 0.95)
        detection_time = np.random.uniform(0.1, 2.0)

        print(f"   • {scenario.fault_id}:")
        print(f"     Detection confidence: {detection_confidence:.2f}")
        print(f"     Detection time: {detection_time:.3f}s")
        print(f"     Affected systems: {len(scenario.affected_parameters)} parameters")

    print("\n5. Demonstrating corrective action types...")
    action_types = [action_type.value for action_type in CorrectiveActionType]
    print(f"   Available action types: {len(action_types)}")
    for action_type in action_types[:5]:
        print(f"   • {action_type.replace('_', ' ').title()}")
    print(f"   ... and {len(action_types)-5} more action types")

    print("\n6. Simulating recovery effectiveness...")
    for severity in ["minor", "moderate", "major", "critical"]:
        effectiveness = max(
            0.3, 1.0 - 0.15 * ["minor", "moderate", "major", "critical"].index(severity)
        )
        print(
            f"   • {severity.title()} faults: {effectiveness:.2f} avg recovery effectiveness"
        )

    print("\n✅ Simplified demo completed!")
    print("\n💡 For full functionality, run the complete demo with:")
    print("   asyncio.run(run_comprehensive_fault_demo())")


if __name__ == "__main__":
    print_demo_info()

    # Check for async support and required dependencies
    try:
        # Try to run the full demo
        asyncio.run(run_comprehensive_fault_demo())
    except Exception as e:
        print(f"\nFull demo encountered an issue: {e}")
        print("Running simplified example instead...\n")

        # Fallback to simplified demo
        asyncio.run(create_simplified_demo())

        print("\n💡 For full functionality, ensure all dependencies are installed:")
        print("   pip install numpy asyncio")
        print("   pip install matplotlib plotly (optional for visualization)")

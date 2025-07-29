"""
Autonomous Lab Protocol Generation and Execution Demo.

This example demonstrates the comprehensive autonomous laboratory capabilities
including protocol generation, equipment drivers, machine-executable protocol
translation, and automated experimental execution.
"""

import asyncio
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemi_accel.ai.autonomous_lab import (  # Core classes; Data structures; Enums; Utility functions
    ActionType,
    EquipmentManager,
    EquipmentType,
    ExecutionResult,
    ExperimentalAction,
    ExperimentalProtocol,
    LabwareDefinition,
    LiquidHandlerDriver,
    ProtocolFormat,
    ProtocolGenerator,
    ReactorDriver,
    Reagent,
    SpectrometerDriver,
    Vessel,
    create_standard_labware,
    create_standard_reagents,
    run_autonomous_experiment,
    setup_autonomous_lab,
)

# Check for optional dependencies
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AutonomousLabDemo:
    """Demonstration of autonomous laboratory capabilities."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        self.temp_dir = None

        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available. YAML protocol export disabled.")

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    async def demo_equipment_setup_and_connection(self) -> Dict[str, Any]:
        """Demonstrate equipment setup and connection."""
        self._log("=== Equipment Setup and Connection Demo ===")

        # Create equipment manager
        manager = EquipmentManager()

        # Create equipment definitions
        labware_definitions = [
            LabwareDefinition(
                name="Hamilton STAR",
                equipment_type=EquipmentType.LIQUID_HANDLER,
                model="STAR",
                manufacturer="Hamilton",
                serial_number="HS2023001",
                capabilities=["liquid_transfer", "mixing", "washing", "tip_handling"],
                parameters={
                    "max_volume": 1000.0,
                    "min_volume": 0.5,
                    "accuracy": 0.02,
                    "precision": 0.01,
                    "tip_capacity": 96,
                },
            ),
            LabwareDefinition(
                name="IKA RCT Digital",
                equipment_type=EquipmentType.REACTOR,
                model="RCT Digital",
                manufacturer="IKA",
                serial_number="IKA2023001",
                capabilities=[
                    "heating",
                    "stirring",
                    "temperature_control",
                    "safety_monitoring",
                ],
                parameters={
                    "max_temperature": 300.0,
                    "temperature_accuracy": 0.5,
                    "stirring_range": [50, 1500],
                    "volume_capacity": 500.0,
                },
            ),
            LabwareDefinition(
                name="Ocean Optics USB4000",
                equipment_type=EquipmentType.SPECTROMETER,
                model="USB4000",
                manufacturer="Ocean Optics",
                serial_number="OO2023001",
                capabilities=[
                    "uv_vis_spectroscopy",
                    "kinetics",
                    "absorbance",
                    "fluorescence",
                ],
                parameters={
                    "wavelength_range": [200, 1100],
                    "resolution": 0.3,
                    "integration_time_range": [0.003, 65],
                    "detector_pixels": 3648,
                },
            ),
            LabwareDefinition(
                name="Eppendorf Centrifuge",
                equipment_type=EquipmentType.CENTRIFUGE,
                model="5430R",
                manufacturer="Eppendorf",
                serial_number="EP2023001",
                capabilities=[
                    "centrifugation",
                    "temperature_control",
                    "rotor_detection",
                ],
                parameters={
                    "max_speed": 30000,
                    "max_rcf": 30130,
                    "temperature_range": [-10, 40],
                    "rotor_capacity": 24,
                },
            ),
        ]

        # Register equipment drivers
        equipment_count = 0
        for labware in labware_definitions:
            if labware.equipment_type == EquipmentType.LIQUID_HANDLER:
                driver = LiquidHandlerDriver(labware)
                equipment_id = f"liquid_handler_{equipment_count:03d}"
            elif labware.equipment_type == EquipmentType.REACTOR:
                driver = ReactorDriver(labware)
                equipment_id = f"reactor_{equipment_count:03d}"
            elif labware.equipment_type == EquipmentType.SPECTROMETER:
                driver = SpectrometerDriver(labware)
                equipment_id = f"spectrometer_{equipment_count:03d}"
            else:
                # Create a mock driver for other equipment types
                driver = MockEquipmentDriver(labware)
                equipment_id = f"equipment_{equipment_count:03d}"

            manager.register_equipment(equipment_id, driver)
            equipment_count += 1

        self._log(f"Registered {equipment_count} pieces of equipment")

        # Connect to all equipment
        connection_results = await manager.connect_all_equipment()

        connected_count = sum(
            1 for connected in connection_results.values() if connected
        )
        self._log(
            f"Successfully connected to {connected_count}/{len(connection_results)} devices"
        )

        # Get equipment status
        status_results = await manager.get_equipment_status()

        self._log("Equipment status:")
        for equipment_id, status in status_results.items():
            if "error" not in status:
                self._log(f"  {equipment_id}: {status.get('status', 'unknown')}")
            else:
                self._log(f"  {equipment_id}: ERROR - {status['error']}")

        # Calibrate equipment
        calibration_results = await manager.calibrate_equipment()

        calibrated_count = sum(1 for success in calibration_results.values() if success)
        self._log(
            f"Successfully calibrated {calibrated_count}/{len(calibration_results)} devices"
        )

        return {
            "equipment_registered": equipment_count,
            "equipment_connected": connected_count,
            "equipment_calibrated": calibrated_count,
            "connection_results": connection_results,
            "calibration_results": calibration_results,
        }

    async def demo_protocol_generation(self) -> Dict[str, Any]:
        """Demonstrate protocol generation from experimental conditions."""
        self._log("=== Protocol Generation Demo ===")

        # Define experimental scenarios
        scenarios = [
            {
                "name": "Chemical Synthesis",
                "conditions": {
                    "temperature": 85.0,
                    "stirring_speed": 300.0,
                    "time": 2.0,  # hours
                    "catalyst_loading": 0.05,  # mol%
                    "atmosphere": "N2",
                },
                "reagents": ["Benzaldehyde", "Acetone", "Sodium Hydroxide", "Ethanol"],
            },
            {
                "name": "Polymer Synthesis",
                "conditions": {
                    "temperature": 70.0,
                    "stirring_speed": 250.0,
                    "time": 4.0,
                    "initiator_concentration": 0.1,
                    "monomer_concentration": 2.0,
                },
                "reagents": ["Styrene", "AIBN", "Toluene"],
            },
            {
                "name": "Analytical Measurement",
                "conditions": {
                    "temperature": 25.0,
                    "measurement_interval": 30.0,  # seconds
                    "wavelength_range": [200, 800],
                    "integration_time": 1.0,
                },
                "reagents": ["Sample_Solution", "Reference_Solution"],
            },
        ]

        # Create protocol generators for different formats
        formats = [
            ProtocolFormat.JSON_PROTOCOL,
            ProtocolFormat.XML_PROTOCOL,
            ProtocolFormat.PYTHON_SCRIPT,
            ProtocolFormat.HAMILTON_VENUS,
            ProtocolFormat.OPENTRONS,
        ]

        if YAML_AVAILABLE:
            formats.append(ProtocolFormat.YAML_PROTOCOL)

        # Standard labware and reagents
        labware = create_standard_labware()
        all_reagents = create_standard_reagents()

        protocols_generated = {}

        for scenario in scenarios:
            scenario_name = scenario["name"]
            self._log(f"Generating protocols for {scenario_name}")

            # Filter reagents for this scenario
            scenario_reagents = [
                reagent
                for reagent in all_reagents
                if reagent.name in scenario.get("reagents", [])
            ]

            # Add missing reagents
            for reagent_name in scenario.get("reagents", []):
                if not any(r.name == reagent_name for r in scenario_reagents):
                    scenario_reagents.append(
                        Reagent(
                            name=reagent_name,
                            cas_number="000-00-0",  # Placeholder
                            molecular_weight=100.0,  # Placeholder
                            concentration=1.0,
                        )
                    )

            scenario_protocols = {}

            for format_type in formats:
                try:
                    generator = ProtocolGenerator(format_type)

                    protocol = generator.generate_protocol(
                        scenario["conditions"], labware, scenario_reagents
                    )

                    # Export protocol
                    if self.temp_dir is None:
                        self.temp_dir = Path(tempfile.mkdtemp())

                    filename = f"{scenario_name.lower().replace(' ', '_')}_{format_type.value}.txt"
                    output_path = self.temp_dir / filename

                    protocol_content = generator.export_protocol(protocol, output_path)

                    scenario_protocols[format_type.value] = {
                        "protocol_id": protocol.protocol_id,
                        "actions_count": len(protocol.actions),
                        "estimated_duration": protocol.estimated_duration,
                        "file_path": str(output_path),
                        "content_length": len(protocol_content),
                    }

                    self._log(
                        f"  Generated {format_type.value} protocol: {len(protocol.actions)} actions, "
                        f"{protocol.estimated_duration:.1f}s estimated duration"
                    )

                except Exception as e:
                    self._log(f"  Failed to generate {format_type.value} protocol: {e}")
                    scenario_protocols[format_type.value] = {"error": str(e)}

            protocols_generated[scenario_name] = scenario_protocols

        total_protocols = sum(
            len([p for p in protocols.values() if "error" not in p])
            for protocols in protocols_generated.values()
        )

        self._log(
            f"Successfully generated {total_protocols} protocols across {len(scenarios)} scenarios"
        )

        return {
            "scenarios_processed": len(scenarios),
            "protocols_generated": total_protocols,
            "formats_tested": len(formats),
            "protocols_by_scenario": protocols_generated,
            "output_directory": str(self.temp_dir) if self.temp_dir else None,
        }

    async def demo_protocol_execution(self) -> Dict[str, Any]:
        """Demonstrate protocol execution with equipment drivers."""
        self._log("=== Protocol Execution Demo ===")

        # Set up equipment manager
        manager = await setup_autonomous_lab()

        # Create a comprehensive experimental protocol
        reagents = [
            Reagent(name="Benzaldehyde", cas_number="100-52-7", concentration=1.0),
            Reagent(name="Acetone", cas_number="67-64-1", concentration=2.0),
            Reagent(name="NaOH", cas_number="1310-73-2", concentration=0.5),
            Reagent(name="Ethanol", cas_number="64-17-5", concentration=1.0),
        ]

        vessels = [
            Vessel(id="reaction_vial", vessel_type="vial", volume_capacity=2000.0),
            Vessel(
                id="reagent_reservoir_1",
                vessel_type="reservoir",
                volume_capacity=10000.0,
            ),
            Vessel(
                id="reagent_reservoir_2",
                vessel_type="reservoir",
                volume_capacity=10000.0,
            ),
            Vessel(id="waste_container", vessel_type="waste", volume_capacity=50000.0),
        ]

        # Create detailed experimental actions
        actions = [
            # Step 1: Add first reagent
            ExperimentalAction(
                action_id="action_001",
                action_type=ActionType.TRANSFER_LIQUID,
                equipment="liquid_handler_000",
                parameters={
                    "volume": 500.0,  # µL
                    "source": "reagent_reservoir_1",
                    "destination": "reaction_vial",
                    "speed": "normal",
                    "reagent": "Benzaldehyde",
                },
                reagents=[reagents[0]],
                duration=15.0,
            ),
            # Step 2: Add second reagent
            ExperimentalAction(
                action_id="action_002",
                action_type=ActionType.TRANSFER_LIQUID,
                equipment="liquid_handler_000",
                parameters={
                    "volume": 250.0,
                    "source": "reagent_reservoir_2",
                    "destination": "reaction_vial",
                    "speed": "slow",
                    "reagent": "Acetone",
                },
                reagents=[reagents[1]],
                duration=20.0,
            ),
            # Step 3: Initial mixing
            ExperimentalAction(
                action_id="action_003",
                action_type=ActionType.MIX,
                equipment="liquid_handler_000",
                parameters={
                    "mix_cycles": 5,
                    "mix_volume": 200.0,
                    "mix_speed": "medium",
                    "position": "reaction_vial",
                },
                duration=30.0,
            ),
            # Step 4: Heat to reaction temperature
            ExperimentalAction(
                action_id="action_004",
                action_type=ActionType.HEAT,
                equipment="reactor_000",
                parameters={"temperature": 85.0, "ramp_rate": 5.0, "atmosphere": "N2"},
                temperature=85.0,
                duration=180.0,
            ),
            # Step 5: Start stirring
            ExperimentalAction(
                action_id="action_005",
                action_type=ActionType.MIX,
                equipment="reactor_000",
                parameters={"stirring_speed": 300.0, "duration": 3600.0},  # 1 hour
                duration=3600.0,
            ),
            # Step 6: Add catalyst solution
            ExperimentalAction(
                action_id="action_006",
                action_type=ActionType.TRANSFER_LIQUID,
                equipment="liquid_handler_000",
                parameters={
                    "volume": 50.0,
                    "source": "reagent_reservoir_1",
                    "destination": "reaction_vial",
                    "speed": "very_slow",
                    "reagent": "NaOH",
                },
                reagents=[reagents[2]],
                duration=30.0,
            ),
            # Step 7: React with stirring and heating
            ExperimentalAction(
                action_id="action_007",
                action_type=ActionType.INCUBATE,
                equipment="reactor_000",
                parameters={
                    "temperature": 85.0,
                    "duration": 7200.0,  # 2 hours
                    "stirring_speed": 300.0,
                    "atmosphere": "N2",
                },
                temperature=85.0,
                duration=7200.0,
            ),
            # Step 8: Cool down
            ExperimentalAction(
                action_id="action_008",
                action_type=ActionType.COOL,
                equipment="reactor_000",
                parameters={"temperature": 25.0, "cooling_rate": 2.0},
                temperature=25.0,
                duration=300.0,
            ),
            # Step 9: Take UV-Vis measurement
            ExperimentalAction(
                action_id="action_009",
                action_type=ActionType.MEASURE,
                equipment="spectrometer_000",
                parameters={
                    "measurement_type": "uv_vis",
                    "wavelength_start": 200,
                    "wavelength_end": 800,
                    "integration_time": 1.0,
                    "sample": "reaction_vial",
                },
                duration=10.0,
            ),
            # Step 10: Cleanup - wash tips
            ExperimentalAction(
                action_id="action_010",
                action_type=ActionType.WASH,
                equipment="liquid_handler_000",
                parameters={
                    "wash_cycles": 3,
                    "wash_solvent": "Ethanol",
                    "dry_time": 30.0,
                },
                reagents=[reagents[3]],
                duration=60.0,
            ),
        ]

        # Create complete protocol
        protocol = ExperimentalProtocol(
            protocol_id="comprehensive_synthesis_001",
            name="Comprehensive Chemical Synthesis Protocol",
            description="Multi-step synthesis with automated liquid handling, reaction control, and analysis",
            actions=actions,
            labware=create_standard_labware(),
            reagents=reagents,
            vessels=vessels,
            safety_requirements=[
                "Fume hood operation required",
                "Personal protective equipment mandatory",
                "Temperature monitoring essential",
                "Emergency shutdown procedures active",
            ],
        )

        self._log(f"Created protocol with {len(actions)} actions")
        self._log(
            f"Estimated total duration: {protocol.estimated_duration:.1f} seconds ({protocol.estimated_duration/3600:.2f} hours)"
        )

        # Execute protocol
        start_time = datetime.now()
        execution_results = await manager.execute_protocol(protocol)
        end_time = datetime.now()

        execution_duration = (end_time - start_time).total_seconds()

        # Analyze execution results
        successful_actions = sum(1 for result in execution_results if result.success)
        failed_actions = len(execution_results) - successful_actions

        # Collect measurements
        all_measurements = {}
        for result in execution_results:
            if result.measurements:
                all_measurements.update(result.measurements)

        # Calculate performance metrics
        total_liquid_transferred = sum(
            measurement
            for key, measurement in all_measurements.items()
            if "volume" in key and isinstance(measurement, (int, float))
        )

        self._log(f"Protocol execution completed in {execution_duration:.1f} seconds")
        self._log(
            f"Success rate: {successful_actions}/{len(execution_results)} ({successful_actions/len(execution_results)*100:.1f}%)"
        )
        self._log(f"Total measurements collected: {len(all_measurements)}")

        if total_liquid_transferred > 0:
            self._log(f"Total liquid transferred: {total_liquid_transferred:.1f} µL")

        # Equipment status after execution
        final_equipment_status = await manager.get_equipment_status()

        return {
            "protocol_id": protocol.protocol_id,
            "total_actions": len(execution_results),
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
            "success_rate": successful_actions / len(execution_results),
            "execution_duration": execution_duration,
            "estimated_duration": protocol.estimated_duration,
            "measurements_collected": len(all_measurements),
            "total_liquid_transferred": total_liquid_transferred,
            "equipment_status": final_equipment_status,
            "execution_results": [
                {
                    "action_id": result.action_id,
                    "success": result.success,
                    "duration": (result.end_time - result.start_time).total_seconds(),
                    "error": result.error_message,
                }
                for result in execution_results
            ],
        }

    async def demo_multi_format_protocol_export(self) -> Dict[str, Any]:
        """Demonstrate protocol export in multiple formats."""
        self._log("=== Multi-Format Protocol Export Demo ===")

        # Create a sample protocol
        experimental_conditions = {
            "temperature": 80.0,
            "stirring_speed": 300.0,
            "time": 1.5,
            "ph": 7.5,
            "atmosphere": "N2",
        }

        labware = create_standard_labware()
        reagents = create_standard_reagents()[:3]  # Use first 3 reagents

        formats_to_test = [
            ProtocolFormat.JSON_PROTOCOL,
            ProtocolFormat.XML_PROTOCOL,
            ProtocolFormat.PYTHON_SCRIPT,
            ProtocolFormat.HAMILTON_VENUS,
            ProtocolFormat.OPENTRONS,
        ]

        if YAML_AVAILABLE:
            formats_to_test.append(ProtocolFormat.YAML_PROTOCOL)

        export_results = {}

        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp())

        for format_type in formats_to_test:
            try:
                self._log(f"Exporting to {format_type.value} format...")

                generator = ProtocolGenerator(format_type)
                protocol = generator.generate_protocol(
                    experimental_conditions, labware, reagents
                )

                # Export to file
                filename = f"demo_protocol_{format_type.value}"
                if format_type == ProtocolFormat.JSON_PROTOCOL:
                    filename += ".json"
                elif format_type == ProtocolFormat.XML_PROTOCOL:
                    filename += ".xml"
                elif format_type == ProtocolFormat.YAML_PROTOCOL:
                    filename += ".yml"
                elif format_type == ProtocolFormat.PYTHON_SCRIPT:
                    filename += ".py"
                elif format_type in [
                    ProtocolFormat.HAMILTON_VENUS,
                    ProtocolFormat.OPENTRONS,
                ]:
                    filename += ".txt"

                output_path = self.temp_dir / filename
                content = generator.export_protocol(protocol, output_path)

                # Analyze content
                lines = content.count("\n")
                words = len(content.split())
                characters = len(content)

                export_results[format_type.value] = {
                    "success": True,
                    "file_path": str(output_path),
                    "content_stats": {
                        "lines": lines,
                        "words": words,
                        "characters": characters,
                    },
                    "protocol_actions": len(protocol.actions),
                    "estimated_duration": protocol.estimated_duration,
                }

                self._log(
                    f"  Exported {format_type.value}: {lines} lines, {words} words, {characters} characters"
                )

            except Exception as e:
                self._log(f"  Failed to export {format_type.value}: {e}")
                export_results[format_type.value] = {"success": False, "error": str(e)}

        successful_exports = sum(
            1 for result in export_results.values() if result.get("success", False)
        )

        self._log(
            f"Successfully exported to {successful_exports}/{len(formats_to_test)} formats"
        )

        return {
            "formats_tested": len(formats_to_test),
            "successful_exports": successful_exports,
            "export_results": export_results,
            "output_directory": str(self.temp_dir),
        }

    async def demo_error_handling_and_recovery(self) -> Dict[str, Any]:
        """Demonstrate error handling and recovery mechanisms."""
        self._log("=== Error Handling and Recovery Demo ===")

        # Create equipment manager with one failing device
        manager = EquipmentManager()

        # Register normal equipment
        normal_labware = LabwareDefinition(
            name="Normal Liquid Handler",
            equipment_type=EquipmentType.LIQUID_HANDLER,
            model="NLH001",
            manufacturer="TestCorp",
        )
        normal_driver = LiquidHandlerDriver(normal_labware)
        manager.register_equipment("liquid_handler_normal", normal_driver)

        # Register a failing device
        failing_labware = LabwareDefinition(
            name="Failing Reactor",
            equipment_type=EquipmentType.REACTOR,
            model="FR001",
            manufacturer="TestCorp",
        )
        failing_driver = FailingReactorDriver(failing_labware)
        manager.register_equipment("reactor_failing", failing_driver)

        # Connect equipment (some may fail)
        connection_results = await manager.connect_all_equipment()

        # Create protocol with actions that will fail
        actions = [
            # This should succeed
            ExperimentalAction(
                action_id="action_success",
                action_type=ActionType.TRANSFER_LIQUID,
                equipment="liquid_handler_normal",
                parameters={"volume": 100.0, "source": "A1", "destination": "B1"},
            ),
            # This will fail (connection issue)
            ExperimentalAction(
                action_id="action_fail_connection",
                action_type=ActionType.HEAT,
                equipment="reactor_failing",
                parameters={"temperature": 80.0},
            ),
            # This will fail (equipment not found)
            ExperimentalAction(
                action_id="action_fail_not_found",
                action_type=ActionType.MEASURE,
                equipment="nonexistent_spectrometer",
                parameters={"wavelength_start": 200, "wavelength_end": 800},
            ),
            # This should succeed again
            ExperimentalAction(
                action_id="action_success_2",
                action_type=ActionType.MIX,
                equipment="liquid_handler_normal",
                parameters={"mix_cycles": 3, "mix_volume": 50.0},
            ),
        ]

        protocol = ExperimentalProtocol(
            protocol_id="error_handling_test",
            name="Error Handling Test Protocol",
            description="Protocol designed to test error handling capabilities",
            actions=actions,
            labware=[normal_labware, failing_labware],
            reagents=[],
            vessels=[],
        )

        # Execute protocol and handle errors
        execution_results = await manager.execute_protocol(protocol)

        # Analyze error types and recovery
        success_count = 0
        connection_errors = 0
        not_found_errors = 0
        other_errors = 0

        for result in execution_results:
            if result.success:
                success_count += 1
            else:
                error_msg = result.error_message.lower() if result.error_message else ""
                if "connection" in error_msg or "connect" in error_msg:
                    connection_errors += 1
                elif "not found" in error_msg or "no suitable equipment" in error_msg:
                    not_found_errors += 1
                else:
                    other_errors += 1

        self._log("Error handling test completed:")
        self._log(f"  Successful actions: {success_count}")
        self._log(f"  Connection errors: {connection_errors}")
        self._log(f"  Equipment not found errors: {not_found_errors}")
        self._log(f"  Other errors: {other_errors}")

        # Test equipment status with errors
        status_results = await manager.get_equipment_status()
        equipment_with_errors = sum(
            1 for status in status_results.values() if "error" in status
        )

        self._log(
            f"Equipment with status errors: {equipment_with_errors}/{len(status_results)}"
        )

        return {
            "total_actions": len(execution_results),
            "successful_actions": success_count,
            "connection_errors": connection_errors,
            "not_found_errors": not_found_errors,
            "other_errors": other_errors,
            "equipment_with_errors": equipment_with_errors,
            "recovery_rate": success_count / len(execution_results),
            "connection_results": connection_results,
            "execution_details": [
                {
                    "action_id": result.action_id,
                    "success": result.success,
                    "error_type": self._classify_error(result.error_message)
                    if not result.success
                    else None,
                }
                for result in execution_results
            ],
        }

    def _classify_error(self, error_message: str) -> str:
        """Classify error message type."""
        if not error_message:
            return "unknown"

        error_lower = error_message.lower()
        if "connection" in error_lower or "connect" in error_lower:
            return "connection"
        elif "not found" in error_lower or "no suitable equipment" in error_lower:
            return "equipment_not_found"
        elif "timeout" in error_lower:
            return "timeout"
        elif "communication" in error_lower:
            return "communication"
        else:
            return "other"

    async def run_all_demos(self) -> Dict[str, Any]:
        """Run all demonstration scenarios."""
        self._log("Starting Autonomous Lab Demonstrations")
        self._log("=" * 60)

        all_results = {}

        try:
            # Demo 1: Equipment Setup and Connection
            all_results[
                "equipment_setup"
            ] = await self.demo_equipment_setup_and_connection()

            # Demo 2: Protocol Generation
            all_results["protocol_generation"] = await self.demo_protocol_generation()

            # Demo 3: Protocol Execution
            all_results["protocol_execution"] = await self.demo_protocol_execution()

            # Demo 4: Multi-Format Export
            all_results[
                "multi_format_export"
            ] = await self.demo_multi_format_protocol_export()

            # Demo 5: Error Handling
            all_results[
                "error_handling"
            ] = await self.demo_error_handling_and_recovery()

            self._log("=" * 60)
            self._log("All demonstrations completed successfully!")

            # Summary statistics
            total_protocols = all_results.get("protocol_generation", {}).get(
                "protocols_generated", 0
            )
            total_actions = all_results.get("protocol_execution", {}).get(
                "total_actions", 0
            )
            success_rate = all_results.get("protocol_execution", {}).get(
                "success_rate", 0
            )

            self._log(f"Total protocols generated: {total_protocols}")
            self._log(f"Total actions executed: {total_actions}")
            self._log(f"Overall success rate: {success_rate:.1%}")

            return all_results

        except Exception as e:
            logger.error(f"Error during demonstrations: {e}")
            all_results["error"] = str(e)
            return all_results

        finally:
            # Cleanup temporary directory
            if self.temp_dir and self.temp_dir.exists():
                import shutil

                try:
                    shutil.rmtree(self.temp_dir)
                    self._log(f"Cleaned up temporary directory: {self.temp_dir}")
                except Exception as e:
                    self._log(f"Could not clean up temporary directory: {e}")


class MockEquipmentDriver:
    """Mock equipment driver for demonstration purposes."""

    def __init__(self, labware_def):
        self.labware_def = labware_def
        self.connected = False
        self.status = "idle"

    async def connect(self):
        await asyncio.sleep(0.1)
        self.connected = True
        return True

    async def disconnect(self):
        self.connected = False
        return True

    async def execute_action(self, action):
        start_time = datetime.now()
        await asyncio.sleep(0.5)  # Simulate action execution
        end_time = datetime.now()

        return ExecutionResult(
            protocol_id=action.action_id,
            action_id=action.action_id,
            success=True,
            start_time=start_time,
            end_time=end_time,
            actual_parameters=action.parameters,
            measurements={"mock_measurement": 42.0},
        )

    async def get_status(self):
        return {
            "connected": self.connected,
            "status": self.status,
            "mock_parameter": 100.0,
        }

    async def calibrate(self):
        await asyncio.sleep(1.0)
        return True


class FailingReactorDriver(ReactorDriver):
    """Reactor driver that simulates failures for testing."""

    async def connect(self):
        # Simulate connection failure
        await asyncio.sleep(0.1)
        return False

    async def execute_action(self, action):
        start_time = datetime.now()
        end_time = start_time

        return ExecutionResult(
            protocol_id=action.action_id,
            action_id=action.action_id,
            success=False,
            start_time=start_time,
            end_time=end_time,
            actual_parameters=action.parameters,
            error_message="Simulated reactor connection failure",
        )

    async def get_status(self):
        return {"error": "Device not responding"}


async def main():
    """Main demonstration function."""
    print("Autonomous Lab Protocol Generation and Execution Demo")
    print("=" * 80)

    # Initialize demo
    demo = AutonomousLabDemo(verbose=True)

    # Run all demonstrations
    results = await demo.run_all_demos()

    # Save results
    results_file = Path("autonomous_lab_demo_results.json")

    # Convert datetime objects to strings for JSON serialization
    def convert_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: convert_datetime(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_datetime(item) for item in obj]
        else:
            return obj

    serializable_results = convert_datetime(results)

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)

    for demo_name, demo_results in results.items():
        if isinstance(demo_results, dict) and "error" not in demo_results:
            print(f"\n{demo_name.replace('_', ' ').title()}:")

            if "equipment_connected" in demo_results:
                connected = demo_results["equipment_connected"]
                registered = demo_results["equipment_registered"]
                print(f"  Equipment Connection: {connected}/{registered}")

            if "protocols_generated" in demo_results:
                print(f"  Protocols Generated: {demo_results['protocols_generated']}")

            if "successful_actions" in demo_results:
                success = demo_results["successful_actions"]
                total = demo_results["total_actions"]
                rate = demo_results.get("success_rate", 0) * 100
                print(f"  Action Success Rate: {success}/{total} ({rate:.1f}%)")

            if "successful_exports" in demo_results:
                exports = demo_results["successful_exports"]
                formats = demo_results["formats_tested"]
                print(f"  Export Success: {exports}/{formats} formats")

    print("\nTechnology Stack:")
    print(f"  PyYAML: {'Available' if YAML_AVAILABLE else 'Not Available'}")
    print(f"  Requests: {'Available' if REQUESTS_AVAILABLE else 'Not Available'}")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

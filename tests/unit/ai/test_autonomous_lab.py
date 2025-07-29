"""
Tests for Autonomous Lab Protocol Generation and Execution.

This module provides comprehensive testing for the autonomous laboratory
framework including protocol generation, equipment drivers, and
machine-executable protocol translation.
"""

import asyncio
import json
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from chemi_accel.ai.autonomous_lab import (  # Core classes; Data structures; Enums; Utility functions; Exceptions
    ActionType,
    AutonomousLabError,
    CommunicationProtocol,
    EquipmentCommunicationError,
    EquipmentManager,
    EquipmentType,
    ExecutionResult,
    ExperimentalAction,
    ExperimentalProtocol,
    LabwareDefinition,
    LiquidHandlerDriver,
    ProtocolFormat,
    ProtocolGenerationError,
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
    import serial

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestReagent:
    """Test reagent representation."""

    def test_reagent_creation(self):
        """Test creating reagent objects."""
        reagent = Reagent(
            name="Benzaldehyde",
            cas_number="100-52-7",
            molecular_weight=106.12,
            density=1.044,
            concentration=1.0,
            purity=0.99,
        )

        assert reagent.name == "Benzaldehyde"
        assert reagent.cas_number == "100-52-7"
        assert reagent.molecular_weight == 106.12
        assert reagent.density == 1.044
        assert reagent.concentration == 1.0
        assert reagent.purity == 0.99

    def test_reagent_with_safety_data(self):
        """Test reagent with safety information."""
        safety_data = {
            "hazard_class": "Flammable liquid",
            "signal_word": "Warning",
            "precautionary_statements": ["P210", "P233", "P240"],
        }

        reagent = Reagent(name="Ethanol", cas_number="64-17-5", safety_data=safety_data)

        assert reagent.safety_data == safety_data
        assert reagent.safety_data["hazard_class"] == "Flammable liquid"


class TestVessel:
    """Test laboratory vessel representation."""

    def test_vessel_creation(self):
        """Test creating vessel objects."""
        vessel = Vessel(
            id="reaction_vial_001",
            vessel_type="reaction_vial",
            volume_capacity=2000.0,
            current_volume=1500.0,
            position=(1, 2),
            barcode="RV001",
        )

        assert vessel.id == "reaction_vial_001"
        assert vessel.vessel_type == "reaction_vial"
        assert vessel.volume_capacity == 2000.0
        assert vessel.current_volume == 1500.0
        assert vessel.position == (1, 2)
        assert vessel.barcode == "RV001"
        assert vessel.is_sealed is False

    def test_vessel_with_contents(self):
        """Test vessel with reagent contents."""
        reagent = Reagent(name="Water", cas_number="7732-18-5")

        vessel = Vessel(
            id="water_reservoir",
            vessel_type="reservoir",
            volume_capacity=50000.0,
            current_volume=45000.0,
            contents=[reagent],
        )

        assert len(vessel.contents) == 1
        assert vessel.contents[0].name == "Water"


class TestLabwareDefinition:
    """Test laboratory equipment definitions."""

    def test_labware_creation(self):
        """Test creating labware definitions."""
        labware = LabwareDefinition(
            name="Hamilton STAR",
            equipment_type=EquipmentType.LIQUID_HANDLER,
            model="STAR",
            manufacturer="Hamilton",
            serial_number="HS001",
            capabilities=["liquid_transfer", "mixing", "washing"],
        )

        assert labware.name == "Hamilton STAR"
        assert labware.equipment_type == EquipmentType.LIQUID_HANDLER
        assert labware.model == "STAR"
        assert labware.manufacturer == "Hamilton"
        assert labware.serial_number == "HS001"
        assert "liquid_transfer" in labware.capabilities

    def test_labware_with_parameters(self):
        """Test labware with configuration parameters."""
        parameters = {
            "max_volume": 1000.0,
            "min_volume": 0.5,
            "accuracy": 0.02,
            "precision": 0.01,
        }

        labware = LabwareDefinition(
            name="Test Pipette",
            equipment_type=EquipmentType.PIPETTE,
            model="P1000",
            manufacturer="TestCorp",
            parameters=parameters,
        )

        assert labware.parameters == parameters
        assert labware.parameters["max_volume"] == 1000.0


class TestExperimentalAction:
    """Test experimental action representation."""

    def test_action_creation(self):
        """Test creating experimental actions."""
        action = ExperimentalAction(
            action_id="action_001",
            action_type=ActionType.TRANSFER_LIQUID,
            equipment="liquid_handler_001",
            parameters={"volume": 100.0, "source": "A1", "destination": "B1"},
            duration=10.0,
        )

        assert action.action_id == "action_001"
        assert action.action_type == ActionType.TRANSFER_LIQUID
        assert action.equipment == "liquid_handler_001"
        assert action.parameters["volume"] == 100.0
        assert action.duration == 10.0

    def test_action_with_reagents(self):
        """Test action with reagent specifications."""
        reagent = Reagent(name="Acetone", cas_number="67-64-1")

        action = ExperimentalAction(
            action_id="action_002",
            action_type=ActionType.DISPENSE,
            equipment="dispenser_001",
            parameters={"volume": 50.0},
            reagents=[reagent],
            source_vessels=["reservoir_1"],
            destination_vessels=["reaction_vial_1"],
        )

        assert len(action.reagents) == 1
        assert action.reagents[0].name == "Acetone"
        assert action.source_vessels == ["reservoir_1"]
        assert action.destination_vessels == ["reaction_vial_1"]


class TestExperimentalProtocol:
    """Test experimental protocol representation."""

    def test_protocol_creation(self):
        """Test creating experimental protocols."""
        actions = [
            ExperimentalAction(
                action_id="action_001",
                action_type=ActionType.TRANSFER_LIQUID,
                equipment="liquid_handler_001",
                parameters={"volume": 100.0},
            )
        ]

        labware = [
            LabwareDefinition(
                name="Test Handler",
                equipment_type=EquipmentType.LIQUID_HANDLER,
                model="TH001",
                manufacturer="TestCorp",
            )
        ]

        reagents = [Reagent(name="Water", cas_number="7732-18-5")]
        vessels = [Vessel(id="vial_1", vessel_type="vial", volume_capacity=2000.0)]

        protocol = ExperimentalProtocol(
            protocol_id="protocol_001",
            name="Test Protocol",
            description="A test protocol",
            actions=actions,
            labware=labware,
            reagents=reagents,
            vessels=vessels,
        )

        assert protocol.protocol_id == "protocol_001"
        assert protocol.name == "Test Protocol"
        assert len(protocol.actions) == 1
        assert len(protocol.labware) == 1
        assert len(protocol.reagents) == 1
        assert len(protocol.vessels) == 1
        assert protocol.version == "1.0"
        assert isinstance(protocol.created_date, datetime)


class TestExecutionResult:
    """Test execution result representation."""

    def test_execution_result_success(self):
        """Test successful execution result."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)

        result = ExecutionResult(
            protocol_id="protocol_001",
            action_id="action_001",
            success=True,
            start_time=start_time,
            end_time=end_time,
            actual_parameters={"volume": 100.0},
            measurements={"volume_transferred": 99.8},
        )

        assert result.success is True
        assert result.protocol_id == "protocol_001"
        assert result.action_id == "action_001"
        assert result.measurements["volume_transferred"] == 99.8
        assert result.error_message is None

    def test_execution_result_failure(self):
        """Test failed execution result."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=5)

        result = ExecutionResult(
            protocol_id="protocol_001",
            action_id="action_001",
            success=False,
            start_time=start_time,
            end_time=end_time,
            actual_parameters={"volume": 100.0},
            error_message="Pipette tip not detected",
        )

        assert result.success is False
        assert result.error_message == "Pipette tip not detected"


class TestLiquidHandlerDriver:
    """Test liquid handler equipment driver."""

    def setup_method(self):
        """Set up test liquid handler."""
        labware_def = LabwareDefinition(
            name="Test Liquid Handler",
            equipment_type=EquipmentType.LIQUID_HANDLER,
            model="TLH001",
            manufacturer="TestCorp",
        )
        self.driver = LiquidHandlerDriver(labware_def)

    @pytest.mark.asyncio
    async def test_connection(self):
        """Test equipment connection."""
        assert self.driver.connected is False

        success = await self.driver.connect()
        assert success is True
        assert self.driver.connected is True

        success = await self.driver.disconnect()
        assert success is True
        assert self.driver.connected is False

    @pytest.mark.asyncio
    async def test_liquid_transfer(self):
        """Test liquid transfer action."""
        await self.driver.connect()

        action = ExperimentalAction(
            action_id="transfer_001",
            action_type=ActionType.TRANSFER_LIQUID,
            equipment="liquid_handler_001",
            parameters={"volume": 100.0, "source": "A1", "destination": "B1"},
        )

        result = await self.driver.execute_action(action)

        assert result.success is True
        assert result.action_id == "transfer_001"
        assert "volume_transferred" in result.measurements
        assert result.measurements["volume_transferred"] == 100.0

    @pytest.mark.asyncio
    async def test_mixing_action(self):
        """Test mixing action."""
        await self.driver.connect()

        action = ExperimentalAction(
            action_id="mix_001",
            action_type=ActionType.MIX,
            equipment="liquid_handler_001",
            parameters={"mix_cycles": 5, "mix_volume": 50.0},
        )

        result = await self.driver.execute_action(action)

        assert result.success is True
        assert "mix_cycles" in result.measurements
        assert result.measurements["mix_cycles"] == 5

    @pytest.mark.asyncio
    async def test_unsupported_action(self):
        """Test handling of unsupported actions."""
        await self.driver.connect()

        action = ExperimentalAction(
            action_id="invalid_001",
            action_type=ActionType.HEAT,  # Not supported by liquid handler
            equipment="liquid_handler_001",
            parameters={},
        )

        result = await self.driver.execute_action(action)

        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_status_reporting(self):
        """Test equipment status reporting."""
        await self.driver.connect()

        status = await self.driver.get_status()

        assert isinstance(status, dict)
        assert "connected" in status
        assert status["connected"] is True
        assert "status" in status
        assert "temperature" in status
        assert "humidity" in status

    @pytest.mark.asyncio
    async def test_calibration(self):
        """Test equipment calibration."""
        await self.driver.connect()

        success = await self.driver.calibrate()
        assert success is True


class TestReactorDriver:
    """Test reactor equipment driver."""

    def setup_method(self):
        """Set up test reactor."""
        labware_def = LabwareDefinition(
            name="Test Reactor",
            equipment_type=EquipmentType.REACTOR,
            model="TR001",
            manufacturer="TestCorp",
        )
        self.driver = ReactorDriver(labware_def)

    @pytest.mark.asyncio
    async def test_heating_action(self):
        """Test heating action."""
        await self.driver.connect()

        action = ExperimentalAction(
            action_id="heat_001",
            action_type=ActionType.HEAT,
            equipment="reactor_001",
            parameters={"temperature": 80.0, "ramp_rate": 5.0},
        )

        result = await self.driver.execute_action(action)

        assert result.success is True
        assert "final_temperature" in result.measurements
        assert result.measurements["final_temperature"] == 80.0
        assert self.driver.current_temperature == 80.0

    @pytest.mark.asyncio
    async def test_cooling_action(self):
        """Test cooling action."""
        await self.driver.connect()

        # First heat up
        self.driver.current_temperature = 100.0

        action = ExperimentalAction(
            action_id="cool_001",
            action_type=ActionType.COOL,
            equipment="reactor_001",
            parameters={"temperature": 60.0, "cooling_rate": 3.0},
        )

        result = await self.driver.execute_action(action)

        assert result.success is True
        assert "final_temperature" in result.measurements
        assert result.measurements["final_temperature"] == 60.0
        assert self.driver.current_temperature == 60.0

    @pytest.mark.asyncio
    async def test_stirring_action(self):
        """Test stirring action."""
        await self.driver.connect()

        action = ExperimentalAction(
            action_id="stir_001",
            action_type=ActionType.MIX,
            equipment="reactor_001",
            parameters={"stirring_speed": 300.0, "duration": 60.0},
        )

        result = await self.driver.execute_action(action)

        assert result.success is True
        assert "stirring_speed" in result.measurements
        assert result.measurements["stirring_speed"] == 300.0
        assert self.driver.stirring_speed == 300.0

    @pytest.mark.asyncio
    async def test_incubation_action(self):
        """Test incubation action."""
        await self.driver.connect()

        action = ExperimentalAction(
            action_id="incubate_001",
            action_type=ActionType.INCUBATE,
            equipment="reactor_001",
            parameters={
                "temperature": 75.0,
                "duration": 3600.0,
                "stirring_speed": 250.0,
            },
        )

        result = await self.driver.execute_action(action)

        assert result.success is True
        assert "incubation_temperature" in result.measurements
        assert result.measurements["incubation_temperature"] == 75.0
        assert "reaction_progress" in result.measurements

    @pytest.mark.asyncio
    async def test_wait_action(self):
        """Test wait action."""
        await self.driver.connect()

        action = ExperimentalAction(
            action_id="wait_001",
            action_type=ActionType.WAIT,
            equipment="reactor_001",
            parameters={"duration": 30.0},
        )

        result = await self.driver.execute_action(action)

        assert result.success is True
        assert "wait_time" in result.measurements
        assert result.measurements["wait_time"] == 30.0


class TestSpectrometerDriver:
    """Test spectrometer equipment driver."""

    def setup_method(self):
        """Set up test spectrometer."""
        labware_def = LabwareDefinition(
            name="Test Spectrometer",
            equipment_type=EquipmentType.SPECTROMETER,
            model="TS001",
            manufacturer="TestCorp",
        )
        self.driver = SpectrometerDriver(labware_def)

    @pytest.mark.asyncio
    async def test_spectrum_measurement(self):
        """Test spectrum measurement."""
        await self.driver.connect()

        action = ExperimentalAction(
            action_id="measure_001",
            action_type=ActionType.MEASURE,
            equipment="spectrometer_001",
            parameters={
                "wavelength_start": 250,
                "wavelength_end": 750,
                "integration_time": 1.0,
            },
        )

        result = await self.driver.execute_action(action)

        assert result.success is True
        assert "wavelengths" in result.measurements
        assert "absorbance" in result.measurements
        assert "signal_to_noise" in result.measurements

        wavelengths = result.measurements["wavelengths"]
        absorbance = result.measurements["absorbance"]

        assert len(wavelengths) == len(absorbance)
        assert min(wavelengths) >= 250
        assert max(wavelengths) <= 750

    @pytest.mark.asyncio
    async def test_status_reporting(self):
        """Test spectrometer status."""
        await self.driver.connect()

        status = await self.driver.get_status()

        assert isinstance(status, dict)
        assert "connected" in status
        assert "lamp_hours" in status
        assert "wavelength_range" in status
        assert "resolution" in status


class TestProtocolGenerator:
    """Test protocol generation."""

    def setup_method(self):
        """Set up test protocol generator."""
        self.generator = ProtocolGenerator(ProtocolFormat.JSON_PROTOCOL)
        self.labware = create_standard_labware()
        self.reagents = create_standard_reagents()

    def test_generator_creation(self):
        """Test protocol generator initialization."""
        assert self.generator.format_type == ProtocolFormat.JSON_PROTOCOL
        assert isinstance(self.generator.action_templates, dict)
        assert "liquid_transfer" in self.generator.action_templates
        assert "heating" in self.generator.action_templates

    def test_protocol_generation(self):
        """Test generating protocol from conditions."""
        experimental_conditions = {
            "temperature": 80.0,
            "stirring_speed": 300.0,
            "time": 2.0,  # hours
        }

        protocol = self.generator.generate_protocol(
            experimental_conditions, self.labware, self.reagents
        )

        assert isinstance(protocol, ExperimentalProtocol)
        assert len(protocol.actions) > 0
        assert len(protocol.vessels) > 0
        assert protocol.reagents == self.reagents
        assert protocol.labware == self.labware
        assert protocol.estimated_duration > 0

    def test_json_export(self):
        """Test JSON protocol export."""
        experimental_conditions = {"temperature": 80.0, "time": 1.0}

        protocol = self.generator.generate_protocol(
            experimental_conditions,
            self.labware[:1],  # Use only one piece of labware
            self.reagents[:1],  # Use only one reagent
        )

        json_content = self.generator.export_protocol(protocol)

        assert isinstance(json_content, str)

        # Parse JSON to verify structure
        protocol_dict = json.loads(json_content)
        assert "protocol_id" in protocol_dict
        assert "actions" in protocol_dict
        assert "labware" in protocol_dict
        assert "reagents" in protocol_dict
        assert "vessels" in protocol_dict

    def test_xml_export(self):
        """Test XML protocol export."""
        self.generator.format_type = ProtocolFormat.XML_PROTOCOL

        experimental_conditions = {"temperature": 80.0}

        protocol = self.generator.generate_protocol(
            experimental_conditions, self.labware[:1], self.reagents[:1]
        )

        xml_content = self.generator.export_protocol(protocol)

        assert isinstance(xml_content, str)

        # Parse XML to verify structure
        root = ET.fromstring(xml_content)
        assert root.tag == "experimental_protocol"
        assert root.get("id") == protocol.protocol_id

        metadata = root.find("metadata")
        assert metadata is not None
        assert metadata.find("name").text == protocol.name

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not available")
    def test_yaml_export(self):
        """Test YAML protocol export."""
        self.generator.format_type = ProtocolFormat.YAML_PROTOCOL

        experimental_conditions = {"temperature": 80.0}

        protocol = self.generator.generate_protocol(
            experimental_conditions, self.labware[:1], self.reagents[:1]
        )

        yaml_content = self.generator.export_protocol(protocol)

        assert isinstance(yaml_content, str)
        assert "protocol_id:" in yaml_content
        assert "actions:" in yaml_content

    def test_python_export(self):
        """Test Python script export."""
        self.generator.format_type = ProtocolFormat.PYTHON_SCRIPT

        experimental_conditions = {"temperature": 80.0}

        protocol = self.generator.generate_protocol(
            experimental_conditions, self.labware[:1], self.reagents[:1]
        )

        python_content = self.generator.export_protocol(protocol)

        assert isinstance(python_content, str)
        assert "#!/usr/bin/env python3" in python_content
        assert "async def run_protocol():" in python_content
        assert "asyncio.run(run_protocol())" in python_content

    def test_hamilton_venus_export(self):
        """Test Hamilton Venus protocol export."""
        self.generator.format_type = ProtocolFormat.HAMILTON_VENUS

        experimental_conditions = {"temperature": 80.0}

        protocol = self.generator.generate_protocol(
            experimental_conditions, self.labware[:1], self.reagents[:1]
        )

        venus_content = self.generator.export_protocol(protocol)

        assert isinstance(venus_content, str)
        assert "// Hamilton Venus Protocol" in venus_content
        assert "method " in venus_content
        assert "Venus::Initialize();" in venus_content
        assert "Venus::Terminate();" in venus_content

    def test_opentrons_export(self):
        """Test Opentrons protocol export."""
        self.generator.format_type = ProtocolFormat.OPENTRONS

        experimental_conditions = {"temperature": 80.0}

        protocol = self.generator.generate_protocol(
            experimental_conditions, self.labware[:1], self.reagents[:1]
        )

        opentrons_content = self.generator.export_protocol(protocol)

        assert isinstance(opentrons_content, str)
        assert "from opentrons import protocol_api" in opentrons_content
        assert "def run(protocol: protocol_api.ProtocolContext):" in opentrons_content
        assert "load_labware" in opentrons_content
        assert "load_instrument" in opentrons_content

    def test_file_export(self):
        """Test exporting protocol to file."""
        experimental_conditions = {"temperature": 80.0}

        protocol = self.generator.generate_protocol(
            experimental_conditions, self.labware[:1], self.reagents[:1]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_protocol.json"

            json_content = self.generator.export_protocol(protocol, output_path)

            assert output_path.exists()

            # Verify file content matches returned content
            file_content = output_path.read_text()
            assert file_content == json_content

    def test_unsupported_format_error(self):
        """Test error handling for unsupported formats."""
        # Manually set an invalid format (bypass enum validation)
        self.generator.format_type = "invalid_format"

        experimental_conditions = {"temperature": 80.0}

        protocol = self.generator.generate_protocol(
            experimental_conditions, self.labware[:1], self.reagents[:1]
        )

        with pytest.raises(ProtocolGenerationError):
            self.generator.export_protocol(protocol)


class TestEquipmentManager:
    """Test equipment management."""

    def setup_method(self):
        """Set up test equipment manager."""
        self.manager = EquipmentManager()

        # Create test drivers
        labware_def = LabwareDefinition(
            name="Test Liquid Handler",
            equipment_type=EquipmentType.LIQUID_HANDLER,
            model="TLH001",
            manufacturer="TestCorp",
        )
        self.liquid_handler = LiquidHandlerDriver(labware_def)

        reactor_def = LabwareDefinition(
            name="Test Reactor",
            equipment_type=EquipmentType.REACTOR,
            model="TR001",
            manufacturer="TestCorp",
        )
        self.reactor = ReactorDriver(reactor_def)

    def test_equipment_registration(self):
        """Test registering equipment."""
        assert len(self.manager.equipment_drivers) == 0

        self.manager.register_equipment("liquid_handler_001", self.liquid_handler)
        self.manager.register_equipment("reactor_001", self.reactor)

        assert len(self.manager.equipment_drivers) == 2
        assert "liquid_handler_001" in self.manager.equipment_drivers
        assert "reactor_001" in self.manager.equipment_drivers

    @pytest.mark.asyncio
    async def test_connect_all_equipment(self):
        """Test connecting to all equipment."""
        self.manager.register_equipment("liquid_handler_001", self.liquid_handler)
        self.manager.register_equipment("reactor_001", self.reactor)

        results = await self.manager.connect_all_equipment()

        assert len(results) == 2
        assert results["liquid_handler_001"] is True
        assert results["reactor_001"] is True
        assert self.liquid_handler.connected is True
        assert self.reactor.connected is True

    @pytest.mark.asyncio
    async def test_protocol_execution(self):
        """Test executing experimental protocol."""
        # Register equipment
        self.manager.register_equipment("liquid_handler_001", self.liquid_handler)
        self.manager.register_equipment("reactor_001", self.reactor)

        # Connect equipment
        await self.manager.connect_all_equipment()

        # Create test protocol
        actions = [
            ExperimentalAction(
                action_id="action_001",
                action_type=ActionType.TRANSFER_LIQUID,
                equipment="liquid_handler_001",
                parameters={"volume": 100.0, "source": "A1", "destination": "B1"},
            ),
            ExperimentalAction(
                action_id="action_002",
                action_type=ActionType.HEAT,
                equipment="reactor_001",
                parameters={"temperature": 80.0},
            ),
        ]

        protocol = ExperimentalProtocol(
            protocol_id="test_protocol",
            name="Test Protocol",
            description="Test protocol execution",
            actions=actions,
            labware=[],
            reagents=[],
            vessels=[],
        )

        # Execute protocol
        results = await self.manager.execute_protocol(protocol)

        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].action_id == "action_001"
        assert results[1].action_id == "action_002"

    @pytest.mark.asyncio
    async def test_equipment_not_found(self):
        """Test handling when equipment is not found."""
        # Register only liquid handler
        self.manager.register_equipment("liquid_handler_001", self.liquid_handler)
        await self.manager.connect_all_equipment()

        # Create protocol requiring non-existent equipment
        actions = [
            ExperimentalAction(
                action_id="action_001",
                action_type=ActionType.HEAT,
                equipment="nonexistent_reactor",
                parameters={"temperature": 80.0},
            )
        ]

        protocol = ExperimentalProtocol(
            protocol_id="test_protocol",
            name="Test Protocol",
            description="Test with missing equipment",
            actions=actions,
            labware=[],
            reagents=[],
            vessels=[],
        )

        results = await self.manager.execute_protocol(protocol)

        assert len(results) == 1
        assert results[0].success is False
        assert "No suitable equipment found" in results[0].error_message

    @pytest.mark.asyncio
    async def test_equipment_status(self):
        """Test getting equipment status."""
        self.manager.register_equipment("liquid_handler_001", self.liquid_handler)
        self.manager.register_equipment("reactor_001", self.reactor)

        await self.manager.connect_all_equipment()

        status = await self.manager.get_equipment_status()

        assert len(status) == 2
        assert "liquid_handler_001" in status
        assert "reactor_001" in status
        assert status["liquid_handler_001"]["connected"] is True
        assert status["reactor_001"]["connected"] is True

    @pytest.mark.asyncio
    async def test_equipment_calibration(self):
        """Test equipment calibration."""
        self.manager.register_equipment("liquid_handler_001", self.liquid_handler)
        self.manager.register_equipment("reactor_001", self.reactor)

        await self.manager.connect_all_equipment()

        # Calibrate all equipment
        results = await self.manager.calibrate_equipment()

        assert len(results) == 2
        assert results["liquid_handler_001"] is True
        assert results["reactor_001"] is True

        # Calibrate specific equipment
        single_result = await self.manager.calibrate_equipment("liquid_handler_001")

        assert len(single_result) == 1
        assert single_result["liquid_handler_001"] is True


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_standard_labware(self):
        """Test standard labware creation."""
        labware = create_standard_labware()

        assert len(labware) > 0
        assert all(isinstance(item, LabwareDefinition) for item in labware)

        # Check for expected equipment types
        equipment_types = {item.equipment_type for item in labware}
        assert EquipmentType.LIQUID_HANDLER in equipment_types
        assert EquipmentType.REACTOR in equipment_types
        assert EquipmentType.SPECTROMETER in equipment_types

    def test_create_standard_reagents(self):
        """Test standard reagent creation."""
        reagents = create_standard_reagents()

        assert len(reagents) > 0
        assert all(isinstance(reagent, Reagent) for reagent in reagents)

        # Check for expected reagents
        reagent_names = {reagent.name for reagent in reagents}
        assert "Water" in reagent_names
        assert "Ethanol" in reagent_names

        # Check that reagents have required properties
        for reagent in reagents:
            assert reagent.name is not None
            assert reagent.cas_number is not None
            assert reagent.molecular_weight is not None

    @pytest.mark.asyncio
    async def test_setup_autonomous_lab(self):
        """Test autonomous lab setup."""
        manager = await setup_autonomous_lab()

        assert isinstance(manager, EquipmentManager)
        assert len(manager.equipment_drivers) > 0

        # Check that equipment is connected
        status = await manager.get_equipment_status()
        for equipment_id, equipment_status in status.items():
            assert equipment_status.get("connected", False) is True

    @pytest.mark.asyncio
    async def test_run_autonomous_experiment(self):
        """Test running complete autonomous experiment."""
        experimental_conditions = {
            "temperature": 80.0,
            "stirring_speed": 300.0,
            "time": 0.5,  # hours
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            summary = await run_autonomous_experiment(
                experimental_conditions, output_dir
            )

            assert isinstance(summary, dict)
            assert "protocol_id" in summary
            assert "experimental_conditions" in summary
            assert "total_actions" in summary
            assert "successful_actions" in summary
            assert "success_rate" in summary
            assert "results" in summary

            assert summary["experimental_conditions"] == experimental_conditions
            assert summary["total_actions"] > 0
            assert summary["success_rate"] >= 0

            # Check that protocol file was created
            protocol_files = list(output_dir.glob("protocol_*.json"))
            assert len(protocol_files) == 1

            # Verify protocol file content
            protocol_content = json.loads(protocol_files[0].read_text())
            assert protocol_content["protocol_id"] == summary["protocol_id"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_equipment_type_enum(self):
        """Test equipment type enumeration."""
        assert EquipmentType.LIQUID_HANDLER.value == "liquid_handler"
        assert EquipmentType.REACTOR.value == "reactor"
        assert EquipmentType.SPECTROMETER.value == "spectrometer"

    def test_protocol_format_enum(self):
        """Test protocol format enumeration."""
        assert ProtocolFormat.JSON_PROTOCOL.value == "json_protocol"
        assert ProtocolFormat.HAMILTON_VENUS.value == "hamilton_venus"
        assert ProtocolFormat.OPENTRONS.value == "opentrons"

    def test_action_type_enum(self):
        """Test action type enumeration."""
        assert ActionType.TRANSFER_LIQUID.value == "transfer_liquid"
        assert ActionType.HEAT.value == "heat"
        assert ActionType.MEASURE.value == "measure"

    def test_autonomous_lab_error(self):
        """Test autonomous lab error exception."""
        with pytest.raises(AutonomousLabError):
            raise AutonomousLabError("Test error")

    def test_equipment_communication_error(self):
        """Test equipment communication error exception."""
        with pytest.raises(EquipmentCommunicationError):
            raise EquipmentCommunicationError("Communication failed")

    def test_protocol_generation_error(self):
        """Test protocol generation error exception."""
        with pytest.raises(ProtocolGenerationError):
            raise ProtocolGenerationError("Protocol generation failed")


class TestIntegration:
    """Integration tests for the full autonomous lab pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test complete autonomous lab pipeline."""
        # Set up lab
        manager = await setup_autonomous_lab()

        # Create experimental conditions
        experimental_conditions = {
            "temperature": 85.0,
            "pressure": 2.5,
            "concentration": 0.75,
            "time": 1.5,
            "stirring_speed": 350.0,
        }

        # Generate protocol
        generator = ProtocolGenerator(ProtocolFormat.JSON_PROTOCOL)
        labware = create_standard_labware()
        reagents = create_standard_reagents()

        protocol = generator.generate_protocol(
            experimental_conditions, labware, reagents
        )

        # Execute protocol
        results = await manager.execute_protocol(protocol)

        # Verify execution
        assert len(results) > 0
        successful_results = [r for r in results if r.success]
        assert len(successful_results) > 0

        # Compile measurements
        all_measurements = {}
        for result in results:
            if result.measurements:
                all_measurements.update(result.measurements)

        assert len(all_measurements) > 0

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery during protocol execution."""
        manager = EquipmentManager()

        # Create a mock driver that fails on specific actions
        class FailingLiquidHandler(LiquidHandlerDriver):
            async def execute_action(self, action):
                if action.action_type == ActionType.TRANSFER_LIQUID:
                    # Simulate failure
                    return ExecutionResult(
                        protocol_id=action.action_id,
                        action_id=action.action_id,
                        success=False,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        actual_parameters=action.parameters,
                        error_message="Simulated transfer failure",
                    )
                else:
                    return await super().execute_action(action)

        labware_def = LabwareDefinition(
            name="Failing Handler",
            equipment_type=EquipmentType.LIQUID_HANDLER,
            model="FH001",
            manufacturer="TestCorp",
        )

        failing_handler = FailingLiquidHandler(labware_def)
        manager.register_equipment("liquid_handler_001", failing_handler)

        await manager.connect_all_equipment()

        # Create protocol with transfer action
        actions = [
            ExperimentalAction(
                action_id="action_001",
                action_type=ActionType.TRANSFER_LIQUID,
                equipment="liquid_handler_001",
                parameters={"volume": 100.0},
            )
        ]

        protocol = ExperimentalProtocol(
            protocol_id="test_protocol",
            name="Error Test Protocol",
            description="Test error handling",
            actions=actions,
            labware=[],
            reagents=[],
            vessels=[],
        )

        # Execute protocol and verify error handling
        results = await manager.execute_protocol(protocol)

        assert len(results) == 1
        assert results[0].success is False
        assert "Simulated transfer failure" in results[0].error_message

    def test_protocol_format_compatibility(self):
        """Test compatibility across different protocol formats."""
        experimental_conditions = {"temperature": 80.0, "time": 2.0}

        labware = create_standard_labware()[:1]
        reagents = create_standard_reagents()[:1]

        formats_to_test = [
            ProtocolFormat.JSON_PROTOCOL,
            ProtocolFormat.XML_PROTOCOL,
            ProtocolFormat.PYTHON_SCRIPT,
            ProtocolFormat.HAMILTON_VENUS,
            ProtocolFormat.OPENTRONS,
        ]

        # Add YAML if available
        if YAML_AVAILABLE:
            formats_to_test.append(ProtocolFormat.YAML_PROTOCOL)

        for format_type in formats_to_test:
            generator = ProtocolGenerator(format_type)

            protocol = generator.generate_protocol(
                experimental_conditions, labware, reagents
            )

            # Generate protocol content
            content = generator.export_protocol(protocol)

            assert isinstance(content, str)
            assert len(content) > 0

            # Verify format-specific content
            if format_type == ProtocolFormat.JSON_PROTOCOL:
                json.loads(content)  # Should not raise exception
            elif format_type == ProtocolFormat.XML_PROTOCOL:
                ET.fromstring(content)  # Should not raise exception
            elif format_type == ProtocolFormat.PYTHON_SCRIPT:
                assert "async def run_protocol():" in content
            elif format_type == ProtocolFormat.HAMILTON_VENUS:
                assert "Venus::Initialize();" in content
            elif format_type == ProtocolFormat.OPENTRONS:
                assert "from opentrons import protocol_api" in content


if __name__ == "__main__":
    pytest.main([__file__])
